#include "postprocess.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cctype>

// ============================================================================
// 类别固定调色板（BGR，10 类 VisDrone）
// 按类别固定取色（确定性），便于跨帧观察同一类别，也省去逐框取色开销。
// ============================================================================
namespace
{
const cv::Scalar kClassColors[NUM_CLASSES] = {
	cv::Scalar(0, 0, 255),      // Pedestrian     - 红
	cv::Scalar(0, 255, 0),      // People         - 绿
	cv::Scalar(255, 0, 0),      // Bicycle        - 蓝
	cv::Scalar(0, 165, 255),    // Car            - 橙
	cv::Scalar(0, 255, 255),    // Van            - 黄
	cv::Scalar(128, 0, 128),    // Truck          - 紫
	cv::Scalar(203, 192, 255),  // Tricycle       - 粉
	cv::Scalar(255, 0, 255),    // Awning-tricycle- 品红
	cv::Scalar(255, 128, 0),    // Bus            - 青
	cv::Scalar(255, 255, 0)     // Motor          - 天蓝
};
}   // namespace

// ============================================================================
// 按类置信度阈值：配置解析与预设档位
// ============================================================================
namespace
{
// 按类标定向量（VisDrone-2019-DET val 全量 548 张、IoU≥0.5、F1 最大准则）
const float kRecallProfile[NUM_CLASSES] = {
	-0.40f,   // Pedestrian
	-0.70f,   // People
	-0.90f,   // Bicycle
	0.00f,    // Car
	-0.25f,   // Van
	-0.30f,   // Truck
	-0.70f,   // Tricycle
	-0.80f,   // Awning-tricycle
	-0.05f,   // Bus
	-0.55f    // Motor
};

std::string trim(const std::string& s)
{
	size_t b = 0, e = s.size();
	while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
	while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
	return s.substr(b, e - b);
}

// 严格数值解析：要求整个字符串（去除首尾空白后）都参与转换，拒绝 "1.2x"、"0x1" 等
bool parse_float_strict(const std::string& s, float& out)
{
	std::string t = trim(s);
	if (t.empty()) return false;
	char* end = nullptr;
	out = std::strtof(t.c_str(), &end);
	return end != nullptr && *end == '\0';
}

bool parse_int_strict(const std::string& s, int& out)
{
	std::string t = trim(s);
	if (t.empty()) return false;
	char* end = nullptr;
	long v = std::strtol(t.c_str(), &end, 10);
	if (end == nullptr || *end != '\0') return false;
	out = (int)v;
	return true;
}
}   // namespace

bool parse_conf_class(const std::string& spec, ConfConfig& cfg, std::string* err)
{
	auto fail = [&](const std::string& m)
	{
		if (err) *err = m;
		return false;
	};
	if (trim(spec).empty()) return fail("--conf-class 规格为空");

	std::string cur;
	bool any = false;
	// 末尾补一个分隔符，让循环内统一处理最后一项，避免额外收尾分支
	std::string input = spec + ",";
	for (char ch : input)
	{
		if (ch == ',' || ch == ';')
		{
			std::string item = trim(cur);
			cur.clear();
			if (item.empty()) continue;
			size_t colon = item.find(':');
			if (colon == std::string::npos) return fail("缺少 ':' 分隔符: " + item);
			int cls = -1;
			float thr = 0.0f;
			if (!parse_int_strict(item.substr(0, colon), cls)) return fail("类别非法: " + item);
			if (!parse_float_strict(item.substr(colon + 1), thr)) return fail("阈值非法: " + item);
			if (cls < 0 || cls >= NUM_CLASSES) return fail("类别越界（需 0~9）: " + item);
			cfg.per_class[cls] = thr;
			any = true;
			continue;
		}
		cur.push_back(ch);
	}
	if (!any) return fail("未解析到任何 <class:thr> 项");
	cfg.per_class_set = true;
	return true;
}

bool apply_conf_profile(const std::string& name, float base, ConfConfig& cfg, std::string* err)
{
	if (name == "balanced")
	{
		cfg.reset(base);
		cfg.per_class_set = false;      // 等价于原全局阈值行为
		return true;
	}
	if (name == "recall")
	{
		cfg.reset(base);
		for (int i = 0; i < NUM_CLASSES; ++i) cfg.per_class[i] = kRecallProfile[i];
		cfg.per_class_set = true;
		return true;
	}
	if (err) *err = "未知档位（支持 balanced / recall）: " + name;
	return false;
}

// ============================================================================
// 自适应画框样式
// ============================================================================
DrawStyle compute_draw_style(int frame_w, int frame_h)
{
	DrawStyle s;
	if (frame_w <= 0 || frame_h <= 0) return s;

	// 以 720p 为基准：2px 线宽 / 0.6 字号；按短边比例线性缩放并限幅
	double scale = std::min(std::max(std::min(frame_w, frame_h) / 720.0, 0.5), 2.0);
	s.thickness      = std::max(1, (int)std::lround(2.0 * scale));
	s.font_scale     = std::min(std::max(0.6 * scale, 0.35), 1.2);
	s.font_thickness = std::max(1, (int)std::lround(1.5 * scale));
	s.small_box_th   = 26.0 * scale;   // 小目标阈值随分辨率等比放大
	return s;
}

// 解码 RT-DETR 输出，对每个框取最大分数，过滤低置信度
std::vector<DetectResult> decode_rtdetr_output(float* boxes_data,
        float* scores_data,
        int num_boxes,
        int orig_w, int orig_h,
        float conf_thres,
        int /* num_classes */,
        const ConfConfig* conf_cfg)
{
	const int nc = NUM_CLASSES;   // 强制使用 10 类
	std::vector<DetectResult> results;
	results.reserve(num_boxes / 4);

	for (int i = 0; i < num_boxes; ++i)
	{
		float* box_ptr = boxes_data + i * 4;
		float* score_ptr = scores_data + i * nc;

		// 找最大分数（直接比较，模型已输出概率）
		float max_score = -1.0f;
		int max_class_id = -1;
		for (int c = 0; c < nc; ++c)
		{
			if (score_ptr[c] > max_score)
			{
				max_score = score_ptr[c];
				max_class_id = c;
			}
		}

		// 按类阈值（未启用按类阈值时等于全局 conf_thres）
		const float thr = conf_cfg ? conf_cfg->threshold_for(max_class_id) : conf_thres;
		if (max_score < thr) continue;

		// 反归一化坐标（乘以原图尺寸）
		float cx = box_ptr[0] * orig_w;
		float cy = box_ptr[1] * orig_h;
		float w  = box_ptr[2] * orig_w;
		float h  = box_ptr[3] * orig_h;

		int x_min = std::round(cx - w / 2.0f);
		int y_min = std::round(cy - h / 2.0f);
		int x_max = std::round(cx + w / 2.0f);
		int y_max = std::round(cy + h / 2.0f);

		// 裁剪到图像边界
		x_min = std::max(0, std::min(x_min, orig_w));
		y_min = std::max(0, std::min(y_min, orig_h));
		x_max = std::max(0, std::min(x_max, orig_w));
		y_max = std::max(0, std::min(y_max, orig_h));

		if (x_max <= x_min || y_max <= y_min) continue;

		DetectResult r;
		r.class_id = max_class_id;
		r.score = max_score;
		r.box = cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min);
		results.push_back(r);
	}
	return results;
}

// 绘制检测框和标签（颜色按类别固定，便于观察）
void draw_results(cv::Mat& image, const std::vector<DetectResult>& results)
{
	if (image.empty() || results.empty()) return;

	DrawStyle style = compute_draw_style(image.cols, image.rows);

	// 本帧内缓存各类别文字度量，避免逐框重复 getTextSize
	cv::Size class_text_size[NUM_CLASSES];
	bool     text_measured[NUM_CLASSES] = {false};

	for (const auto& res : results)
	{
		if (res.class_id < 0 || res.class_id >= NUM_CLASSES) continue;
		if (res.box.width <= 0 || res.box.height <= 0) continue;

		const cv::Scalar& color = kClassColors[res.class_id];
		const cv::Rect&   box   = res.box;

		// 小目标（如远景密集车辆/行人）：只画细框，不画文字，避免遮挡目标本身
		bool is_small = (std::min(box.width, box.height) < style.small_box_th);
		int thickness = is_small ? 1 : style.thickness;
		cv::rectangle(image, box, color, thickness);
		if (is_small) continue;

		// 首次遇到该类别时计算文字尺寸（一次/类别/帧）
		if (!text_measured[res.class_id])
		{
			const std::string& name = CLASSES[res.class_id];
			class_text_size[res.class_id] = cv::getTextSize(name, cv::FONT_HERSHEY_SIMPLEX,
			                                                style.font_scale, style.font_thickness, nullptr);
			text_measured[res.class_id] = true;
		}
		cv::Size ts = class_text_size[res.class_id];
		int text_w = ts.width + (int)std::lround(34.0 * style.font_scale);  // 预留 " 0.99" 宽度
		int text_h = ts.height;

		// 优先画在框上方；顶部空间不足时画在框内顶部；框内也放不下则跳过文字
		int baseline_y = box.y - 4;
		bool inside = false;
		if (baseline_y - text_h < 0)
		{
			baseline_y = box.y + text_h + 2;
			inside = true;
		}
		if (inside && text_h > box.height) continue;

		int x = std::max(0, box.x);
		if (x + text_w > image.cols) x = std::max(0, image.cols - text_w);

		char label[64];
		snprintf(label, sizeof(label), "%s %.2f", CLASSES[res.class_id].c_str(), res.score);
		cv::putText(image, label, cv::Point(x, baseline_y),
		            cv::FONT_HERSHEY_SIMPLEX, style.font_scale, color, style.font_thickness);
	}
}
