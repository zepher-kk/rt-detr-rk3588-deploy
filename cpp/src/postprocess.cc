#include "postprocess.h"
#include <cmath>
#include <algorithm>
#include <cstdio>

// ============================================================================
// 类别固定调色板（BGR，10 类 VisDrone）
// 原实现每框用 cv::RNG 随机取色（颜色逐框跳动、每帧变化），
// 改为按类别固定的确定性调色板：便于观察/调试，也省去逐框 RNG 开销。
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
        int /* num_classes */)
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

		if (max_score < conf_thres) continue;

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
