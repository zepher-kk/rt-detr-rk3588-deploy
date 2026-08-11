#include "postprocess.h"
#include <cmath>
#include <algorithm>

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
	for (const auto& res : results)
	{
		cv::RNG rng(res.class_id + 100);
		cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		cv::rectangle(image, res.box, color, 2);
		char label[256];
		snprintf(label, sizeof(label), "%s %.2f", CLASSES[res.class_id].c_str(), res.score);
		cv::putText(image, label, cv::Point(res.box.x, res.box.y - 5),
		            cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
	}
}
