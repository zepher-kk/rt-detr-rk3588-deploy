#include "postprocess.h"
#include <cmath>
#include <algorithm>

std::vector<DetectResult> decode_rtdetr_output(float* boxes_data, float* scores_data, int num_boxes, int orig_w, int orig_h, float conf_thres) {
    std::vector<DetectResult> results;
    for (int i = 0; i < num_boxes; ++i) {
        float* box_ptr = boxes_data + i * 4;
        float* score_ptr = scores_data + i * NUM_CLASSES;

        float max_score = -1.0f;
        int max_class_id = -1;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            if (score_ptr[c] > max_score) {
                max_score = score_ptr[c];
                max_class_id = c;
            }
        }

        if (max_score > conf_thres) {
            float cx = box_ptr[0] * orig_w;
            float cy = box_ptr[1] * orig_h;
            float w  = box_ptr[2] * orig_w;
            float h  = box_ptr[3] * orig_h;

            int x_min = std::round(cx - w / 2.0f);
            int y_min = std::round(cy - h / 2.0f);
            int x_max = std::round(cx + w / 2.0f);
            int y_max = std::round(cy + h / 2.0f);

            x_min = std::max(0, std::min(x_min, orig_w));
            y_min = std::max(0, std::min(y_min, orig_h));
            x_max = std::max(0, std::min(x_max, orig_w));
            y_max = std::max(0, std::min(y_max, orig_h));

            results.push_back({max_class_id, max_score, cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min)});
        }
    }
    return results;
}

void draw_results(cv::Mat& image, const std::vector<DetectResult>& results) {
    for (const auto& res : results) {
        cv::RNG rng(res.class_id + 100); 
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::rectangle(image, res.box, color, 2);
        char label[256];
        snprintf(label, sizeof(label), "%s %.2f", CLASSES[res.class_id].c_str(), res.score);
        cv::putText(image, label, cv::Point(res.box.x, res.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}