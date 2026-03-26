#pragma once
#include <vector>
#include "types.h"

std::vector<DetectResult> decode_rtdetr_output(float* boxes_data, float* scores_data, int num_boxes, int orig_w, int orig_h, float conf_thres);
void draw_results(cv::Mat& image, const std::vector<DetectResult>& results);