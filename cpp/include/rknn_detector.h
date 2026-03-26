#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "rknn_api.h"

class RKNNDetector {
public:
    RKNNDetector();
    ~RKNNDetector();

    bool init(const std::string& model_path);
    bool infer_only(const cv::Mat& preprocessed_img, std::vector<float>& out_boxes, std::vector<float>& out_logits, int& num_boxes);
    bool detect(const cv::Mat& orig_img, float conf_thres, std::vector<DetectResult>& results);

private:
    rknn_context ctx_;
    unsigned char* model_data_;
    bool is_init_;
    unsigned char* load_model(const char* filename, int* model_size);
};