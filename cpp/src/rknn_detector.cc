#include "rknn_detector.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

RKNNDetector::RKNNDetector() : ctx_(0), model_data_(nullptr), is_init_(false) {}

RKNNDetector::~RKNNDetector() {
    if (ctx_ > 0) rknn_destroy(ctx_);
    if (model_data_) free(model_data_);
}

unsigned char* RKNNDetector::load_model(const char* filename, int* model_size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return nullptr;
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char* data = (unsigned char*)malloc(size);
    if (data) fread(data, 1, size, fp);
    fclose(fp);
    *model_size = size;
    return data;
}

bool RKNNDetector::init(const std::string& model_path) {
    int model_size;
    model_data_ = load_model(model_path.c_str(), &model_size);
    if (!model_data_) return false;
    int ret = rknn_init(&ctx_, model_data_, model_size, 0, NULL);
    if (ret < 0) return false;
    is_init_ = true;
    return true;
}

bool RKNNDetector::detect(const cv::Mat& orig_img, float conf_thres, std::vector<DetectResult>& results) { return false; }

bool RKNNDetector::infer_only(const cv::Mat& preprocessed_img, std::vector<float>& out_boxes, std::vector<float>& out_logits, int& num_boxes) {
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    
    // 🌟 核心修复：把 RKNN_TENSOR_FLOAT32 改成 RKNN_TENSOR_UINT8，100% 对齐 Python
    inputs[0].type = RKNN_TENSOR_UINT8; 
    
    inputs[0].size = preprocessed_img.total() * preprocessed_img.elemSize();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = preprocessed_img.data;
    
    // ... 后面的代码都不用动 ...

    if (rknn_inputs_set(ctx_, 1, inputs) < 0) return false;
    if (rknn_run(ctx_, NULL) < 0) return false;
    
    rknn_input_output_num io_num;
    rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) outputs[i].want_float = 1; 
    rknn_outputs_get(ctx_, io_num.n_output, outputs, NULL);

    int boxes_idx = -1, logits_idx = -1;
    for (int i = 0; i < io_num.n_output; i++) {
        rknn_tensor_attr out_attr;
        out_attr.index = i;
        rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
        if (out_attr.n_elems == 300 * 4) boxes_idx = i;
        if (out_attr.n_elems == 300 * NUM_CLASSES) logits_idx = i;
    }

    if (boxes_idx != -1 && logits_idx != -1) {
        float* b_data = (float*)outputs[boxes_idx].buf;
        float* l_data = (float*)outputs[logits_idx].buf;
        out_boxes.assign(b_data, b_data + (300 * 4));
        out_logits.assign(l_data, l_data + (300 * NUM_CLASSES));
        num_boxes = 300;
    }

    rknn_outputs_release(ctx_, io_num.n_output, outputs);
    return true;
}