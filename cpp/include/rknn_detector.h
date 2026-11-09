#pragma once

#include "types.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "rknn_api.h"

class RKNNDetector {
  public:
    RKNNDetector() = default;
    ~RKNNDetector();

    bool init(const std::string& model_path, rknn_core_mask mask);
    bool infer(const DmaBufferPtr& input, FrameBundle& out);

  private:
    bool query_io();

    rknn_context ctx_ = 0;
    unsigned char* model_data_ = nullptr;
    int n_input_ = 0, n_output_ = 0;
    bool ready_ = false;

    std::vector<rknn_tensor_attr> out_attrs_;
    std::vector<std::vector<float>> out_buffers_;
    int boxes_idx_ = -1, logits_idx_ = -1;
    int combined_idx_ = -1;
    int queries_ = 0;
    int boxes_elems_ = 0, logits_elems_ = 0;
    rknn_core_mask mask_ = RKNN_NPU_CORE_AUTO;
    rknn_tensor_attr input_attr_{};
    rknn_tensor_attr native_in_attr_{};
    rknn_tensor_mem* in_mem_ = nullptr;
    bool in_mem_ready_ = false;
    bool use_io_mem_ = true;
    bool io_mem_ok_ = false;
    int64_t t_set_ = 0, t_run_ = 0, t_get_ = 0, calls_ = 0;
};
