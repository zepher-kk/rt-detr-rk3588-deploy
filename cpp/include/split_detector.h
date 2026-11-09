#pragma once

#include <string>
#include <vector>

#include "rknn_api.h"
#include "types.h"

// Two-model pipeline: backbone+neck (INT8) -> decoder (FP16).
//
// The backbone outputs three neck feature maps as INT8 in the native NC1HWC2
// layout; the decoder expects the same maps as FP16 in its own native NC1HWC2
// layout. The hand-off therefore dequantises (per-tensor scale/zp) and
// re-blocks 16-channel groups into 8-channel groups, straight into the decoder
// input memory (no float32 round trip).
class SplitDetector {
  public:
    SplitDetector() = default;
    ~SplitDetector();

    bool init(const std::string& backbone_path, const std::string& decoder_path,
              rknn_core_mask mask);
    bool infer(const DmaBufferPtr& input, FrameBundle& out);

  private:
    bool load_and_query(const std::string& path, rknn_context& ctx,
                        unsigned char*& data, int& n_in, int& n_out);
    bool prepare_buffers();

    rknn_context a_ctx_ = 0, b_ctx_ = 0;
    unsigned char* a_data_ = nullptr;
    unsigned char* b_data_ = nullptr;
    int a_in_ = 0, a_out_ = 0, b_in_ = 0, b_out_ = 0;
    bool ready_ = false;

    rknn_tensor_attr a_in_attr_{}, a_native_in_{};
    std::vector<rknn_tensor_attr> a_out_attrs_, a_native_out_;
    std::vector<rknn_tensor_attr> b_in_attrs_, b_native_in_;
    std::vector<rknn_tensor_attr> b_out_attrs_;

    rknn_tensor_mem* a_in_mem_ = nullptr;
    std::vector<rknn_tensor_mem*> b_in_mem_;
    std::vector<std::vector<int8_t>> a_out_raw_;
    std::vector<int> feat_h_, feat_w_, feat_c_;

    int combined_idx_ = -1;
    int queries_ = 0;
    int boxes_elems_ = 0, logits_elems_ = 0;
    std::vector<std::vector<float>> b_out_buf_;
    rknn_core_mask mask_ = RKNN_NPU_CORE_AUTO;
    int calls_ = 0;
    bool io_mem_ready_ = false;
};
