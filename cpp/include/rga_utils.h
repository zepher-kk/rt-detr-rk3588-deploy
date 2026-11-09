#pragma once

#include "drm_alloc.h"
#include "types.h"

#include <memory>

// RGA preprocessing: cv::Mat(BGR) -> 640x640 RGB DMA buffer.
// Tries RGA virt->DMA first, falls back to CPU resize+cvtColor.
class RgaPreprocessor {
  public:
    ~RgaPreprocessor();

    bool init(size_t pool_capacity = 8);
    DmaBufferPtr preprocess(const cv::Mat& src);
    void release();

    bool rga_active() const { return rga_ok_; }
    bool use_heap() const;

  private:
    std::unique_ptr<DmaBufferPool> dst_pool_;
    bool inited_ = false;
    bool rga_ok_ = false;
    bool use_heap_ = true;
    int  last_w_ = -1, last_h_ = -1, last_l_ = -1, last_t_ = -1;
    std::vector<uint8_t> tmp_;
};
