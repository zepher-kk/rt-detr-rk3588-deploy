#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#ifndef RTDETR_INPUT_SIZE
#define RTDETR_INPUT_SIZE 640
#endif

constexpr int INPUT_W = RTDETR_INPUT_SIZE;
constexpr int INPUT_H = RTDETR_INPUT_SIZE;
constexpr int NUM_CLASSES = 10;

struct DmaBuffer {
    int      fd     = -1;
    void*    ptr    = nullptr;
    size_t   size   = 0;
    int      width  = 0;
    int      height = 0;
    int      format = 0;
    size_t   stride = 0;
    uint32_t handle = 0;
    int      drm_fd = -1;
    bool     heap   = false;   // true: malloc'd cached buffer (no DRM/PRIME)
    bool     via_rga = false;  // true: produced by the RGA hardware path

    DmaBuffer() = default;
    ~DmaBuffer();
    DmaBuffer(const DmaBuffer&) = delete;
    DmaBuffer& operator=(const DmaBuffer&) = delete;
    DmaBuffer(DmaBuffer&& o) noexcept;
    DmaBuffer& operator=(DmaBuffer&& o) noexcept;
};
using DmaBufferPtr = std::shared_ptr<DmaBuffer>;

struct FrameBundle {
    int          frame_id = -1;
    cv::Mat      src;                 // shared ref (no deep copy in benchmark mode)
    DmaBufferPtr input_buf;           // 640x640 RGB DMA
    std::vector<float> pred_boxes;
    std::vector<float> pred_logits;
    int          num_boxes   = 0;
    int          num_classes = NUM_CLASSES;
    int64_t      t_enqueue = 0, t_pre_done = 0, t_npu_done = 0, t_post_done = 0;
    int64_t      t_pre_start = 0, t_npu_start = 0, t_post_start = 0;
};
using FrameBundlePtr = std::shared_ptr<FrameBundle>;

struct DetectResult {
    int      class_id = -1;
    float    score    = 0.f;
    cv::Rect box;
};

int rga_format_bpp(int format);
size_t drm_aligned_stride(int width, int bpp);
int64_t now_us();
