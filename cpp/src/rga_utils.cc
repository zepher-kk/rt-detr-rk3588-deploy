#include "rga_utils.h"

#include <im2d.hpp>
#include <rga.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// librga 2.x is not safe under concurrent calls from several pre workers.
static std::mutex g_rga_mtx;

RgaPreprocessor::~RgaPreprocessor() { release(); }

void RgaPreprocessor::release() {
    dst_pool_.reset();
    inited_ = false;
    rga_ok_ = false;
}

bool RgaPreprocessor::init(size_t pool_capacity) {
    if (inited_) return true;
    const char* dst_mode = getenv("RTDETR_DST");
    use_heap_ = !(dst_mode && std::string(dst_mode) == "dma");
    IM_STATUS st = imcheckHeader(RGA_CURRENT_API_HEADER_VERSION);
    // IM_STATUS_SUCCESS == 1, IM_STATUS_NOERROR == 2 on librga 2.x
    rga_ok_ = (st == IM_STATUS_SUCCESS || st == IM_STATUS_NOERROR);
    if (getenv("RTDETR_NO_RGA")) rga_ok_ = false;  // A/B testing switch
    if (!rga_ok_) {
        fprintf(stderr, "[rga] imcheckHeader=%d -> CPU fallback only\n", (int)st);
    } else {
        fprintf(stderr, "[rga] initialized API %s\n", RGA_API_VERSION);
    }
    dst_pool_ = std::make_unique<DmaBufferPool>(INPUT_W, INPUT_H,
                                                RK_FORMAT_BGR_888, pool_capacity,
                                                use_heap_);
    inited_ = true;
    return true;
}

bool RgaPreprocessor::use_heap() const { return use_heap_; }

DmaBufferPtr RgaPreprocessor::preprocess(const cv::Mat& src) {
    if (!inited_ || src.empty()) return nullptr;

    // letterbox: keep aspect ratio, pad with zeros (same convention as the
    // Python reference pipeline)
    const float scale = std::min((float)INPUT_W / src.cols, (float)INPUT_H / src.rows);
    const int nw = std::max(1, (int)std::lround(src.cols * scale));
    const int nh = std::max(1, (int)std::lround(src.rows * scale));
    const int left = (INPUT_W - nw) / 2;
    const int top = (INPUT_H - nh) / 2;

    // ---- RGA path: write the scaled image straight into the canvas ----
    if (rga_ok_ && src.isContinuous()) {
        std::lock_guard<std::mutex> lk(g_rga_mtx);
        DmaBufferPtr dst = dst_pool_->alloc();
        if (dst) {
            // Padding is zeroed once per buffer. If the letterbox geometry
            // changes (different aspect ratio), clear the whole canvas so no
            // stale pixels leak into the pad area.
            if (last_w_ != nw || last_h_ != nh || last_l_ != left || last_t_ != top) {
                memset(dst->ptr, 0, dst->stride * INPUT_H);
                last_w_ = nw; last_h_ = nh; last_l_ = left; last_t_ = top;
            }

            uint8_t* off = (uint8_t*)dst->ptr + (size_t)top * dst->stride +
                           (size_t)left * 3;
            rga_buffer_t s = wrapbuffer_virtualaddr_t((void*)src.data, src.cols,
                                                      src.rows, src.cols,
                                                      src.rows, RK_FORMAT_BGR_888);
            // hstride = nh: the row pitch is INPUT_W pixels, the sub-image is nh tall
            rga_buffer_t d = wrapbuffer_virtualaddr_t(off, nw, nh, INPUT_W, nh,
                                                      RK_FORMAT_BGR_888);
            IM_STATUS st = IM_STATUS_FAILED;
            if (s.vir_addr != nullptr && d.vir_addr != nullptr) {
                st = imresize(s, d);
                if (st == IM_STATUS_SUCCESS)
                    st = imcvtcolor(d, d, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
            }
            if (st == IM_STATUS_SUCCESS) {
                dst->format = RK_FORMAT_RGB_888;
                dst->via_rga = true;
                return dst;
            }
            static bool warned = false;
            if (!warned) {
                fprintf(stderr, "[rga] pipeline failed (%d), CPU fallback\n", (int)st);
                warned = true;
            }
            dst.reset();
        }
    }

    // ---- CPU fallback: letterbox ----
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    DmaBufferPtr dst = dst_pool_->alloc();
    if (!dst) return nullptr;
    memset(dst->ptr, 0, dst->stride * INPUT_H);
    uint8_t* d = (uint8_t*)dst->ptr + (size_t)top * dst->stride + (size_t)left * 3;
    const uint8_t* s = resized.data;
    const size_t row = (size_t)nw * 3;
    for (int y = 0; y < nh; ++y) {
        memcpy(d, s, row);
        d += dst->stride;
        s += resized.step;
    }
    dst->format = RK_FORMAT_RGB_888;
    return dst;
}
