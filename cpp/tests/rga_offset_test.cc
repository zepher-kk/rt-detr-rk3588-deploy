// Verify RGA can write a letterboxed image directly into an offset region of a
// DMA buffer (so the RKNN input can be used with rknn_set_io_mem later).
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>
#include <im2d.hpp>
#include <rga.h>

#include "drm_alloc.h"
#include "types.h"

int main(int argc, char** argv) {
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) return 1;

    const int W = 640, H = 640;
    const float scale = std::min((float)W / img.cols, (float)H / img.rows);
    const int nw = (int)std::lround(img.cols * scale);
    const int nh = (int)std::lround(img.rows * scale);
    const int left = (W - nw) / 2, top = (H - nh) / 2;
    printf("src %dx%d -> %dx%d at (%d,%d)\n", img.cols, img.rows, nw, nh, left, top);

    DmaBufferPool pool(W, H, RK_FORMAT_BGR_888, 2, false);
    DmaBufferPtr dst = pool.alloc();
    if (!dst) { printf("dma alloc failed\n"); return 1; }
    printf("dma fd=%d ptr=%p stride=%zu size=%zu\n", dst->fd, dst->ptr, dst->stride, dst->size);
    memset(dst->ptr, 0, dst->stride * H);

    for (int variant = 0; variant < 2; ++variant) {
        memset(dst->ptr, 0, dst->stride * H);
        uint8_t* off = (uint8_t*)dst->ptr + (size_t)top * dst->stride + (size_t)left * 3;

        rga_buffer_t s = wrapbuffer_virtualaddr_t((void*)img.data, img.cols, img.rows,
                                                  img.cols, img.rows, RK_FORMAT_BGR_888);
        rga_buffer_t d;
        if (variant == 0)
            d = wrapbuffer_virtualaddr_t(off, nw, nh, W, nh, RK_FORMAT_BGR_888);
        else
            d = wrapbuffer_virtualaddr_t(off, nw, nh, W, H, RK_FORMAT_BGR_888);

        IM_STATUS st = imresize(s, d);
        printf("variant %d imresize=%d", variant, (int)st);
        if (st == IM_STATUS_SUCCESS) {
            st = imcvtcolor(d, d, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
            printf(" imcvtcolor=%d", (int)st);
        }
        printf("\n");
        if (st != IM_STATUS_SUCCESS) continue;

        // compare against the CPU reference
        cv::Mat ref;
        cv::resize(img, ref, cv::Size(nw, nh));
        cv::cvtColor(ref, ref, cv::COLOR_BGR2RGB);
        long diff = 0, n = 0;
        for (int y = 0; y < nh; y += 37)
            for (int x = 0; x < nw; x += 41)
                for (int c = 0; c < 3; ++c) {
                    int a = ((uint8_t*)dst->ptr)[(size_t)(top + y) * dst->stride +
                                                 (size_t)(left + x) * 3 + c];
                    int b = ref.at<cv::Vec3b>(y, x)[c];
                    diff += std::abs(a - b);
                    ++n;
                }
        printf("  sample mean abs diff = %.2f (n=%ld)\n", (double)diff / n, n);
        // padding must stay zero
        printf("  corner px = %d %d %d\n",
               ((uint8_t*)dst->ptr)[0],
               ((uint8_t*)dst->ptr)[1],
               ((uint8_t*)dst->ptr)[2]);
    }
    return 0;
}
