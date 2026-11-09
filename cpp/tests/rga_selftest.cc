#include <chrono>
#include <cstdio>

#include <opencv2/opencv.hpp>

#include "rga_utils.h"

int main(int argc, char** argv) {
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) { printf("cannot read\n"); return 1; }
    printf("src %dx%d\n", img.cols, img.rows);
    fflush(stdout);

    RgaPreprocessor p;
    printf("init...\n"); fflush(stdout);
    p.init(8);
    printf("rga_active=%d\n", (int)p.rga_active()); fflush(stdout);

    for (int i = 0; i < 5; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        auto d = p.preprocess(img);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        printf("iter %d buf=%p stride=%zu fmt=%d %.3f ms\n", i, d ? d->ptr : nullptr,
               d ? d->stride : 0, d ? d->format : -1, ms);
        fflush(stdout);
        if (!d) break;
        if (i == 0) {
            // compare with CPU reference on a few pixels
            cv::Mat ref;
            cv::resize(img, ref, cv::Size(INPUT_W, INPUT_H));
            cv::cvtColor(ref, ref, cv::COLOR_BGR2RGB);
            uint8_t* a = (uint8_t*)d->ptr;
            int diff = 0;
            for (int y = 0; y < INPUT_H; y += 97)
                for (int x = 0; x < INPUT_W; x += 89)
                    for (int c = 0; c < 3; ++c) {
                        int va = a[(size_t)y * d->stride + x * 3 + c];
                        int vb = ref.at<cv::Vec3b>(y, x)[c];
                        diff += std::abs(va - vb);
                    }
            printf("sample abs diff sum = %d\n", diff);
            fflush(stdout);
        }
    }
    return 0;
}
