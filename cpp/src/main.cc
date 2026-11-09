#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

#include "pipeline.h"

static void print_breakdown(int64_t n,
                            int64_t wait_raw, int64_t pre, int64_t wait_npu,
                            int64_t npu, int64_t wait_post, int64_t post,
                            int64_t e2e, double p50, double p95, double mx) {
    if (n <= 0) return;
    printf("[stage] wait_raw %.2f | pre %.2f | wait_npu %.2f | npu %.2f | "
           "wait_post %.2f | post %.2f ms\n",
           wait_raw / 1000.0 / n, pre / 1000.0 / n, wait_npu / 1000.0 / n,
           npu / 1000.0 / n, wait_post / 1000.0 / n, post / 1000.0 / n);
    printf("[stage] e2e mean %.2f ms | p50 %.2f | p95 %.2f | max %.2f\n",
           e2e / 1000.0 / n, p50, p95, mx);
}

static void usage(const char* argv0) {    fprintf(stderr,
            "usage: %s -m model.rknn [-i image] [options]\n"
            "  -m, --model      RKNN model path (required)\n"
            "  -i, --image      input image (default: synthesize)\n"
            "      --image-list image list file, one path per line\n"
            "  -v, --video      input video (software decode via OpenCV/FFmpeg)\n"
            "  -F, --fps        pacing for video playback (0 = decode as fast as possible)\n"
            "  -n, --npu        NPU workers (default: 3)\n"
            "  -p, --pre        preprocess workers (default: 2)\n"
            "  -P, --post       postprocess workers (default: 1)\n"
            "  -q, --queue      queue capacity (default: 16)\n"
            "  -N, --frames     measured frames (default: 200)\n"
            "      --warmup     warmup frames (default: 20)\n"
            "      --cores      split|auto|all (default: split)\n"
            "  -c, --conf       confidence threshold (default: 0.45)\n"
            "      --latency    only run 1 frame, report end-to-end latency\n",
            argv0);
}

int main(int argc, char** argv) {
    std::string model, image, image_list, video;
    int n_npu = 3, n_pre = 2, n_post = 1, frames = 200, warmup = 20;
    size_t qcap = 16;
    std::string cores = "split";
    float conf = 0.45f;
    float pace_fps = 0.f;   // 0 = decode as fast as possible
    bool latency = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "-m" || a == "--model") model = next();
        else if (a == "-i" || a == "--image") image = next();
        else if (a == "--image-list") image_list = next();
        else if (a == "-v" || a == "--video") video = next();
        else if (a == "-F" || a == "--fps") pace_fps = std::stof(next());
        else if (a == "-n" || a == "--npu") n_npu = std::stoi(next());
        else if (a == "-p" || a == "--pre") n_pre = std::stoi(next());
        else if (a == "-P" || a == "--post") n_post = std::stoi(next());
        else if (a == "-q" || a == "--queue") qcap = (size_t)std::stoi(next());
        else if (a == "-N" || a == "--frames") frames = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--cores") cores = next();
        else if (a == "-c" || a == "--conf") conf = std::stof(next());
        else if (a == "--latency") latency = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }
    if (model.empty()) { usage(argv[0]); return 1; }

    cv::Mat img;
    std::vector<cv::Mat> imgs;

    if (!video.empty()) {
        // ---- video input: reader thread pushes frames, pipeline applies backpressure ----
        cv::VideoCapture cap(video);
        if (!cap.isOpened()) {
            fprintf(stderr, "cannot open video %s\n", video.c_str());
            return 1;
        }
        const double src_fps = cap.get(cv::CAP_PROP_FPS);
        const int64_t src_count = (int64_t)cap.get(cv::CAP_PROP_FRAME_COUNT);
        fprintf(stderr, "[main] video %s: %.1f fps, %lld frames\n", video.c_str(), src_fps,
                (long long)src_count);

        PipelineConfig cfg;
        cfg.model_path = model;
        cfg.num_pre = n_pre;
        cfg.num_npu = n_npu;
        cfg.num_post = n_post;
        cfg.queue_cap = qcap;
        cfg.core_mode = cores;
        cfg.conf_thres = conf;

        Pipeline p(cfg);
        for (int i = 0; i < 100 && p.workers_ready() < n_npu; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

        const int limit = (frames > 0) ? frames : 0;
        std::atomic<int64_t> decode_us{0};
        auto t0 = std::chrono::steady_clock::now();
        std::thread reader([&] {
            cv::Mat f;
            int idx = 0;
            auto next_frame_time = t0;
            const auto period = pace_fps > 0
                ? std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                      std::chrono::duration<double>(1.0 / pace_fps))
                : std::chrono::steady_clock::duration(0);
            while (limit == 0 || idx < limit) {
                auto d0 = std::chrono::steady_clock::now();
                if (!cap.read(f) || f.empty()) break;
                decode_us += std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - d0).count();
                p.push_frame(idx++, f.clone());  // clone: VideoCapture reuses its buffer
                if (pace_fps > 0) {
                    next_frame_time += period;
                    std::this_thread::sleep_until(next_frame_time);
                }
            }
        });
        reader.join();
        p.wait_idle();
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        int64_t n = p.frames();
        printf("[result] video frames=%lld wall=%.3fs\n", (long long)n, wall);
        printf("[result] aggregate %.2f FPS (source %.1f fps)\n",
               n > 0 ? n / wall : 0.0, src_fps);
        if (n > 0) {
            printf("[result] decode %.2f ms/frame | pre %.2f ms | npu %.2f ms | post %.2f ms\n",
                   decode_us / 1000.0 / n, p.pre_us() / 1000.0 / n,
                   p.npu_us() / 1000.0 / n, p.post_us() / 1000.0 / n);
            printf("[result] detections total=%lld (avg %.1f/frame), rga=%lld cpu_fallback=%lld\n",
                   (long long)p.detections(), (double)p.detections() / n,
                   (long long)p.rga_frames(), (long long)p.cpu_fallback_frames());
            double e50, e95, emx;
            p.e2e_percentiles(e50, e95, emx);
            print_breakdown(n, p.wait_raw_us(), p.pre_us(), p.wait_npu_us(),
                            p.npu_us(), p.wait_post_us(), p.post_us(), p.e2e_us(),
                            e50, e95, emx);
        }
        return 0;
    }

    if (!image_list.empty()) {
        FILE* f = fopen(image_list.c_str(), "r");
        if (!f) { fprintf(stderr, "cannot open image list %s\n", image_list.c_str()); return 1; }
        char line[1024];
        while (fgets(line, sizeof(line), f)) {
            std::string p(line);
            while (!p.empty() && (p.back() == '\n' || p.back() == '\r' || p.back() == ' '))
                p.pop_back();
            if (p.empty() || p[0] == '#') continue;
            cv::Mat m = cv::imread(p, cv::IMREAD_COLOR);
            if (m.empty()) { fprintf(stderr, "skip unreadable: %s\n", p.c_str()); continue; }
            imgs.push_back(m);
        }
        fclose(f);
        if (imgs.empty()) { fprintf(stderr, "image list empty\n"); return 1; }
        img = imgs[0];
        fprintf(stderr, "[main] image list: %zu images, first %dx%d\n", imgs.size(),
                img.cols, img.rows);
    } else if (!image.empty()) {
        img = cv::imread(image, cv::IMREAD_COLOR);
        if (img.empty()) { fprintf(stderr, "cannot read %s\n", image.c_str()); return 1; }
        imgs.push_back(img);
    } else {
        img = cv::Mat(1080, 1920, CV_8UC3, cv::Scalar(40, 60, 80));
        imgs.push_back(img);
    }
    fprintf(stderr, "[main] input %dx%d, model=%s\n", img.cols, img.rows, model.c_str());

    PipelineConfig cfg;
    cfg.model_path = model;
    cfg.num_pre = n_pre;
    cfg.num_npu = n_npu;
    cfg.num_post = n_post;
    cfg.queue_cap = qcap;
    cfg.core_mode = cores;
    cfg.conf_thres = conf;

    if (latency) {
        Pipeline p(cfg);
        if (p.workers_ready() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        double best = 1e9, sum = 0;
        for (int r = 0; r < 30; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            p.push_frame(r, imgs[0]);
            p.wait_idle();
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            if (r >= 5) { sum += ms; if (ms < best) best = ms; }
        }
        printf("[latency] e2e mean %.2f ms, best %.2f ms (n=%d)\n", sum / 25.0, best, 25);
        return 0;
    }

    Pipeline p(cfg);

    // wait until all NPU workers are up
    for (int i = 0; i < 100 && p.workers_ready() < n_npu; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

    for (int i = 0; i < warmup; ++i) p.push_frame(i, imgs[i % imgs.size()]);
    p.wait_idle();

    int64_t f0 = p.frames(), pre0 = p.pre_us(), npu0 = p.npu_us(), post0 = p.post_us();
    int64_t d0 = p.detections();
    int64_t wr0 = p.wait_raw_us(), wn0 = p.wait_npu_us(), wp0 = p.wait_post_us();
    int64_t e0 = p.e2e_us();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; ++i) p.push_frame(warmup + i, imgs[i % imgs.size()]);
    p.wait_idle();
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    int64_t n = p.frames() - f0;
    printf("[result] workers=%d cores=%s frames=%lld wall=%.3fs\n", n_npu, cores.c_str(),
           (long long)n, wall);
    printf("[result] aggregate %.2f FPS\n", n / wall);
    if (n > 0) {
        printf("[result] pre %.2f ms | npu %.2f ms | post %.2f ms (per frame, pipeline sum)\n",
               (p.pre_us() - pre0) / 1000.0 / n, (p.npu_us() - npu0) / 1000.0 / n,
               (p.post_us() - post0) / 1000.0 / n);
        printf("[result] detections total=%lld (avg %.1f/frame), rga=%lld cpu_fallback=%lld\n",
               (long long)(p.detections() - d0), (double)(p.detections() - d0) / n,
               (long long)p.rga_frames(), (long long)p.cpu_fallback_frames());
        double e50, e95, emx;
        p.e2e_percentiles(e50, e95, emx);
        print_breakdown(n, p.wait_raw_us() - wr0, p.pre_us() - pre0,
                        p.wait_npu_us() - wn0, p.npu_us() - npu0,
                        p.wait_post_us() - wp0, p.post_us() - post0,
                        p.e2e_us() - e0, e50, e95, emx);
    }
    return 0;
}
