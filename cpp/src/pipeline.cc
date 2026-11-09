#include "pipeline.h"

#include <algorithm>

#include <chrono>
#include <cstdio>
#include <thread>

#include "rga_utils.h"
#include "rknn_detector.h"

static RgaPreprocessor& rga() {
    static RgaPreprocessor p;
    return p;
}

static int decode_count(FrameBundle& b, float thr) {
    const int nc = b.num_classes;
    int n = 0;
    for (int i = 0; i < b.num_boxes; ++i) {
        const float* s = b.pred_logits.data() + (size_t)i * nc;
        float best = -1e9f;
        for (int c = 0; c < nc; ++c)
            if (s[c] > best) best = s[c];
        if (best >= thr) ++n;
    }
    return n;
}

Pipeline::Pipeline(const PipelineConfig& cfg)
    : cfg_(cfg), q_raw_(cfg.queue_cap), q_npu_(cfg.queue_cap), q_post_(cfg.queue_cap) {
    rga().init((size_t)cfg.num_npu + 4);
    t_start_ = now_us();
    for (int i = 0; i < cfg_.num_pre; ++i) pre_.emplace_back(&Pipeline::worker_pre, this);
    for (int i = 0; i < cfg_.num_npu; ++i) npu_.emplace_back(&Pipeline::worker_npu, this, i);
    for (int i = 0; i < cfg_.num_post; ++i) post_.emplace_back(&Pipeline::worker_post, this);
}

Pipeline::~Pipeline() {
    stop();
    for (auto& t : pre_) if (t.joinable()) t.join();
    for (auto& t : npu_) if (t.joinable()) t.join();
    for (auto& t : post_) if (t.joinable()) t.join();
}

void Pipeline::stop() {
    if (!running_.exchange(false)) return;
    q_raw_.shutdown();
    q_npu_.shutdown();
    q_post_.shutdown();
}

void Pipeline::push_frame(int frame_id, const cv::Mat& img) {
    if (!workers_ok_) return;
    auto b = std::make_shared<FrameBundle>();
    b->frame_id = frame_id;
    b->src = img;  // shallow ref; caller must keep the buffer alive/unchanged
    b->t_enqueue = now_us();
    inflight_ += 1;
    q_raw_.push(std::move(b));
}

void Pipeline::wait_idle() {
    while (inflight_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

double Pipeline::elapsed_s() const {
    std::lock_guard<std::mutex> lk(t_mtx_);
    return (now_us() - t_start_) / 1e6;
}

void Pipeline::e2e_percentiles(double& p50, double& p95, double& mx) const {
    std::lock_guard<std::mutex> lk(e2e_mtx_);
    if (e2e_samples_.empty()) { p50 = p95 = mx = 0; return; }
    std::vector<int64_t> v = e2e_samples_;
    std::sort(v.begin(), v.end());
    p50 = v[v.size() / 2] / 1000.0;
    p95 = v[(size_t)(v.size() * 0.95)] / 1000.0;
    mx = v.back() / 1000.0;
}

void Pipeline::worker_pre() {
    while (true) {
        FrameBundlePtr b;
        if (!q_raw_.pop(b)) break;
        if (!b) break;
        int64_t t0 = now_us();
        b->t_pre_start = t0;
        wait_raw_us_ += t0 - b->t_enqueue;
        DmaBufferPtr dma = rga().preprocess(b->src);
        if (!dma) {
            fprintf(stderr, "[pre] frame %d preprocess failed\n", b->frame_id);
            inflight_ -= 1;
            continue;
        }
        b->input_buf = dma;
        b->t_pre_done = now_us();
        pre_us_ += b->t_pre_done - t0;
        (dma->via_rga ? rga_frames_ : cpu_fallback_frames_) += 1;
        q_npu_.push(std::move(b));
    }
}

void Pipeline::worker_npu(int idx) {
    rknn_core_mask mask = RKNN_NPU_CORE_AUTO;
    if (cfg_.core_mode == "all") {
        mask = RKNN_NPU_CORE_0_1_2;
    } else if (cfg_.core_mode == "split") {
        // One worker should use all three cores; several workers get pinned
        // to one core each (that is what makes multi-instance scale).
        if (cfg_.num_npu <= 1) {
            mask = RKNN_NPU_CORE_0_1_2;
        } else {
            switch (idx % 3) {
                case 0: mask = RKNN_NPU_CORE_0; break;
                case 1: mask = RKNN_NPU_CORE_1; break;
                default: mask = RKNN_NPU_CORE_2; break;
            }
        }
    }

    RKNNDetector det;
    if (!det.init(cfg_.model_path, mask)) {
        fprintf(stderr, "[npu] worker %d init failed\n", idx);
        workers_ok_ = false;
        return;
    }
    ready_ += 1;

    while (true) {
        FrameBundlePtr b;
        if (!q_npu_.pop(b)) break;
        if (!b) break;
        int64_t t0 = now_us();
        b->t_npu_start = t0;
        wait_npu_us_ += t0 - b->t_pre_done;
        bool ok = det.infer(b->input_buf, *b);
        if (!ok) {
            fprintf(stderr, "[npu] worker %d infer failed (frame %d)\n", idx, b->frame_id);
            inflight_ -= 1;
            continue;
        }
        b->input_buf.reset();
        b->t_npu_done = now_us();
        npu_us_ += b->t_npu_done - t0;
        q_post_.push(std::move(b));
    }
}

void Pipeline::worker_post() {
    while (true) {
        FrameBundlePtr b;
        if (!q_post_.pop(b)) break;
        if (!b) break;
        int64_t t0 = now_us();
        b->t_post_start = t0;
        wait_post_us_ += t0 - b->t_npu_done;
        dets_ += decode_count(*b, cfg_.conf_thres);
        b->t_post_done = now_us();
        post_us_ += b->t_post_done - t0;
        e2e_us_ += b->t_post_done - b->t_enqueue;
        {
            std::lock_guard<std::mutex> lk(e2e_mtx_);
            e2e_samples_.push_back(b->t_post_done - b->t_enqueue);
        }
        frames_ += 1;
        inflight_ -= 1;
    }
}
