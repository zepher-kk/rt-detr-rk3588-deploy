#pragma once

#include "types.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

template <typename T>
class BoundedQueue {
  public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}

    void shutdown() {
        run_ = false;
        not_full_.notify_all();
        not_empty_.notify_all();
    }

    void push(T item) {
        std::unique_lock<std::mutex> lk(mtx_);
        not_full_.wait(lk, [&] { return q_.size() < cap_ || !run_; });
        if (!run_) return;
        q_.push(std::move(item));
        lk.unlock();
        not_empty_.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lk(mtx_);
        not_empty_.wait(lk, [&] { return !q_.empty() || !run_; });
        if (!run_ && q_.empty()) return false;
        item = std::move(q_.front());
        q_.pop();
        lk.unlock();
        not_full_.notify_one();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return q_.size();
    }

  private:
    std::queue<T> q_;
    mutable std::mutex mtx_;
    std::condition_variable not_full_, not_empty_;
    size_t cap_;
    std::atomic<bool> run_{true};
};

struct PipelineConfig {
    std::string model_path;
    std::string decoder_path;  // non-empty: backbone(INT8) -> decoder(FP16)
    int num_pre = 2;
    int num_npu = 3;
    int num_post = 1;
    size_t queue_cap = 16;
    std::string core_mode = "split";  // split | auto | all
    float conf_thres = 0.45f;
};

class Pipeline {
  public:
    explicit Pipeline(const PipelineConfig& cfg);
    ~Pipeline();

    void push_frame(int frame_id, const cv::Mat& img);
    void wait_idle();
    void stop();

    double elapsed_s() const;
    int64_t frames() const { return frames_.load(); }

    // stats (microseconds, summed over frames)
    int64_t pre_us() const { return pre_us_.load(); }
    int64_t npu_us() const { return npu_us_.load(); }
    int64_t post_us() const { return post_us_.load(); }
    int64_t detections() const { return dets_.load(); }
    int64_t rga_frames() const { return rga_frames_.load(); }
    int64_t cpu_fallback_frames() const { return cpu_fallback_frames_.load(); }
    int64_t workers_ready() const { return ready_.load(); }

    // stage breakdown (microseconds, summed); averages = value / frames
    int64_t wait_raw_us() const { return wait_raw_us_.load(); }
    int64_t wait_npu_us() const { return wait_npu_us_.load(); }
    int64_t wait_post_us() const { return wait_post_us_.load(); }
    int64_t e2e_us() const { return e2e_us_.load(); }
    void e2e_percentiles(double& p50, double& p95, double& mx) const;

  private:
    void worker_pre();
    void worker_npu(int idx);
    void worker_post();

    PipelineConfig cfg_;
    BoundedQueue<FrameBundlePtr> q_raw_, q_npu_, q_post_;
    std::vector<std::thread> pre_, npu_, post_;
    std::atomic<bool> running_{true};
    std::atomic<bool> workers_ok_{true};
    std::atomic<int> ready_{0};

    std::atomic<int64_t> frames_{0}, dets_{0};
    std::atomic<int64_t> pre_us_{0}, npu_us_{0}, post_us_{0};
    std::atomic<int64_t> rga_frames_{0}, cpu_fallback_frames_{0};
    std::atomic<int> inflight_{0};
    // stage accounting
    std::atomic<int64_t> wait_raw_us_{0}, wait_npu_us_{0}, wait_post_us_{0}, e2e_us_{0};
    mutable std::mutex e2e_mtx_;
    std::vector<int64_t> e2e_samples_;

    int64_t t_start_ = 0;
    mutable std::mutex t_mtx_;
};
