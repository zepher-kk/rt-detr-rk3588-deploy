#pragma once
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <map>  // 🌟 引入 map 用于帧重排
#include <opencv2/opencv.hpp>
#include "rknn_detector.h"

struct RawTask { int frame_id; cv::Mat orig_img; };
struct NpuTask { int frame_id; cv::Mat orig_img; cv::Mat preprocessed_img; };
struct PostTask {
    int frame_id;
    cv::Mat orig_img;
    std::vector<float> pred_boxes;
    std::vector<float> pred_logits;
    int num_boxes;
};

template <typename T>
class SafeQueue {
private:
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
public:
    void push(T task) {
        std::unique_lock<std::mutex> lock(mtx_);
        queue_.push(task);
        lock.unlock();
        cv_.notify_one();
    }
    bool pop(T& task, bool& is_running) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this, &is_running]() { return !queue_.empty() || !is_running; });
        if (!is_running && queue_.empty()) return false;
        task = queue_.front();
        queue_.pop();
        return true;
    }
};

class PipelineManager {
private:
    SafeQueue<RawTask>  queue_raw_;
    SafeQueue<NpuTask>  queue_npu_;
    SafeQueue<PostTask> queue_post_;
    std::vector<std::thread> workers_pre_;
    std::vector<std::thread> workers_npu_;
    std::vector<std::thread> workers_post_;
    bool is_running_;
    std::string model_path_;

    // ==========================================
    // 🌟 视频保存与乱序重排专用组件
    // ==========================================
    cv::VideoWriter video_writer_;        // OpenCV 视频写入器
    std::map<int, cv::Mat> frame_buffer_; // 存放乱序到达的帧
    int next_write_frame_id_ = 0;         // 记录当前应该写入视频的正确帧号
    std::mutex writer_mtx_;               // 写入锁

    void worker_preprocess();
    void worker_npu_infer(int core_id);
    void worker_postprocess();
public:
    PipelineManager(int num_pre, int num_npu, int num_post, const std::string& model_path);
    ~PipelineManager();
    void push_image(int frame_id, const cv::Mat& img);
};