#include "npu_pipeline.h"
#include "postprocess.h"
#include <iostream>

PipelineManager::PipelineManager(int num_pre, int num_npu, int num_post, const std::string& model_path)
    : is_running_(true), model_path_(model_path)
{
    for (int i = 0; i < num_pre; ++i) workers_pre_.emplace_back(&PipelineManager::worker_preprocess, this);
    for (int i = 0; i < num_npu; ++i) workers_npu_.emplace_back(&PipelineManager::worker_npu_infer, this, i);
    for (int i = 0; i < num_post; ++i) workers_post_.emplace_back(&PipelineManager::worker_postprocess, this);
    std::cout << "✅ 三段式接力流水线启动成功！\n";
}

// ==========================================
// 🌟 核心修复 1：教科书级的完美清空流水线
// ==========================================
PipelineManager::~PipelineManager() {
    // 1. 发通知给【洗菜工】，等他们把积压的原图全洗完
    for (size_t i = 0; i < workers_pre_.size(); ++i) queue_raw_.push({-1, cv::Mat()});
    for (auto& t : workers_pre_) if (t.joinable()) t.join();

    // 2. 此时原图全进 NPU 队列了。发通知给【大厨】，等他们把 NPU 队列全跑完
    for (size_t i = 0; i < workers_npu_.size(); ++i) queue_npu_.push({-1, cv::Mat(), cv::Mat()});
    for (auto& t : workers_npu_) if (t.joinable()) t.join();

    // 3. 此时所有目标都被检出了。发通知给【洗碗工】，等他们把最后一帧画完存好
    for (size_t i = 0; i < workers_post_.size(); ++i) queue_post_.push({-1, cv::Mat(), std::vector<float>(), std::vector<float>(), 0});
    for (auto& t : workers_post_) if (t.joinable()) t.join();

    // 4. 工人全走光了，安全拉闸
    is_running_ = false;

    if (video_writer_.isOpened()) {
        video_writer_.release();
        std::cout << "🎬 完美按序重组！视频已成功保存为 result_video.mp4\n";
    }
}

void PipelineManager::push_image(int frame_id, const cv::Mat& img) {
    queue_raw_.push({frame_id, img.clone()}); 
}

void PipelineManager::worker_preprocess() {
    RawTask task;
    while (queue_raw_.pop(task, is_running_)) {
        // 🌟 核心修复 2：看到 -1 毒丸，直接退出线程 (break)，而不是 continue！
        if (task.frame_id == -1) break; 
        
        cv::Mat img;
        cv::resize(task.orig_img, img, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        NpuTask npu_task;
        npu_task.frame_id = task.frame_id;
        npu_task.orig_img = task.orig_img;
        npu_task.preprocessed_img = img.clone(); 
        queue_npu_.push(npu_task); 
    }
}

void PipelineManager::worker_npu_infer(int core_id) {
    RKNNDetector detector;
    if (!detector.init(model_path_)) return;

    NpuTask task;
    while (queue_npu_.pop(task, is_running_)) {
        // 🌟 核心修复 2：直接退出线程
        if (task.frame_id == -1) break; 

        std::vector<float> out_boxes;
        std::vector<float> out_logits;
        int num_boxes = 0;
        
        detector.infer_only(task.preprocessed_img, out_boxes, out_logits, num_boxes);

        PostTask post_task;
        post_task.frame_id = task.frame_id;
        post_task.orig_img = task.orig_img;
        post_task.pred_boxes = std::move(out_boxes);
        post_task.pred_logits = std::move(out_logits);
        post_task.num_boxes = num_boxes;
        queue_post_.push(post_task); 
    }
}

void PipelineManager::worker_postprocess() {
    PostTask task;
    while (queue_post_.pop(task, is_running_)) {
        // 🌟 核心修复 2：直接退出线程
        if (task.frame_id == -1) break; 

        std::vector<DetectResult> results = decode_rtdetr_output(
            task.pred_boxes.data(), task.pred_logits.data(),
            task.num_boxes, task.orig_img.cols, task.orig_img.rows, 0.45f
        );

        draw_results(task.orig_img, results);

        {
            std::lock_guard<std::mutex> lock(writer_mtx_);
            if (!video_writer_.isOpened()) {
                video_writer_.open("result_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, task.orig_img.size());
            }

            frame_buffer_[task.frame_id] = task.orig_img;

            while (frame_buffer_.find(next_write_frame_id_) != frame_buffer_.end()) {
                video_writer_.write(frame_buffer_[next_write_frame_id_]);
                frame_buffer_.erase(next_write_frame_id_);
                next_write_frame_id_++;
            }
        }

        static std::mutex cout_mtx;
        std::lock_guard<std::mutex> lock_cout(cout_mtx);
        std::cout << "🚀 [Pipeline] 第 " << task.frame_id << " 帧处理完工，检出 " << results.size() << " 个目标。\n";
    }
}