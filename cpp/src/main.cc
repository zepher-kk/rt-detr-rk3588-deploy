#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "npu_pipeline.h" 

struct AppConfig {
    std::string model_path;
    std::string source;
    int loop_count = 1;       
    int pre_workers = 2;      
    int npu_workers = 3;      
    int post_workers = 1;     
};

void print_help(const char* prog_name) {
    std::cout << "🚀 RK3588 RT-DETR C++ 多线程推理框架\n"
              << "用法: " << prog_name << " -m <模型路径> -s <输入源> [选项]\n\n";
}

bool parse_args(int argc, char** argv, AppConfig& config) {
    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-h" || args[i] == "--help") {
            print_help(argv[0]);
            return false;
        } else if ((args[i] == "-m" || args[i] == "--model") && i + 1 < args.size()) {
            config.model_path = args[++i];
        } else if ((args[i] == "-s" || args[i] == "--source") && i + 1 < args.size()) {
            config.source = args[++i];
        } else if ((args[i] == "-l" || args[i] == "--loop") && i + 1 < args.size()) {
            config.loop_count = std::stoi(args[++i]);
        } else if (args[i] == "--pre" && i + 1 < args.size()) {
            config.pre_workers = std::stoi(args[++i]);
        } else if (args[i] == "--npu" && i + 1 < args.size()) {
            config.npu_workers = std::stoi(args[++i]);
        } else if (args[i] == "--post" && i + 1 < args.size()) {
            config.post_workers = std::stoi(args[++i]);
        }
    }
    if (config.model_path.empty() || config.source.empty()) {
        std::cerr << "❌ 错误: 必须指定模型路径(-m)和输入源(-s)！\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    AppConfig config;
    if (!parse_args(argc, argv, config)) return -1;

    bool is_video = false;
    std::string src_lower = config.source;
    std::transform(src_lower.begin(), src_lower.end(), src_lower.begin(), ::tolower);
    if (src_lower.find(".mp4") != std::string::npos || 
        src_lower.find(".avi") != std::string::npos || 
        src_lower.length() <= 2) { 
        is_video = true; 
    }

    int total_frames = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    {
        std::cout << "-> 初始化 NPU 流水线工场...\n";
        PipelineManager pipeline(config.pre_workers, config.npu_workers, config.post_workers, config.model_path);

        if (!is_video) {
            cv::Mat orig_img = cv::imread(config.source);
            if (orig_img.empty()) return -1;
            total_frames = config.loop_count;
            for (int i = 0; i < total_frames; ++i) {
                pipeline.push_image(i, orig_img);
            }
        } else {
            cv::VideoCapture cap;
            if (config.source.length() <= 2) cap.open(std::stoi(config.source), cv::CAP_V4L2);
            else cap.open(config.source);
            
            if (!cap.isOpened()) return -1;

            cv::Mat frame;
            while (cap.read(frame)) {
                pipeline.push_image(total_frames++, frame.clone());
            }
            cap.release();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    if (is_video || total_frames > 1) {
        auto total_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        float fps = 1000.0f / (static_cast<float>(total_duration_ms) / total_frames);
        std::cout << "✅ 🚀 端到端FPS : " << fps << "\n";
    } else {
        std::cout << "✅ 单图推理完成！请查看保存的结果图片。\n";
    }
    return 0;
}