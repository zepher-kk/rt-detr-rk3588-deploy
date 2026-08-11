#include "npu_pipeline.h"
#include "v4l2_capture.h"
#include "rga_utils.h"
#include "drm_alloc.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <csignal>
#include <cstring>
#include <opencv2/opencv.hpp>

// RGA 格式常量
#include <rga.h>
#include "rknn_api.h"   // rknn_core_mask 枚举

static std::atomic<bool> g_should_exit{false};

static void signal_handler(int sig)
{
	g_should_exit = true;
	std::cerr << "\n[Main] Received signal " << sig << ", exiting...\n";
}

static inline int64_t now_us()
{
	return std::chrono::duration_cast<std::chrono::microseconds>(
	           std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// ============================================================================
// 配置参数
// ============================================================================
struct Args
{
	std::string model_path;
	std::string video_path;
	std::string device     = "/dev/video0";
	std::string output_path;
	int  width      = 1920;
	int  height     = 1080;
	int  fps        = 30;
	int  pre_workers  = 2;
	int  npu_workers  = 3;
	int  post_workers = 1;
	int  queue_cap    = 16;
	float conf        = 0.45f;
	bool use_v4l2     = true;
	bool show_fps     = true;
	rknn_core_mask npu_mask = RKNN_NPU_CORE_AUTO;
};

// ============================================================================
// 用法说明
// ============================================================================
void print_usage(const char* prog)
{
	std::cerr << "Usage: " << prog << " [options]\n"
	          << "Options:\n"
	          << "  -m, --model <path>      RT-DETR RKNN model path (required)\n"
	          << "  -v, --video <path>      Video file path (default: camera)\n"
	          << "  -d, --device <dev>      V4L2 device (default: /dev/video0)\n"
	          << "  -W, --width <n>         Capture width (default: 1920, fallback)\n"
	          << "  -H, --height <n>        Capture height (default: 1080, fallback)\n"
	          << "  -F, --fps <n>           Capture fps (default: 30)\n"
	          << "  -o, --output <path>     Output video path (default: none)\n"
	          << "  -c, --conf <f>          Confidence threshold (default: 0.45)\n"
	          << "  -n, --npu-workers <n>   NPU workers (default: 3)\n"
	          << "  -p, --pre-workers <n>   Preprocess workers (default: 2)\n"
	          << "  -P, --post-workers <n>  Postprocess workers (default: 1)\n"
	          << "  -q, --queue-cap <n>    Queue capacity (default: 16)\n"
	          << "  --npu-cores <val>      NPU core mask: auto|0|1|2|0,1|0,1,2 (default: auto)\n"
	          << "  --opencv                Use OpenCV camera (fallback)\n"
	          << "  -h, --help              Show this help\n";
}

// ============================================================================
// 参数解析
// ============================================================================
bool parse_args(int argc, char** argv, Args& args)
{
	for (int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];
		auto get_val = [&](const char* name) -> std::string
		{
			if (i + 1 >= argc)
			{
				std::cerr << "Error: " << name << " requires a value\n";
				exit(1);
			}
			return argv[++i];
		};
		if (arg == "-h" || arg == "--help")
		{
			print_usage(argv[0]);
			return false;
		}
		else if (arg == "-m" || arg == "--model")
		{
			args.model_path = get_val("model");
		}
		else if (arg == "-v" || arg == "--video")
		{
			args.video_path = get_val("video");
		}
		else if (arg == "-d" || arg == "--device")
		{
			args.device = get_val("device");
		}
		else if (arg == "-W" || arg == "--width")
		{
			args.width = std::stoi(get_val("width"));
		}
		else if (arg == "-H" || arg == "--height")
		{
			args.height = std::stoi(get_val("height"));
		}
		else if (arg == "-F" || arg == "--fps")
		{
			args.fps = std::stoi(get_val("fps"));
		}
		else if (arg == "-o" || arg == "--output")
		{
			args.output_path = get_val("output");
		}
		else if (arg == "-c" || arg == "--conf")
		{
			args.conf = std::stof(get_val("conf"));
		}
		else if (arg == "-n" || arg == "--npu-workers")
		{
			args.npu_workers = std::stoi(get_val("npu-workers"));
		}
		else if (arg == "-p" || arg == "--pre-workers")
		{
			args.pre_workers = std::stoi(get_val("pre-workers"));
		}
		else if (arg == "-P" || arg == "--post-workers")
		{
			args.post_workers = std::stoi(get_val("post-workers"));
		}
		else if (arg == "-q" || arg == "--queue-cap")
		{
			args.queue_cap = std::stoi(get_val("queue-cap"));
		}
		else if (arg == "--npu-cores")
		{
			std::string val = get_val("npu-cores");
			if (val == "auto")
			{
				args.npu_mask = RKNN_NPU_CORE_AUTO;
			}
			else if (val == "0")
			{
				args.npu_mask = RKNN_NPU_CORE_0;
			}
			else if (val == "1")
			{
				args.npu_mask = RKNN_NPU_CORE_1;
			}
			else if (val == "2")
			{
				args.npu_mask = RKNN_NPU_CORE_2;
			}
			else if (val == "0,1")
			{
				args.npu_mask = RKNN_NPU_CORE_0_1;
			}
			else if (val == "0,1,2" || val == "all")
			{
				args.npu_mask = RKNN_NPU_CORE_ALL;
			}
			else
			{
				std::cerr << "Invalid --npu-cores value: " << val << "\n";
				return false;
			}
		}
		else if (arg == "--opencv")
		{
			args.use_v4l2 = false;
		}
		else
		{
			std::cerr << "Unknown option: " << arg << "\n";
			print_usage(argv[0]);
			return false;
		}
	}
	if (args.model_path.empty())
	{
		std::cerr << "Error: model path is required\n";
		print_usage(argv[0]);
		return false;
	}
	return true;
}

// ============================================================================
// V4L2 零拷贝摄像头模式
// ============================================================================
int run_v4l2_mode(const Args& args, PipelineManager& pipeline)
{
	V4l2ZeroCopyCapture cap;
	if (!cap.open(args.device, args.width, args.height, args.fps))
	{
		std::cerr << "[Main] Failed to open V4L2 device: " << args.device << "\n";
		return 1;
	}
	if (!cap.start())
	{
		std::cerr << "[Main] Failed to start V4L2 stream\n";
		return 1;
	}

	// 获取实际尺寸并设置源 DMA 池
	int actual_w = cap.width();
	int actual_h = cap.height();
	std::cerr << "[Main] V4L2 actual resolution: " << actual_w << "x" << actual_h << "\n";
	rga_preprocessor().get_src_pool(actual_w, actual_h, RK_FORMAT_BGR_888);

	// 如果指定了输出视频，设置输出（使用args.fps）
	if (!args.output_path.empty())
	{
		pipeline.set_video_output(args.output_path, args.fps);
	}

	std::cerr << "[Main] V4L2 capture started: "
	          << cap.width() << "x" << cap.height() << "\n";

	int frame_id = 0;
	while (!g_should_exit)
	{
		DmaBufferPtr src_buf = cap.read_frame();
		if (!src_buf)
		{
			std::cerr << "[Main] V4L2 read_frame failed, retrying...\n";
			continue;
		}
		pipeline.push_dma_frame(frame_id++, src_buf);
	}

	cap.stop();
	return 0;
}

// ============================================================================
// 视频文件模式
// ============================================================================
int run_video_mode(const Args& args, PipelineManager& pipeline)
{
	// TODO: 后续使用 GStreamer + MPP 硬件解码替换 OpenCV 软解码，以降低 CPU 负载
	cv::VideoCapture cap(args.video_path);
	if (!cap.isOpened())
	{
		std::cerr << "[Main] Failed to open video: " << args.video_path << "\n";
		return 1;
	}

	// 读取实际帧率
	double fps = cap.get(cv::CAP_PROP_FPS);
	if (fps <= 0) fps = 30.0;

	int actual_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int actual_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cerr << "[Main] Video resolution: " << actual_w << "x" << actual_h << "\n";
	// rga_preprocessor().get_src_pool(actual_w, actual_h, RK_FORMAT_BGR_888);

	if (!args.output_path.empty())
	{
		pipeline.set_video_output(args.output_path, fps);
	}

	// 创建源 DMA buffer 池（用于桥接 cv::Mat → DMA）
	DmaBufferPool& src_pool = rga_preprocessor().get_src_pool(actual_w, actual_h, RK_FORMAT_BGR_888);

	int frame_id = 0;
	cv::Mat frame;
	while (!g_should_exit && cap.read(frame))
	{
		// 桥接：cv::Mat → DMA buffer（一次性拷贝）
		DmaBufferPtr src_buf = rga_preprocessor().bridge_mat_to_dma(frame, src_pool);
		if (!src_buf) continue;
		pipeline.push_dma_frame(frame_id++, src_buf, frame);  // 传入原始图像
	}
	cap.release();
	return 0;
}

// ============================================================================
// OpenCV 摄像头模式（回退）
// ============================================================================
int run_opencv_camera_mode(const Args& args, PipelineManager& pipeline)
{
	cv::VideoCapture cap(args.device, cv::CAP_V4L2);
	if (!cap.isOpened())
	{
		std::cerr << "[Main] Failed to open camera: " << args.device << "\n";
		return 1;
	}
	cap.set(cv::CAP_PROP_FRAME_WIDTH, args.width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, args.height);
	cap.set(cv::CAP_PROP_FPS, args.fps);

	// 读取第一帧获取实际尺寸
	cv::Mat first_frame;
	if (!cap.read(first_frame))
	{
		std::cerr << "[Main] Failed to read first frame from camera\n";
		return 1;
	}
	int actual_w = first_frame.cols;
	int actual_h = first_frame.rows;
	std::cerr << "[Main] Camera actual resolution: " << actual_w << "x" << actual_h << "\n";
	rga_preprocessor().get_src_pool(actual_w, actual_h, RK_FORMAT_BGR_888);

	if (!args.output_path.empty())
	{
		pipeline.set_video_output(args.output_path, args.fps);
	}

	DmaBufferPool& src_pool = rga_preprocessor().get_src_pool(actual_w, actual_h, RK_FORMAT_BGR_888);

	int frame_id = 0;
	cv::Mat frame = first_frame.clone();
	while (!g_should_exit)
	{
		DmaBufferPtr src_buf = rga_preprocessor().bridge_mat_to_dma(frame, src_pool);
		if (!src_buf) continue;
		pipeline.push_dma_frame(frame_id++, src_buf, frame);
		if (!cap.read(frame)) break;
	}
	cap.release();
	return 0;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv)
{
	Args args;
	if (!parse_args(argc, argv, args)) return 1;

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	// 创建流水线
	PipelineManager pipeline(args.pre_workers, args.npu_workers,
	                         args.post_workers, args.model_path,
	                         args.queue_cap, args.conf, args.npu_mask);

	// 注意：此处不再统一调用 set_video_output，而是在各个采集模式中按需调用

	// 选择采集模式
	int ret = 0;
	if (!args.video_path.empty())
	{
		ret = run_video_mode(args, pipeline);
	}
	else if (args.use_v4l2)
	{
		ret = run_v4l2_mode(args, pipeline);
	}
	else
	{
		ret = run_opencv_camera_mode(args, pipeline);
	}

	// 等待流水线处理完所有帧
	pipeline.wait_idle();
	pipeline.print_perf_summary();

	return ret;
}
