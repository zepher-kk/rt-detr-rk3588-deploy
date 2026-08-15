#include "npu_pipeline.h"
#include "v4l2_capture.h"
#include "rga_utils.h"
#include "drm_alloc.h"
#include "logger.h"
#include "gst_io.h"

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
	LOG(MOD_MAIN, LOG_WARN) << "\nReceived signal " << sig << ", exiting...\n";
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
	std::string image_path;
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
	int  debug        = -1;   // -G/--DEBUG：-1 表示未指定（默认全模块）
	bool use_v4l2     = true;
	bool show_fps     = true;
	rknn_core_mask npu_mask = RKNN_NPU_CORE_AUTO;
};

// ============================================================================
// 用法说明
// ============================================================================
void print_usage(const char* prog)
{
	LOG(MOD_MAIN, LOG_INFO) << "Usage: " << prog << " [options]\n"
	          << "Options:\n"
	          << "  -m, --model <path>      RT-DETR RKNN model path (required)\n"
	          << "  -v, --video <path>      Video file path (default: camera)\n"
	          << "  -i, --image <path>      Image file path (single image detection)\n"
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
	          << "  -G, --debug <n>        Log modules: 0=errors+report; 1=[Main]; 2=+[RKNN]; 3=+[Pipeline]; ... 8=all (default: all)\n"
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
				LOG(MOD_MAIN, LOG_ERROR) << "Error: " << name << " requires a value\n";
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
		else if (arg == "-i" || arg == "--image")
		{
			args.image_path = get_val("image");
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
				LOG(MOD_MAIN, LOG_ERROR) << "Invalid --npu-cores value: " << val << "\n";
				return false;
			}
		}
		else if (arg == "--opencv")
		{
			args.use_v4l2 = false;
		}
		else if (arg == "-G" || arg == "--debug")
		{
			args.debug = std::stoi(get_val("debug"));
		}
		else
		{
			LOG(MOD_MAIN, LOG_INFO) << "Unknown option: " << arg << "\n";
			print_usage(argv[0]);
			return false;
		}
	}
	if (args.model_path.empty())
	{
		LOG(MOD_MAIN, LOG_ERROR) << "Error: model path is required\n";
		print_usage(argv[0]);
		return false;
	}
	if (!args.image_path.empty() && !args.video_path.empty())
	{
		LOG(MOD_MAIN, LOG_ERROR) << "Error: -i/--image and -v/--video are mutually exclusive\n";
		return false;
	}
	return true;
}

// ============================================================================
// 图片检测模式（单帧同步检测）
// ============================================================================
int run_image_mode(const Args& args, PipelineManager& pipeline)
{
	// 任务 4：图片优先走 GStreamer MPP JPEG 硬解，失败回退 OpenCV 软解
	cv::Mat img;
	GstVideoReader gst_reader;
	if (gst_reader.open(args.image_path))
	{
		if (!gst_reader.read(img))
		{
			LOG(MOD_MAIN, LOG_ERROR) << "GStreamer decode image failed: " << args.image_path << "\n";
			gst_reader.release();
			return 1;
		}
		gst_reader.release();
	}
	else
	{
		img = cv::imread(args.image_path, cv::IMREAD_COLOR);
	}
	if (img.empty())
	{
		LOG(MOD_MAIN, LOG_ERROR) << "Failed to load image: " << args.image_path << "\n";
		return 1;
	}

	int64_t t0 = now_us();
	cv::Mat out;
	if (!pipeline.detect_image(img, out))
	{
		LOG(MOD_MAIN, LOG_ERROR) << "detect_image failed\n";
		return 1;
	}
	int64_t dt_us = now_us() - t0;

	std::string out_path = args.output_path.empty() ? "out_detect.jpg" : args.output_path;
	if (!cv::imwrite(out_path, out))
	{
		LOG(MOD_MAIN, LOG_ERROR) << "Failed to write output image: " << out_path << "\n";
		return 1;
	}

	LOG(MOD_MAIN, LOG_INFO) << "Image detection done: " << out_path
	          << " (e2e " << dt_us / 1000.0 << " ms)\n";
	return 0;
}

// ============================================================================
// V4L2 零拷贝摄像头模式
// ============================================================================
int run_v4l2_mode(const Args& args, PipelineManager& pipeline)
{
	V4l2ZeroCopyCapture cap;
	if (!cap.open(args.device, args.width, args.height, args.fps))
	{
		LOG(MOD_MAIN, LOG_ERROR) << "Failed to open V4L2 device: " << args.device << "\n";
		return 1;
	}
	if (!cap.start())
	{
		LOG(MOD_MAIN, LOG_ERROR) << "Failed to start V4L2 stream\n";
		return 1;
	}

	// 获取实际尺寸并设置源 DMA 池
	int actual_w = cap.width();
	int actual_h = cap.height();
	LOG(MOD_MAIN, LOG_INFO) << "V4L2 actual resolution: " << actual_w << "x" << actual_h << "\n";
	rga_preprocessor().get_src_pool(actual_w, actual_h, RK_FORMAT_BGR_888);

	// 如果指定了输出视频，设置输出（使用args.fps）
	if (!args.output_path.empty())
	{
		pipeline.set_video_output(args.output_path, args.fps);
	}

	LOG(MOD_MAIN, LOG_INFO) << "V4L2 capture started: "
	          << cap.width() << "x" << cap.height() << "\n";

	int frame_id = 0;
	while (!g_should_exit)
	{
		DmaBufferPtr src_buf = cap.read_frame();
		if (!src_buf)
		{
			LOG(MOD_MAIN, LOG_ERROR) << "V4L2 read_frame failed, retrying...\n";
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
	// 任务 4：优先 GStreamer + RK MPP 硬件解码，失败回退 OpenCV 软解
	GstVideoReader gst_reader;
	bool use_gst = gst_reader.open(args.video_path);
	cv::VideoCapture cap;
	double fps = 30.0;
	int actual_w = 0, actual_h = 0;
	int frame_id = 0;
	cv::Mat frame;

	if (use_gst)
	{
		// 【任务 4 追加迭代】帧率来源：
		// - 容器/caps 提供有效帧率（CFR）→ 直接使用；
		// - VFR 或帧率元数据缺失（如 0/1）→ 两遍法：先纯解码统计容器时长/平均帧率
		//   （MPP 硬解，555 帧约 2-3s），再正常处理。避免"取前两帧间隔"在 VFR 源上
		//   误判（如 480×332 录屏前段 60fps burst → 输出 60fps → 时长 21.7s 被压成 9.2s）。
		double fps_est = gst_reader.fps();
		if (!gst_reader.caps_fps_authoritative())
		{
			GstVideoReader probe;
			if (probe.open(args.video_path))
			{
				cv::Mat dummy;
				while (probe.read(dummy)) {}
				double avg = probe.measured_avg_fps();
				probe.release();
				if (avg >= 1.0 && avg <= 240.0)
				{
					fps_est = avg;
					LOG(MOD_MAIN, LOG_INFO) << "VFR probe avg fps=" << fps_est << "\n";
				}
			}
		}

		// 硬解：先取首帧获取尺寸/帧率
		if (!gst_reader.read(frame))
		{
			LOG(MOD_MAIN, LOG_ERROR) << "GStreamer read first frame failed: " << args.video_path << "\n";
			gst_reader.release();
			return 1;
		}
		cv::Mat frame2;
		bool has_second = gst_reader.read(frame2);
		fps = fps_est;
		actual_w = frame.cols;
		actual_h = frame.rows;
		LOG(MOD_MAIN, LOG_INFO) << "GStreamer+MPP hardware decode: "
		          << actual_w << "x" << actual_h << " @" << fps << "\n";
		if (!args.output_path.empty())
		{
			pipeline.set_video_output(args.output_path, fps);
		}
		pipeline.push_image(frame_id++, frame);
		if (has_second)
		{
			pipeline.push_image(frame_id++, frame2);
			while (!g_should_exit && gst_reader.read(frame))
			{
				pipeline.push_image(frame_id++, frame);
			}
		}
		gst_reader.release();
	}
	else
	{
		// 回退：OpenCV 软解
		cap.open(args.video_path);
		if (!cap.isOpened())
		{
			LOG(MOD_MAIN, LOG_ERROR) << "Failed to open video: " << args.video_path << "\n";
			return 1;
		}
		fps = cap.get(cv::CAP_PROP_FPS);
		if (fps <= 0) fps = 30.0;
		actual_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
		actual_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		LOG(MOD_MAIN, LOG_WARN) << "GStreamer unavailable, fallback to OpenCV soft decode: "
		          << actual_w << "x" << actual_h << "\n";
		if (!args.output_path.empty())
		{
			pipeline.set_video_output(args.output_path, fps);
		}
		while (!g_should_exit && cap.read(frame))
		{
			pipeline.push_image(frame_id++, frame);
		}
		cap.release();
	}
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
		LOG(MOD_MAIN, LOG_ERROR) << "Failed to open camera: " << args.device << "\n";
		return 1;
	}
	cap.set(cv::CAP_PROP_FRAME_WIDTH, args.width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, args.height);
	cap.set(cv::CAP_PROP_FPS, args.fps);

	// 读取第一帧获取实际尺寸
	cv::Mat first_frame;
	if (!cap.read(first_frame))
	{
		LOG(MOD_MAIN, LOG_ERROR) << "Failed to read first frame from camera\n";
		return 1;
	}
	int actual_w = first_frame.cols;
	int actual_h = first_frame.rows;
	LOG(MOD_MAIN, LOG_INFO) << "Camera actual resolution: " << actual_w << "x" << actual_h << "\n";

	if (!args.output_path.empty())
	{
		pipeline.set_video_output(args.output_path, args.fps);
	}

	int frame_id = 0;
	cv::Mat frame = first_frame.clone();
	while (!g_should_exit)
	{
		pipeline.push_image(frame_id++, frame);
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

	// 应用 -G/--DEBUG 日志模块配置（-G 0=仅错误；-G N=前 N 个模块；默认全开）
	if (args.debug >= 0)
	{
		int mask = 0;
		int n = args.debug > MOD_COUNT ? MOD_COUNT : args.debug;
		for (int i = 0; i < n; ++i) mask |= (1 << i);
		log_set_modules(mask);
	}

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	// 创建流水线（图片模式无需线程池 worker，走同步单帧接口）
	int pre_workers  = args.image_path.empty() ? args.pre_workers  : 0;
	int npu_workers  = args.image_path.empty() ? args.npu_workers  : 0;
	int post_workers = args.image_path.empty() ? args.post_workers : 0;
	PipelineManager pipeline(pre_workers, npu_workers, post_workers,
	                         args.model_path,
	                         args.queue_cap, args.conf, args.npu_mask);

	// 注意：此处不再统一调用 set_video_output，而是在各个采集模式中按需调用

	// 选择采集模式
	int ret = 0;
	if (!args.image_path.empty())
	{
		ret = run_image_mode(args, pipeline);
	}
	else if (!args.video_path.empty())
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
