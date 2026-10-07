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
#include <fstream>
#include <sstream>
#include <algorithm>
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
	std::string image_list_path;   //!< --image-list：批量图片清单（每行一个路径）
	std::string dump_det_path;     //!< --dump-det：检测结果导出文件
	std::string device     = "/dev/video0";
	std::string output_path;
	int  width      = 1920;
	int  height     = 1080;
	int  fps        = 30;
	bool fps_set    = false;   // 用户是否显式指定 -F/--fps
	int  pre_workers  = 2;
	int  npu_workers  = 3;
	int  post_workers = 1;
	int  queue_cap    = 16;
	float conf        = 0.45f;
	std::string conf_class;   //!< --conf-class：按类阈值规格串（如 "0:-0.40,3:0.00"）
	std::string profile;      //!< --profile：阈值预设档位（balanced / recall）
	int  debug        = -1;   // -G/--DEBUG：-1 表示未指定（默认全模块）
	bool use_v4l2     = true;
	bool show_fps     = true;
	bool show_display = false;
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
	          << "  --image-list <file>     Batch image list: one image path per line\n"
	          << "  --dump-det <file>       Dump detections: <stem> <class_id> <score> <x1> <y1> <x2> <y2>\n"
	          << "  -d, --device <dev>      V4L2 device (default: /dev/video0)\n"
	          << "  -W, --width <n>         Capture width (default: 1920, fallback)\n"
	          << "  -H, --height <n>        Capture height (default: 1080, fallback)\n"
	          << "  -F, --fps <n>           Override input/output fps (video file & camera; default: auto/source)\n"
	          << "  -o, --output <path>     Output video path (default: none)\n"
	          << "  -c, --conf <f>          Confidence threshold (default: 0.45)\n"
	          << "  --conf-class <spec>     Per-class thresholds, e.g. \"0:-0.40,3:0.00,9:-0.55\"\n"
	          << "  --profile <name>        Threshold preset: balanced (global, default) | recall (class-wise calibrated)\n"
	          << "  -n, --npu-workers <n>   NPU workers (default: 3)\n"
	          << "  -p, --pre-workers <n>   Preprocess workers (default: 2)\n"
	          << "  -P, --post-workers <n>  Postprocess workers (default: 1)\n"
	          << "  -q, --queue-cap <n>    Queue capacity (default: 16)\n"
	          << "  --npu-cores <val>      NPU core mask: auto|0|1|2|0,1|0,1,2 (default: auto)\n"
	          << "  --opencv                Use OpenCV camera (fallback)\n"
	          << "  --display               Show real-time detection window (requires display)\n"
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
		else if (arg == "--image-list")
		{
			args.image_list_path = get_val("image-list");
		}
		else if (arg == "--dump-det")
		{
			args.dump_det_path = get_val("dump-det");
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
			args.fps_set = true;
		}
		else if (arg == "-o" || arg == "--output")
		{
			args.output_path = get_val("output");
		}
		else if (arg == "-c" || arg == "--conf")
		{
			args.conf = std::stof(get_val("conf"));
		}
		else if (arg == "--conf-class")
		{
			args.conf_class = get_val("conf-class");
		}
		else if (arg == "--profile")
		{
			args.profile = get_val("profile");
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
		else if (arg == "--display")
		{
			args.show_display = true;
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
	// 图片优先走 GStreamer MPP JPEG 硬解，失败回退 OpenCV 软解
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
// 批量图片模式（离线评估）：单次加载模型，顺序处理清单中的图片
//   - 默认只导出检测框（不回写图片），用于 mAP 等离线精度评估；
//   - 导出格式：<stem> <class_id> <score> <x1> <y1> <x2> <y2>（每行一个框）
//   - 指定 -o 时额外输出 <stem>_det.jpg（仅用于抽查/可视化）
// ============================================================================
//! 逐框写出检测结果：<stem> <class_id> <score> <x1> <y1> <x2> <y2>
static void dump_detections(std::ostream& os, const std::string& stem,
                            const std::vector<DetectResult>& results)
{
	for (const auto& r : results)
	{
		os << stem << ' ' << r.class_id << ' ' << r.score << ' '
		   << r.box.x << ' ' << r.box.y << ' '
		   << (r.box.x + r.box.width) << ' ' << (r.box.y + r.box.height) << '\n';
	}
}

int run_image_list_mode(const Args& args, PipelineManager& pipeline)
{
	std::ifstream list(args.image_list_path);
	if (!list)
	{
		LOG(MOD_MAIN, LOG_ERROR) << "Cannot open image list: " << args.image_list_path << "\n";
		return 1;
	}

	std::ofstream dump;
	if (!args.dump_det_path.empty())
	{
		dump.open(args.dump_det_path, std::ios::out | std::ios::trunc);
		if (!dump)
		{
			LOG(MOD_MAIN, LOG_ERROR) << "Cannot open dump file: " << args.dump_det_path << "\n";
			return 1;
		}
	}

	int64_t t_start = now_us();
	int n_total = 0, n_ok = 0, n_fail = 0, n_det = 0;
	std::string path;
	while (std::getline(list, path))
	{
		if (!path.empty() && path.back() == '\r') path.pop_back();
		if (path.empty()) continue;
		if (g_should_exit) break;
		++n_total;

		// 图片优先走 GStreamer MPP JPEG 硬解，失败回退 OpenCV 软解
		cv::Mat img;
		GstVideoReader gst_reader;
		if (gst_reader.open(path))
		{
			if (!gst_reader.read(img)) img = cv::Mat();
			gst_reader.release();
		}
		if (img.empty()) img = cv::imread(path, cv::IMREAD_COLOR);
		if (img.empty())
		{
			LOG(MOD_MAIN, LOG_WARN) << "Failed to load image: " << path << "\n";
			++n_fail;
			continue;
		}

		std::vector<DetectResult> results;
		cv::Mat out;
		if (!pipeline.detect_image_reuse(img, results, args.output_path.empty() ? nullptr : &out))
		{
			LOG(MOD_MAIN, LOG_WARN) << "detect failed: " << path << "\n";
			++n_fail;
			continue;
		}
		++n_ok;
		n_det += (int)results.size();

		// 以文件名（去扩展名）作为图像标识，与 VisDrone 标注文件名一致
		size_t slash = path.find_last_of("/\\");
		std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
		size_t dot = base.find_last_of('.');
		std::string stem = (dot == std::string::npos) ? base : base.substr(0, dot);

		if (dump) dump_detections(dump, stem, results);
		if (!args.output_path.empty()) cv::imwrite(stem + "_det.jpg", out);

		if (n_total % 25 == 0)
		{
			double el = (now_us() - t_start) / 1e6;
			LOG(MOD_MAIN, LOG_INFO) << "[image-list] " << n_total << " images, ok=" << n_ok
			                        << " fail=" << n_fail << ", " << el << " s, "
			                        << (n_total / (el > 0 ? el : 1e-6)) << " img/s\n";
		}
	}

	double el = (now_us() - t_start) / 1e6;
	if (dump) dump.flush();
	LOG(MOD_MAIN, LOG_INFO) << "[image-list] done: total=" << n_total << " ok=" << n_ok
	                        << " fail=" << n_fail << " dets=" << n_det
	                        << " elapsed=" << el << " s ("
	                        << (n_total / (el > 0 ? el : 1e-6)) << " img/s)\n";
	return (n_fail == n_total && n_total > 0) ? 1 : 0;
}

// ============================================================================
// V4L2 零拷贝摄像头模式
// ============================================================================
// 将 V4L2 DMA 帧拷贝为 cv::Mat（用于画框/写视频；推理仍走 DMA 零拷贝路径）
static void v4l2_dma_to_mat(const DmaBufferPtr& buf, cv::Mat& out)
{
	if (!buf) return;
	out = cv::Mat();
	if (buf->format == RK_FORMAT_YUYV_422)
	{
		cv::Mat yuyv(buf->height, buf->width, CV_8UC2, buf->ptr, buf->stride);
		cv::cvtColor(yuyv, out, cv::COLOR_YUV2BGR_YUY2);
	}
	else if (buf->format == RK_FORMAT_BGR_888 || buf->format == RK_FORMAT_RGB_888)
	{
		out.create(buf->height, buf->width, CV_8UC3);
		uint8_t* d = out.data;
		const uint8_t* s = (const uint8_t*)buf->ptr;
		for (int y = 0; y < buf->height; ++y)
		{
			memcpy(d, s, (size_t)buf->width * 3);
			d += out.step;
			s += buf->stride;
		}
	}
}

int run_v4l2_mode(const Args& args, PipelineManager& pipeline,
                  V4l2ZeroCopyCapture& cap)
{
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
	rga_preprocessor().get_src_pool(actual_w, actual_h, cap.format());

	// 如果指定了输出视频，设置输出（使用 args.fps）
	if (!args.output_path.empty())
	{
		pipeline.set_video_output(args.output_path, args.fps);
	}

	LOG(MOD_MAIN, LOG_INFO) << "V4L2 capture started: "
	          << cap.width() << "x" << cap.height() << "\n";

	int frame_id = 0;
	int consecutive_empty = 0;
	while (!g_should_exit)
	{
		DmaBufferPtr src_buf = cap.read_frame();
		if (!src_buf)
		{
			// 无新帧（poll 超时/EAGAIN）或设备错误：仅首次与每 10 次提示一次，避免刷屏
			consecutive_empty++;
			if (consecutive_empty == 1 || consecutive_empty % 10 == 0)
			{
				LOG(MOD_MAIN, LOG_WARN) << "V4L2 read_frame returned no frame ("
				          << consecutive_empty << "x)\n";
			}
			continue;
		}
		consecutive_empty = 0;
		cv::Mat orig_img;
		v4l2_dma_to_mat(src_buf, orig_img);
		pipeline.push_dma_frame(frame_id++, src_buf, orig_img);
	}

	cap.stop();
	return 0;
}

// ============================================================================
// 视频文件模式
// ============================================================================
int run_video_mode(const Args& args, PipelineManager& pipeline)
{
	// 优先 GStreamer + RK MPP 硬件解码，失败回退 OpenCV 软解
	GstVideoReader gst_reader;
	bool use_gst = gst_reader.open(args.video_path);
	cv::VideoCapture cap;
	double fps = 30.0;
	int actual_w = 0, actual_h = 0;
	int frame_id = 0;
	cv::Mat frame;

	// -F/--fps 显式指定时：输入侧按指定速率限速喂帧，输出侧用指定 fps 写容器；
	// 未指定时保持默认行为（输入不限速，输出用源帧率）。
	double feed_fps = (args.fps_set && args.fps > 0) ? (double)args.fps : 0.0;
	auto feed_start = std::chrono::steady_clock::now();
	int64_t fed_frames = 0;
	auto pace_feed = [&]()
	{
		if (feed_fps <= 0) return;
		fed_frames++;
		auto target = feed_start + std::chrono::duration<double>(fed_frames / feed_fps);
		std::this_thread::sleep_until(target);
	};

	if (use_gst)
	{
		// 帧率来源：
		// - 容器/caps 提供有效帧率（CFR）→ 直接使用；
		// - VFR 或帧率元数据缺失（如 0/1）→ 两遍法：先纯解码统计容器时长/平均帧率
		//   （MPP 硬解，555 帧约 2-3s），再正常处理。避免"取前两帧间隔"在 VFR 源上
		//   误判（如 480×332 录屏前段 60fps burst → 输出 60fps → 时长 21.7s 被压成 9.2s）。
		double fps_est = gst_reader.fps();
		if (!args.fps_set && !gst_reader.caps_fps_authoritative())
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
		double out_fps = args.fps_set ? (double)args.fps : fps;
		if (args.fps_set)
		{
			LOG(MOD_MAIN, LOG_INFO) << "User --fps override: input feed & output fps = "
			          << args.fps << "\n";
		}
		if (!args.output_path.empty())
		{
			pipeline.set_video_output(args.output_path, out_fps);
		}
		pipeline.push_image(frame_id++, frame);
		pace_feed();
		if (has_second)
		{
			pipeline.push_image(frame_id++, frame2);
			pace_feed();
			while (!g_should_exit && gst_reader.read(frame))
			{
				pipeline.push_image(frame_id++, frame);
				pace_feed();
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
		double out_fps = args.fps_set ? (double)args.fps : fps;
		if (!args.output_path.empty())
		{
			pipeline.set_video_output(args.output_path, out_fps);
		}
		while (!g_should_exit && cap.read(frame))
		{
			pipeline.push_image(frame_id++, frame);
			pace_feed();
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

	// 创建流水线（图片/批量图片模式无需线程池 worker，走同步单帧接口）
	const bool single_image_mode = !args.image_path.empty() || !args.image_list_path.empty();
	int pre_workers  = single_image_mode ? 0 : args.pre_workers;
	int npu_workers  = single_image_mode ? 0 : args.npu_workers;
	int post_workers = single_image_mode ? 0 : args.post_workers;
	PipelineManager pipeline(pre_workers, npu_workers, post_workers,
	                         args.model_path,
	                         args.queue_cap, args.conf, args.npu_mask);
	pipeline.set_display(args.show_display);

	// 置信度阈值配置：全局 -c，可用 --profile / --conf-class 覆盖为按类阈值
	// （默认不启用按类阈值）
	{
		ConfConfig conf_cfg;
		conf_cfg.reset(args.conf);
		if (!args.profile.empty())
		{
			std::string err;
			if (!apply_conf_profile(args.profile, args.conf, conf_cfg, &err))
			{
				LOG(MOD_MAIN, LOG_ERROR) << "Error: " << err << "\n";
				return 1;
			}
		}
		if (!args.conf_class.empty())
		{
			std::string err;
			if (!parse_conf_class(args.conf_class, conf_cfg, &err))
			{
				LOG(MOD_MAIN, LOG_ERROR) << "Error: " << err << "\n";
				return 1;
			}
		}
		pipeline.set_conf_config(conf_cfg);
		if (conf_cfg.per_class_set)
		{
			std::ostringstream oss;
			oss << "Per-class confidence: global=" << conf_cfg.global;
			for (int c = 0; c < NUM_CLASSES; ++c)
			{
				oss << " " << CLASSES[c] << "=" << conf_cfg.per_class[c];
			}
			LOG(MOD_MAIN, LOG_INFO) << oss.str() << "\n";
		}
		else
		{
			LOG(MOD_MAIN, LOG_INFO) << "Confidence threshold (global): " << conf_cfg.global << "\n";
		}
	}
	pipeline.set_quit_callback([]()
	{
		g_should_exit = true;
	});

	// 输出视频由各采集模式按需调用 set_video_output 设置（不在此统一处理）

	// V4l2ZeroCopyCapture 必须在 PipelineManager 之前创建、之后析构：
	// 帧的 DmaBufferPtr deleter 会向采集对象归还 buffer，若采集对象先于
	// 流水线销毁（如 run_xxx_mode 局部对象），析构期会访问已销毁对象 → 崩溃。
	V4l2ZeroCopyCapture cap;

	// 选择采集模式
	int ret = 0;
	if (!args.image_path.empty())
	{
		ret = run_image_mode(args, pipeline);
	}
	else if (!args.image_list_path.empty())
	{
		ret = run_image_list_mode(args, pipeline);
	}
	else if (!args.video_path.empty())
	{
		ret = run_video_mode(args, pipeline);
	}
	else if (args.use_v4l2)
	{
		ret = run_v4l2_mode(args, pipeline, cap);
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
