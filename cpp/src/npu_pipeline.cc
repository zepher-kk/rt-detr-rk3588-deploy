#include "npu_pipeline.h"
#include "rga_utils.h"
#include "postprocess.h"
#include "drm_alloc.h"
#include "logger.h"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <rga.h>

static inline int64_t now_us()
{
	return std::chrono::duration_cast<std::chrono::microseconds>(
	           std::chrono::steady_clock::now().time_since_epoch()
	       ).count();
}

PerfCounter g_perf;

namespace
{
constexpr const char* kDisplayWindow = "RT-DETR";
}

// ============================================================================
// 构造 / 析构
// ============================================================================
PipelineManager::PipelineManager(int num_pre, int num_npu, int num_post,
                                 const std::string& model_path,
                                 size_t queue_cap,
                                 float conf_thres,
                                 rknn_core_mask npu_mask)
	: queue_raw_(queue_cap),
	  queue_npu_(queue_cap),
	  queue_post_(queue_cap),
	  model_path_(model_path),
	  num_npu_workers_(num_npu),
	  conf_thres_(conf_thres),
	  npu_mask_(npu_mask),
	  video_output_path_(""),   // 默认不输出视频；仅 -o 指定后写入（与 README 一致）
	  video_fps_(30.0),
	  video_initialized_(false),
	  next_write_frame_id_(0),
	  is_running_(true),
	  display_queue_(4)
{

	rga_preprocessor().init(num_npu + 4);

	for (int i = 0; i < num_pre; ++i)
		workers_pre_.emplace_back(&PipelineManager::worker_preprocess, this);
	for (int i = 0; i < num_npu; ++i)
		workers_npu_.emplace_back(&PipelineManager::worker_npu_infer, this, i);
	for (int i = 0; i < num_post; ++i)
		workers_post_.emplace_back(&PipelineManager::worker_postprocess, this);

	// 显示线程：无显示帧时阻塞在队列上，析构时 shutdown 唤醒退出
	display_thread_ = std::thread(&PipelineManager::display_worker, this);

	LOG(MOD_PIPELINE, LOG_INFO) << "Started: pre=" << num_pre
	          << " npu=" << num_npu
	          << " post=" << num_post
	          << " queue_cap=" << queue_cap
	          << " npu_mask=" << npu_mask << "\n";
}

PipelineManager::~PipelineManager()
{
	// 【健壮性】NPU 全部初始化失败时，先关闭队列再投毒丸：
	// 否则队列可能被未消费帧占满，析构的 push(nullptr) 会永久阻塞。
	if (!workers_ok_.load())
	{
		queue_raw_.shutdown();
		queue_npu_.shutdown();
		queue_post_.shutdown();
	}

	// 1. 终止预处理
	for (size_t i = 0; i < workers_pre_.size(); ++i)
		queue_raw_.push(nullptr);
	for (auto& t : workers_pre_) if (t.joinable()) t.join();

	// 2. 终止NPU
	for (size_t i = 0; i < workers_npu_.size(); ++i)
		queue_npu_.push(nullptr);
	for (auto& t : workers_npu_) if (t.joinable()) t.join();

	// 3. 终止后处理
	for (size_t i = 0; i < workers_post_.size(); ++i)
		queue_post_.push(nullptr);
	for (auto& t : workers_post_) if (t.joinable()) t.join();

	// 终止显示线程（shutdown 唤醒 pop）
	display_queue_.shutdown();
	if (display_thread_.joinable()) display_thread_.join();

	is_running_ = false;
	queue_raw_.shutdown();
	queue_npu_.shutdown();
	queue_post_.shutdown();

	// 刷新视频缓存（写入所有剩余帧）
	flush_video_buffer();
}

// ============================================================================
// 输出设置（视频输出路径与帧率）
// ============================================================================
void PipelineManager::set_video_output(const std::string& path, double fps)
{
	video_output_path_ = path;
	video_fps_ = (fps > 0) ? fps : 30.0;
	video_initialized_ = false;
	LOG(MOD_PIPELINE, LOG_INFO) << "Video output set to: " << path
	          << " @ " << video_fps_ << " fps\n";
}

void PipelineManager::set_display(bool enable)
{
	display_enabled_ = enable;
	LOG(MOD_PIPELINE, LOG_INFO) << "Real-time display " << (enable ? "enabled" : "disabled") << "\n";
}

// ============================================================================
// 输入接口
// ============================================================================
void PipelineManager::push_dma_frame(int frame_id, const DmaBufferPtr& src_buf, const cv::Mat& orig_img)
{
	// 【健壮性】NPU 不可用时快速丢帧，避免 reader 阻塞在满队列上无法退出
	if (!workers_ok_.load()) return;

	// 记录 FPS 统计起始时间（首次入帧时）
	if (!started_)
	{
		start_time_ = std::chrono::steady_clock::now();
		started_ = true;
	}

	auto bundle = std::make_shared<FrameBundle>();
	bundle->frame_id = frame_id;
	bundle->src_buf = src_buf;
	bundle->use_dma_src = true;
	bundle->pred_boxes.resize(NUM_BOXES * 4);
	bundle->pred_logits.resize(NUM_BOXES * NUM_CLASSES);
	if (!orig_img.empty())
	{
		bundle->orig_img = orig_img.clone();   // 克隆原始图像用于画框/写视频（DMA 源帧会被相机回收，须先复制）
	}
	bundle->t_enqueue = now_us();
	queue_raw_.push(std::move(bundle));
}

void PipelineManager::push_image(int frame_id, const cv::Mat& img)
{
	// 【健壮性】NPU 不可用时快速丢帧，避免 reader 阻塞在满队列上无法退出
	if (!workers_ok_.load()) return;

	// 记录 FPS 统计起始时间（首次入帧时）
	if (!started_)
	{
		start_time_ = std::chrono::steady_clock::now();
		started_ = true;
	}

	auto bundle = std::make_shared<FrameBundle>();
	bundle->frame_id = frame_id;
	bundle->orig_img = img.clone();   // 克隆输入图像，避免上游 cv::Mat 复用导致内容被覆写
	bundle->use_dma_src = false;
	bundle->pred_boxes.resize(NUM_BOXES * 4);
	bundle->pred_logits.resize(NUM_BOXES * NUM_CLASSES);
	bundle->t_enqueue = now_us();
	queue_raw_.push(std::move(bundle));
}

// ============================================================================
// 同步单帧图片检测（图片输入模式）
// ============================================================================
bool PipelineManager::detect_image(const cv::Mat& src, cv::Mat& out)
{
	if (src.empty())
	{
		LOG(MOD_PIPELINE, LOG_INFO) << "detect_image: empty input image\n";
		return false;
	}

	// 1. cv::Mat → 640x640 RGB DMA（RGA virt→DMA 主路径，失败自动回退 CPU）
	DmaBufferPtr input_buf = rga_preprocessor().preprocess_mat_to_dma(src);
	if (!input_buf)
	{
		LOG(MOD_PIPELINE, LOG_ERROR) << "detect_image: preprocess failed\n";
		return false;
	}

	// 2. NPU 推理（独立 context，与视频 worker 互不影响）
	RKNNDetector detector;
	if (!detector.init(model_path_, npu_mask_))
	{
		LOG(MOD_PIPELINE, LOG_ERROR) << "detect_image: detector init failed\n";
		return false;
	}
	std::vector<float> pred_boxes;
	std::vector<float> pred_logits;
	int num_boxes = 0;
	int num_classes = 0;
	if (!detector.infer_zero_copy(input_buf, pred_boxes, pred_logits, num_boxes, num_classes))
	{
		LOG(MOD_PIPELINE, LOG_ERROR) << "detect_image: inference failed\n";
		return false;
	}

	// 3. 后处理解码 + 画框
	std::vector<DetectResult> results = decode_rtdetr_output(
	                                      pred_boxes.data(), pred_logits.data(),
	                                      num_boxes, src.cols, src.rows,
	                                      conf_thres_, num_classes);
	out = src.clone();
	draw_results(out, results);

	LOG(MOD_PIPELINE, LOG_INFO) << "detect_image: " << results.size() << " targets\n";
	return true;
}

// ============================================================================
// 预处理 Worker（使用RGA零拷贝）
// ============================================================================
void PipelineManager::worker_preprocess()
{
	while (true)
	{
		FrameBundlePtr bundle;
		if (!queue_raw_.pop(bundle)) break;
		if (!bundle) break;

		int64_t t0 = now_us();

		if (bundle->use_dma_src && bundle->src_buf)
		{
			bundle->input_buf = rga_preprocessor().preprocess_dma_to_dma(bundle->src_buf);
		}
		else if (!bundle->orig_img.empty())
		{
			bundle->input_buf = rga_preprocessor().preprocess_mat_to_dma(bundle->orig_img);
		}
		else
		{
			LOGT(MOD_PIPELINE, LOG_ERROR, "Pre") << "No valid source for frame " << bundle->frame_id << "\n";
			continue;
		}

		if (!bundle->input_buf)
		{
			LOGT(MOD_PIPELINE, LOG_ERROR, "Pre") << "RGA failed for frame " << bundle->frame_id << "\n";
			continue;
		}

		// 提前释放源 DMA 缓冲：V4L2 相机 buffer 应尽快归还采集队列，
		// 否则源 buffer 会一直被帧包持有到后处理结束，相机因无 buffer 可用而降速。
		bundle->src_buf.reset();
		bundle->t_pre_done = now_us();
		total_pre_us_ += (bundle->t_pre_done - t0);
		queue_npu_.push(std::move(bundle));
	}
}

// ============================================================================
// NPU 推理 Worker（零拷贝）
// ============================================================================
void PipelineManager::worker_npu_infer(int /*core_id*/)
{
	RKNNDetector detector;
	if (!detector.init(model_path_, npu_mask_))
	{
		LOGT(MOD_PIPELINE, LOG_ERROR, "NPU") << "detector init failed with mask " << npu_mask_ << "\n";
		npu_init_failures_++;
		if (npu_init_failures_ >= num_npu_workers_) workers_ok_ = false;
		return;
	}

	while (true)
	{
		FrameBundlePtr bundle;
		if (!queue_npu_.pop(bundle)) break;
		if (!bundle) break;

		int64_t t0 = now_us();

		bool ok = detector.infer_zero_copy(
		              bundle->input_buf,
		              bundle->pred_boxes,
		              bundle->pred_logits,
		              bundle->num_boxes,
		              bundle->num_classes);

		if (!ok)
		{
			LOGT(MOD_PIPELINE, LOG_ERROR, "NPU") << "infer failed for frame " << bundle->frame_id << "\n";
			continue;
		}

		bundle->input_buf.reset();
		bundle->t_npu_done = now_us();
		total_npu_us_ += (bundle->t_npu_done - t0);
		queue_post_.push(std::move(bundle));
	}
}

// ============================================================================
// 后处理 Worker（视频写入等待连续帧，不跳过、不丢弃）
// ============================================================================
void PipelineManager::worker_postprocess()
{
	while (true)
	{
		FrameBundlePtr bundle;
		if (!queue_post_.pop(bundle)) break;
		if (!bundle)
		{
			flush_video_buffer();
			break;
		}

		int64_t t0 = now_us();

		// 解码检测结果
		std::vector<DetectResult> results = decode_rtdetr_output(
		                                        bundle->pred_boxes.data(),
		                                        bundle->pred_logits.data(),
		                                        bundle->num_boxes,
		                                        bundle->orig_img.cols,
		                                        bundle->orig_img.rows,
		                                        conf_thres_,
		                                        bundle->num_classes);

		if (!bundle->orig_img.empty())
		{
			draw_results(bundle->orig_img, results);
		}

		// 实时显示：丢旧保新队列，避免显示阻塞拖慢流水线
		if (display_enabled_.load() && !bundle->orig_img.empty())
		{
			display_queue_.push_drop_oldest(bundle->orig_img);
		}

		// ==================== 视频写入（核心逻辑） ====================
		if (!video_output_path_.empty())
		{
			std::lock_guard<std::mutex> lock(writer_mtx_);

			// 1. 若VideoWriter未初始化，尝试打开
			if (!video_initialized_ && !bundle->orig_img.empty())
			{
				cv::Size frame_size = bundle->orig_img.size();
				if (video_writer_.open(video_output_path_, video_fps_, frame_size))
				{
					video_initialized_ = true;
					LOG(MOD_PIPELINE, LOG_INFO) << "VideoWriter opened: "
					          << frame_size.width << "x" << frame_size.height
					          << " @ " << video_fps_ << " fps\n";
				}
				else
				{
					static bool warned = false;
					if (!warned)
					{
						LOG(MOD_PIPELINE, LOG_ERROR) << "Failed to open video writer: " << video_output_path_ << "\n";
						warned = true;
					}
				}
			}

			// 2. 始终缓存当前帧
			frame_buffer_[bundle->frame_id] = bundle;

			// 3. 若写入器已就绪，按序写入连续帧（遇缺失则停止等待）
			if (video_initialized_ && video_writer_.isOpened())
			{
				while (true)
				{
					auto it = frame_buffer_.find(next_write_frame_id_);
					if (it != frame_buffer_.end())
					{
						cv::Mat write_img = it->second->orig_img;
						if (!write_img.isContinuous()) write_img = write_img.clone();
						video_writer_.write(write_img);
						frame_buffer_.erase(it);
						next_write_frame_id_++;
					}
					else
					{
						// 缺失帧，停止写入，等待后续帧到达
						break;
					}
				}
			}
		}

		bundle->t_post_done = now_us();
		total_post_us_ += (bundle->t_post_done - t0);
		frames_completed_++;

		if (frames_completed_ % 30 == 0)
		{
			int64_t n = frames_completed_.load();
			LOGR(MOD_PIPELINE) << "frame=" << n
			          << " pre=" << (total_pre_us_.load() / n / 1000) << "ms"
			          << " npu=" << (total_npu_us_.load() / n / 1000) << "ms"
			          << " post=" << (total_post_us_.load() / n / 1000) << "ms"
			          << " results=" << results.size() << "\n";

			// 周期性打印实时 FPS
			if (n % 30 == 0)
			{
				auto now = std::chrono::steady_clock::now();
				double elapsed = std::chrono::duration<double>(now - start_time_).count();
				double fps = n / elapsed;
				LOGR(MOD_PIPELINE) << "FPS: " << fps << " (frames=" << n << ", elapsed=" << elapsed << "s)\n";
			}
		}

	}

}

// ============================================================================
// 实时显示线程
// ============================================================================
void PipelineManager::display_worker()
{
	while (true)
	{
		cv::Mat frame;
		if (!display_queue_.pop(frame)) break;
		if (frame.empty()) continue;
		try
		{
			cv::imshow(kDisplayWindow, frame);
			int key = cv::waitKey(1);
			if (key == 'q' || key == 27)
			{
				display_quit_ = true;
				if (quit_callback_) quit_callback_();
			}
		}
		catch (const cv::Exception& e)
		{
			// 无显示环境（headless）时 imshow 抛异常：记录并关闭显示，不中断检测
			LOG(MOD_PIPELINE, LOG_ERROR) << "Display error (headless?): " << e.what() << "\n";
			display_enabled_ = false;
			break;
		}
	}
}

// ============================================================================
// 刷新视频缓存（析构时调用，写入所有剩余帧）
// ============================================================================
void PipelineManager::flush_video_buffer()
{
	std::lock_guard<std::mutex> lock(writer_mtx_);
	if (frame_buffer_.empty()) return;

	// 若写入器未打开，尝试打开
	if (!video_initialized_ || !video_writer_.isOpened())
	{
		cv::Size frame_size = frame_buffer_.begin()->second->orig_img.size();
		if (video_writer_.open(video_output_path_, video_fps_, frame_size))
		{
			video_initialized_ = true;
			LOG(MOD_PIPELINE, LOG_INFO) << "VideoWriter opened during flush.\n";
		}
		else
		{
			LOG(MOD_PIPELINE, LOG_ERROR) << "Cannot open video writer during flush, dropping "
			          << frame_buffer_.size() << " frames.\n";
			frame_buffer_.clear();
			return;
		}
	}

	// 按 frame_id 排序
	std::vector<std::pair<int, FrameBundlePtr>> sorted_frames;
	sorted_frames.reserve(frame_buffer_.size());
	for (auto& kv : frame_buffer_)
		sorted_frames.emplace_back(kv.first, kv.second);
	std::sort(sorted_frames.begin(), sorted_frames.end(),
	          [](auto& a, auto& b)
	{
		return a.first < b.first;
	});

	// 写入所有缓存帧
	for (auto& [fid, fb] : sorted_frames)
	{
		if (fb && !fb->orig_img.empty())
		{
			cv::Mat write_img = fb->orig_img;
			if (!write_img.isContinuous()) write_img = write_img.clone();
			video_writer_.write(write_img);
		}
	}
	size_t count = sorted_frames.size();
	frame_buffer_.clear();
	video_writer_.release();
	LOG(MOD_PIPELINE, LOG_INFO) << "Flushed " << count << " frames to video.\n";
}

// ============================================================================
// 等待空闲 & 打印性能
// ============================================================================
void PipelineManager::wait_idle()
{
	// 【健壮性】有界等待：30s 上限 + NPU 熔断时立即返回，杜绝永久挂起
	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
	while (workers_ok_.load() &&
	       (queue_raw_.size() > 0 || queue_npu_.size() > 0 || queue_post_.size() > 0))
	{
		if (std::chrono::steady_clock::now() > deadline) break;
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

void PipelineManager::print_perf_summary()
{
	int64_t n = frames_completed_.load();
	if (n == 0) return;
	auto now = std::chrono::steady_clock::now();
	double total_sec = std::chrono::duration<double>(now - start_time_).count();
	double overall_fps = n / total_sec;

	LogStream ls(MOD_PIPELINE, LOG_INFO, nullptr, true);
	ls << "\n========== Performance Summary ==========\n";
	ls << "Frames completed    : " << n << "\n";
	ls << "Total time (sec)    : " << total_sec << "\n";
	ls << "Overall FPS         : " << overall_fps << "\n";
	ls << "Avg preprocess      : " << (double)total_pre_us_.load() / n << " us\n";
	ls << "Avg NPU infer       : " << (double)total_npu_us_.load() / n << " us\n";
	ls << "Avg postprocess     : " << (double)total_post_us_.load() / n << " us\n";
	ls << "==========================================\n";
}
