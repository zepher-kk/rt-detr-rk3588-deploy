#include "npu_pipeline.h"
#include "rga_utils.h"
#include "postprocess.h"
#include "drm_alloc.h"

#include <iostream>
#include <chrono>
#include <algorithm>

static inline int64_t now_us()
{
	return std::chrono::duration_cast<std::chrono::microseconds>(
	           std::chrono::steady_clock::now().time_since_epoch()
	       ).count();
}

PerfCounter g_perf;

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
	  video_output_path_("result_video.mp4"),
	  video_fps_(30.0),
	  video_initialized_(false),
	  next_write_frame_id_(0),
	  is_running_(true)
{

	rga_preprocessor().init(num_npu + 4);

	for (int i = 0; i < num_pre; ++i)
		workers_pre_.emplace_back(&PipelineManager::worker_preprocess, this);
	for (int i = 0; i < num_npu; ++i)
		workers_npu_.emplace_back(&PipelineManager::worker_npu_infer, this, i);
	for (int i = 0; i < num_post; ++i)
		workers_post_.emplace_back(&PipelineManager::worker_postprocess, this);

	std::cerr << "[Pipeline] Started: pre=" << num_pre
	          << " npu=" << num_npu
	          << " post=" << num_post
	          << " queue_cap=" << queue_cap
	          << " npu_mask=" << npu_mask << "\n";
}

PipelineManager::~PipelineManager()
{
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

	is_running_ = false;
	queue_raw_.shutdown();
	queue_npu_.shutdown();
	queue_post_.shutdown();

	// 刷新视频缓存（写入所有剩余帧）
	flush_video_buffer();
}

// ============================================================================
// 输出产物config设置
// ============================================================================
void PipelineManager::set_video_output(const std::string& path, double fps)
{
	video_output_path_ = path;
	video_fps_ = (fps > 0) ? fps : 30.0;
	video_initialized_ = false;
	std::cerr << "[Pipeline] Video output set to: " << path
	          << " @ " << video_fps_ << " fps\n";
}

// ============================================================================
// 输入接口
// ============================================================================
void PipelineManager::push_dma_frame(int frame_id, const DmaBufferPtr& src_buf, const cv::Mat& orig_img)
{

	// fps_记录起始点_
	if (!started_)
	{
		start_time_ = std::chrono::steady_clock::now();
		started_ = true;
	}

	auto bundle = std::make_shared<FrameBundle>();
	bundle->frame_id = frame_id;
	bundle->src_buf = src_buf;
	bundle->use_dma_src = true;
	if (!orig_img.empty())
	{
		bundle->orig_img = orig_img.clone();   // ★ 关键：克隆图像
	}
	bundle->t_enqueue = now_us();
	queue_raw_.push(std::move(bundle));
}

void PipelineManager::push_image(int frame_id, const cv::Mat& img)
{

	// fps_记录起始点_
	if (!started_)
	{
		start_time_ = std::chrono::steady_clock::now();
		started_ = true;
	}

	auto bundle = std::make_shared<FrameBundle>();
	bundle->frame_id = frame_id;
	bundle->orig_img = img.clone();   // ★ 已经 clone，但保留
	bundle->use_dma_src = false;
	bundle->t_enqueue = now_us();
	queue_raw_.push(std::move(bundle));
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
			std::cerr << "[Pre] No valid source for frame " << bundle->frame_id << "\n";
			continue;
		}

		if (!bundle->input_buf)
		{
			std::cerr << "[Pre] RGA failed for frame " << bundle->frame_id << "\n";
			continue;
		}

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
		std::cerr << "[NPU] detector init failed with mask " << npu_mask_ << "\n";
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
			std::cerr << "[NPU] infer failed for frame " << bundle->frame_id << "\n";
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

		// ==================== 视频写入（核心逻辑） ====================
		if (!video_output_path_.empty())
		{
			std::lock_guard<std::mutex> lock(writer_mtx_);

			// 1. 若VideoWriter未初始化，尝试打开
			if (!video_initialized_ && !bundle->orig_img.empty())
			{
				cv::Size frame_size = bundle->orig_img.size();
				int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
				if (video_writer_.open(video_output_path_, fourcc, video_fps_, frame_size))
				{
					video_initialized_ = true;
					std::cerr << "[Pipeline] VideoWriter opened: "
					          << frame_size.width << "x" << frame_size.height
					          << " @ " << video_fps_ << " fps\n";
				}
				else
				{
					static bool warned = false;
					if (!warned)
					{
						std::cerr << "[Pipeline] Failed to open video writer: " << video_output_path_ << "\n";
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
			std::cerr << "[Pipeline] frame=" << n
			          << " pre=" << (total_pre_us_.load() / n / 1000) << "ms"
			          << " npu=" << (total_npu_us_.load() / n / 1000) << "ms"
			          << " post=" << (total_post_us_.load() / n / 1000) << "ms"
			          << " results=" << results.size() << "\n";

			// fps_display
			if (n % 30 == 0)
			{
				auto now = std::chrono::steady_clock::now();
				double elapsed = std::chrono::duration<double>(now - start_time_).count();
				double fps = n / elapsed;
				std::cerr << "[Pipeline] FPS: " << fps << " (frames=" << n << ", elapsed=" << elapsed << "s)\n";
			}
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
		int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
		if (video_writer_.open(video_output_path_, fourcc, video_fps_, frame_size))
		{
			video_initialized_ = true;
			std::cerr << "[Pipeline] VideoWriter opened during flush.\n";
		}
		else
		{
			std::cerr << "[Pipeline] Cannot open video writer during flush, dropping "
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
	std::cerr << "[Pipeline] Flushed " << count << " frames to video.\n";
}

// ============================================================================
// 等待空闲 & 打印性能
// ============================================================================
void PipelineManager::wait_idle()
{
	while (queue_raw_.size() > 0 || queue_npu_.size() > 0 || queue_post_.size() > 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void PipelineManager::print_perf_summary()
{
	int64_t n = frames_completed_.load();
	if (n == 0) return;
	auto now = std::chrono::steady_clock::now();
	double total_sec = std::chrono::duration<double>(now - start_time_).count();
	double overall_fps = n / total_sec;

	std::cerr << "\n========== Performance Summary ==========\n";
	std::cerr << "Frames completed    : " << n << "\n";
	std::cerr << "Total time (sec)    : " << total_sec << "\n";
	std::cerr << "Overall FPS         : " << overall_fps << "\n";
	std::cerr << "Avg preprocess      : " << (double)total_pre_us_.load() / n << " us\n";
	std::cerr << "Avg NPU infer       : " << (double)total_npu_us_.load() / n << " us\n";
	std::cerr << "Avg postprocess     : " << (double)total_post_us_.load() / n << " us\n";
	std::cerr << "==========================================\n";
}
