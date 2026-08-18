#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <map>
#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "types.h"
#include "rknn_detector.h"
#include "rknn_api.h"
#include "gst_io.h"

// ============================================================================
// BoundedSafeQueue（有界阻塞队列）
// ============================================================================
/**
 * @brief 线程安全的有界阻塞队列，支持毒丸（nullptr）终止。
 * @tparam T 存储的元素类型
 */
template <typename T>
class BoundedSafeQueue
{
	private:
		std::queue<T>        queue_;
		mutable std::mutex   mtx_;
		std::condition_variable cv_not_full_;
		std::condition_variable cv_not_empty_;
		size_t               capacity_;
		std::atomic<bool>    is_running_{true};

	public:
		explicit BoundedSafeQueue(size_t cap) : capacity_(cap) {}

		/** @brief 关闭队列，唤醒所有等待线程。 */
		void shutdown()
		{
			is_running_ = false;
			cv_not_full_.notify_all();
			cv_not_empty_.notify_all();
		}

		/**
		 * @brief 入队一个任务，若队列满则阻塞。
		 * @param task 要推入的任务（支持移动语义）
		 * @note 若队列已关闭，任务会被丢弃。
		 */
		void push(T task)
		{
			std::unique_lock<std::mutex> lock(mtx_);
			cv_not_full_.wait(lock, [this]()
			{
				return queue_.size() < capacity_ || !is_running_.load();
			});
			if (!is_running_.load()) return;
			queue_.push(std::move(task));
			lock.unlock();
			cv_not_empty_.notify_one();
		}

		/**
		 * @brief 入队一个元素，队满时丢弃最旧元素，绝不阻塞调用方。
		 * @param task 要推入的元素
		 * @note 用于实时显示等低时延场景：保证最新帧优先，避免显示拖慢流水线。
		 */
		void push_drop_oldest(T task)
		{
			std::unique_lock<std::mutex> lock(mtx_);
			if (capacity_ > 0 && queue_.size() >= capacity_)
			{
				queue_.pop();
			}
			queue_.push(std::move(task));
			lock.unlock();
			cv_not_empty_.notify_one();
		}

		/**
		 * @brief 出队一个任务，若队列空则阻塞。
		 * @param task 输出参数，接收元素
		 * @return true 成功取出，false 队列已关闭且为空
		 */
		bool pop(T& task)
		{
			std::unique_lock<std::mutex> lock(mtx_);
			cv_not_empty_.wait(lock, [this]()
			{
				return !queue_.empty() || !is_running_.load();
			});
			if (!is_running_.load() && queue_.empty()) return false;
			task = std::move(queue_.front());
			queue_.pop();
			lock.unlock();
			cv_not_full_.notify_one();
			return true;
		}

		size_t size() const
		{
			std::lock_guard<std::mutex> lock(mtx_);
			return queue_.size();
		}
};

// ============================================================================
// PipelineManager
// ============================================================================
/**
 * @brief 三级流水线管理器（预处理 → NPU推理 → 后处理/视频输出）。
 *
 * 采用有界队列解耦各阶段，支持多线程并行。
 * 视频输出按帧 ID 顺序写入，确保不会丢帧或乱序。
 */
class PipelineManager
{
	private:
		BoundedSafeQueue<FrameBundlePtr> queue_raw_;
		BoundedSafeQueue<FrameBundlePtr> queue_npu_;
		BoundedSafeQueue<FrameBundlePtr> queue_post_;

		std::vector<std::thread> workers_pre_;
		std::vector<std::thread> workers_npu_;
		std::vector<std::thread> workers_post_;

		std::atomic<bool> is_running_{true};
		// 【健壮性】NPU worker 初始化失败熔断：全部失败时输入直接丢帧，避免队列卡死
		std::atomic<bool> workers_ok_{true};
		std::atomic<int>  npu_init_failures_{0};
		std::string       model_path_;
		int               num_npu_workers_;
		float             conf_thres_ = 0.45f;
		rknn_core_mask    npu_mask_;          // NPU核心掩码

		// 视频输出（延迟初始化）
		std::string       video_output_path_;
		double            video_fps_ = 30.0;
		bool              video_initialized_ = false;
		GstVideoWriter    video_writer_;
		std::map<int, FrameBundlePtr> frame_buffer_;   // 缓存未按序写入的帧
		int               next_write_frame_id_ = 0;     // 期望写入的下一个帧 ID
		std::mutex        writer_mtx_;
		std::atomic<bool> video_flushed_{false};

		// 性能统计
		std::atomic<int64_t> frames_completed_{0};
		std::atomic<int64_t> total_pre_us_{0};
		std::atomic<int64_t> total_npu_us_{0};
		std::atomic<int64_t> total_post_us_{0};

		// 性能统计 ② :: fps
		std::chrono::steady_clock::time_point start_time_;
		bool started_ = false;

		// 实时显示：专用线程 + 丢旧保新队列，显示不阻塞流水线
		std::atomic<bool> display_enabled_{false};
		std::atomic<bool> display_quit_{false};
		BoundedSafeQueue<cv::Mat> display_queue_;
		std::thread display_thread_;
		std::function<void()> quit_callback_;   // 显示窗口按 q/ESC 时通知主流程退出

		// function
		void worker_preprocess();
		void worker_npu_infer(int core_id);
		void worker_postprocess();
		void display_worker();
		void flush_video_buffer();  // 析构时强制写入所有缓存帧

	public:
		/**
		 * @brief 构造流水线管理器。
		 * @param num_pre    预处理线程数
		 * @param num_npu    NPU推理线程数（每个线程独立加载模型）
		 * @param num_post   后处理线程数
		 * @param model_path RKNN 模型文件路径
		 * @param queue_cap  各队列容量
		 * @param conf_thres 检测置信度阈值
		 * @param npu_mask   NPU 核心掩码（多核分配策略）
		 */
		PipelineManager(int num_pre, int num_npu, int num_post,
		                const std::string& model_path,
		                size_t queue_cap = 16,
		                float conf_thres = 0.45f,
		                rknn_core_mask npu_mask = RKNN_NPU_CORE_AUTO);
		~PipelineManager();

		/**
		 * @brief 输入 DMA 帧（零拷贝路径）。
		 * @param frame_id  帧序号（用于输出排序）
		 * @param src_buf   DMA 缓冲（来自 V4L2 或桥接）
		 * @param orig_img  原始图像（用于画框，可为空）
		 */
		void push_dma_frame(int frame_id, const DmaBufferPtr& src_buf, const cv::Mat& orig_img = cv::Mat());

		/**
		 * @brief 输入 cv::Mat 图像（兼容路径，内部会转为 DMA）。
		 * @param frame_id  帧序号
		 * @param img       输入图像（BGR）
		 */
		void push_image(int frame_id, const cv::Mat& img);

		/**
		 * @brief 同步单帧图片检测（图片输入模式）。
		 *
		 * 内部完成 cv::Mat → DMA 桥接、RGA 预处理、NPU 推理、后处理画框，
		 * 不经过线程池队列，便于单张图片端到端测时延。
		 * @param src 输入图片（BGR）
		 * @param out 输出图片（绘制检测框与类别标签）
		 * @return true 成功；false 输入为空/模型不可用/推理失败
		 */
		bool detect_image(const cv::Mat& src, cv::Mat& out);

		/**
		 * @brief 设置输出视频文件路径。
		 * @param path 输出文件路径（如 .mp4）
		 * @param fps  输出帧率
		 */
		void set_video_output(const std::string& path, double fps = 30.0);

		/**
		 * @brief 启用/关闭实时检测画面显示。
		 * @param enable true 时由专用线程调用 cv::imshow + waitKey 播放检测帧。
		 */
		void set_display(bool enable);

		/** @brief 用户是否通过显示窗口请求退出（按 q 或 ESC）。 */
		bool display_quit_requested() const
		{
			return display_quit_.load();
		}

		/** @brief 注册显示窗口退出回调（通常置全局退出标志）。 */
		void set_quit_callback(std::function<void()> cb)
		{
			quit_callback_ = std::move(cb);
		}

		/** @brief 等待所有队列处理完毕（用于优雅退出）。 */
		void wait_idle();

		/** @brief 打印性能汇总（平均耗时、总 FPS）。 */
		void print_perf_summary();
};
