#pragma once

#include "types.h"
#include "drm_alloc.h"

#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>

// ============================================================================
// V4l2ZeroCopyCapture：V4L2 MMAP 零拷贝摄像头采集
//
// 【RGA_DMA 端到端零拷贝的关键起点】
//
// 传统 OpenCV VideoCapture 路径：
//   摄像头 → V4L2 内核驱动 → 用户态 buffer → cv::Mat（CPU 解码+拷贝）
//   ↑ 至少 2 次内存拷贝，CPU 负载高
//
// V4L2 MMAP 零拷贝路径（本实现）：
//   摄像头 → V4L2 内核驱动 → MMAP 映射的 DMA buffer
//   ↑ 内核与用户态共享同一块物理连续内存，零拷贝
//
// 【工作原理】
//   1. open("/dev/videoX") 打开 V4L2 设备
//   2. VIDIOC_QUERYCAP 查询能力
//   3. VIDIOC_S_FMT 协商格式（优先 BGR3 / RGB3 / YUYV；MJPG 等压缩格式走 OpenCV 回退）
//   4. VIDIOC_REQBUFS 请求 N 个 MMAP buffer（V4L2_MEMORY_MMAP）
//   5. VIDIOC_QBUF 入队所有 buffer（初始化）
//   6. VIDIOC_STREAMON 启动流
//   7. 采集循环：
//      - VIDIOC_DQBUF 取出一个已填充的 buffer（含 DMA fd）
//      - 包装为 DmaBufferPtr 送入流水线
//      - 流水线处理完后 VIDIOC_QBUF 归还
//
// 【DMA fd 获取】
//   V4L2 MMAP buffer 本身就是内核分配的物理连续内存，
//   通过 VIDIOC_EXPBUF 可导出为 PRIME fd，
//   该 fd 可直接传给 RGA（wrapbuffer_fd）和 NPU（rknn_create_mem_from_fd）。
//
// 【性能优势】
//   - 零拷贝：摄像头 → RGA → NPU 全程无 memcpy
//   - 低延迟：硬件直传，无 CPU 解码
//   - 低 CPU：仅 ioctl 系统调用开销
// ============================================================================

/**
 * @brief V4L2 零拷贝摄像头采集类。
 *
 * 使用 MMAP 和 EXPBUF 导出 PRIME fd，实现与 RGA/NPU 的零拷贝数据共享。
 */
class V4l2ZeroCopyCapture
{
	public:
		V4l2ZeroCopyCapture();
		~V4l2ZeroCopyCapture();

		/**
		 * @brief 打开摄像头设备并初始化。
		 * @param device 设备路径（如 /dev/video0）
		 * @param width  期望宽度
		 * @param height 期望高度
		 * @param fps    期望帧率
		 * @return true 成功
		 */
		bool open(const std::string& device, int width, int height, int fps = 30);

		/** @brief 启动视频流。 */
		bool start();

		/**
		 * @brief 读取一帧（阻塞）。
		 * @return DmaBufferPtr 包含图像数据，失败返回 nullptr
		 * @note 返回的 DMA 缓冲在 shared_ptr 析构时自动归还 V4L2 队列。
		 */
		DmaBufferPtr read_frame();

		/** @brief 停止流并释放资源。 */
		void stop();

		bool is_opened() const
		{
			return fd_ >= 0;
		}
		int width() const
		{
			return width_;
		}
		int height() const
		{
			return height_;
		}
		int format() const
		{
			return format_;    // RGA 格式常量
		}
		size_t stride() const
		{
			return stride_;
		}

	private:
		int     fd_         = -1;     // V4L2 设备 fd
		int     width_      = 0;
		int     height_     = 0;
		int     format_     = 0;      // RGA 格式（如 RK_FORMAT_BGR_888）
		size_t  stride_     = 0;
		int     buffer_count_ = 4;    // MMAP buffer 数量

		struct V4l2MmapBuffer
		{
			void*   start       = nullptr;
			size_t  length      = 0;
			int     v4l2_index  = -1;   // V4L2 buffer 索引
			int     dma_fd      = -1;   // EXPBUF 导出的 PRIME fd
		};
		std::vector<V4l2MmapBuffer>  mmap_buffers_;

		// 归还队列：已处理完的 V4L2 buffer 索引，等待 VIDIOC_QBUF
		std::queue<int>            recycle_queue_;
		mutable std::mutex         recycle_mtx_;
		std::condition_variable    recycle_cv_;

		bool    streaming_ = false;

		// 内部方法
		bool negotiate_format();
		bool request_mmap_buffers();
		bool export_dma_fds();
		void enqueue_recycle(int v4l2_index);
		void do_qbuf(int v4l2_index);
};
