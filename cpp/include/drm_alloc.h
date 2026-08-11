#pragma once

#include "types.h"

#include <vector>
#include <mutex>
#include <condition_variable>

// ============================================================================
// DmaBufferPool：DMA 缓冲池（RGA_DMA 基础设施）
//
// 设计动机（来自 Neardi RGA DMA 文档核心思想）：
//   - DRM dumb buffer 分配需要 ioctl 系统调用，开销 ~50us
//   - 视频帧率 30fps 时，每秒需 30 个缓冲，频繁分配会拖垮性能
//   - 预分配固定数量缓冲，循环复用，避免 malloc/free 抖动
//
// 内存生命周期（RGA_DMA 零拷贝路径）：
//   1. pool.alloc()           → 从空闲队列取一个 DmaBuffer（无可用则新建）
//   2. V4L2/RGA 写入数据       → wrapbuffer_fd(fd, ...) 硬件操作，纯物理地址 DMA
//   3. NPU 读取数据            → rknn_create_mem_from_fd(ctx, fd, virt, size, 0) 零拷贝
//   4. pool.recycle(buf)      → shared_ptr 析构时自动归还，等待下一帧复用
//
// 关键技术点（Neardi wiki DMA buffer 章节）：
//   - 使用 DRM_IOCTL_MODE_CREATE_DUMB 分配物理连续内存
//   - 使用 DRM_IOCTL_PRIME_HANDLE_TO_FD 导出为 PRIME fd
//   - PRIME fd 是跨硬件共享的标准机制（RGA/NPU/V4L2/GPU 均支持）
//   - mmap 后获得虚拟地址，CPU 可访问（但热路径不读写，避免 cache 抖动）
// ============================================================================

/**
 * @brief DMA 缓冲池，管理预分配的物理连续内存块。
 *
 * 使用 DRM dumb buffer 和 PRIME fd 实现跨硬件零拷贝。
 * 缓冲池固定容量，分配/回收无动态内存开销。
 */
class DmaBufferPool
{
	public:
		/**
		 * @brief 构造缓冲池。
		 * @param width    缓冲宽度（像素）
		 * @param height   缓冲高度（像素）
		 * @param format   RGA 格式常量（如 RK_FORMAT_BGR_888）
		 * @param capacity 池中最多缓存多少个空闲缓冲（预分配数量）
		 */
		DmaBufferPool(int width, int height, int format, size_t capacity = 8);
		~DmaBufferPool();

		/**
		 * @brief 从池中获取一个空闲 DMA 缓冲。
		 * @return shared_ptr<DmaBuffer>，析构时自动归还到池。
		 * @note 若池空则新建，但总数不会超过 capacity_（内部自动扩容至 capacity）。
		 */
		DmaBufferPtr alloc();

		/**
		 * @brief 回收缓冲（由 shared_ptr 析构时自动调用）。
		 * @param buf 要归还的 DmaBuffer 指针
		 */
		void recycle(DmaBuffer* buf);

		size_t size() const;            //!< 当前空闲缓冲数
		size_t capacity() const
		{
			return capacity_;
		}

		int width()  const
		{
			return width_;
		}
		int height() const
		{
			return height_;
		}
		int format() const
		{
			return format_;
		}
		size_t stride() const
		{
			return stride_;
		}

	private:
		int     width_;
		int     height_;
		int     format_;
		size_t  capacity_;
		size_t  stride_;        // DRM 分配的行跨度（对齐到 16/32 字节）

		std::vector<DmaBuffer*> free_list_;
		mutable std::mutex      mtx_;
		std::condition_variable cv_;

		// 内部分配：调用 DRM ioctl 创建 dumb buffer + 导出 PRIME fd + mmap
		bool alloc_one_drm(DmaBuffer& out);
};

/**
 * @brief 获取全局 DRM render node fd（懒加载，进程级单例）。
 * @return DRM 文件描述符，失败返回 -1
 * @note RK3588 上通常为 /dev/dri/renderD128 或 /dev/dri/card0
 */
int get_drm_fd();

/**
 * @brief 根据 RGA 格式计算每像素字节数。
 * @param format RGA 格式常量
 * @return 每像素字节数（如 RGB888 返回 3）
 */
int rga_format_bpp(int format);

/**
 * @brief 根据宽度和每像素字节数计算 DRM 对齐的 stride。
 * @param width 图像宽度（像素）
 * @param bpp   每像素字节数
 * @return 对齐后的 stride（字节），RK3588 默认为 64 字节对齐
 */
size_t drm_aligned_stride(int width, int bpp);
