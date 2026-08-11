#pragma once

#include "types.h"
#include "drm_alloc.h"

// ============================================================================
// RgaPreprocessor：RGA_DMA 硬件加速预处理
//
//     src = wrapbuffer_fd(DMA fd)  ← 物理地址，纯 DMA
//     dst = wrapbuffer_fd(DMA fd)  ← 物理地址，纯 DMA
//     imresize_then_cvtcolor(src, dst, BGR, RGB)  ← 单 pass 完成 resize+cvt
//     ↑ RGA 硬件直接 DMA 搬运，零 CPU 负载，零 IOMMU 开销
//
// 【性能对比】（1080p→640p，30fps）：
//   CPU cv::resize+cvtColor :  ~8ms/帧，CPU 100% 占用
//   RGA virt→DMA           :  ~1.5ms/帧，CPU ~5%（ioctl 开销）
//   RGA DMA→DMA（本版）     :  ~0.8ms/帧，CPU ~2%（纯 ioctl）
//
// 【RGA_DMA 技术要点】（可参考 Neardi wiki RGA DMA 章节）：
//   1. 源/目标均通过 wrapbuffer_fd 包装为 rga_buffer_t
//   2. wrapbuffer_fd 内部用 PRIME fd 查询物理地址，RGA 直接 DMA
//   3. imcheck 预检参数合法性，避免硬件错误
//   4. imresize_then_cvtcolor 单 pass 完成 resize + 颜色转换
//   5. 同步等待 imsync 确保硬件完成（默认同步模式）
// ============================================================================

/**
 * @brief RGA 硬件加速预处理器，实现零拷贝 DMA→DMA 缩放和颜色转换。
 *
 * 提供三种使用模式：
 *  - DMA→DMA（主路径，最高性能）
 *  - cv::Mat→DMA（回退路径，源端走 IOMMU）
 *  - 桥接 cv::Mat→DMA（仅一次拷贝，后续走 DMA 路径）
 */
class RgaPreprocessor
{
	public:
		RgaPreprocessor();
		~RgaPreprocessor();

		/**
		 * @brief 初始化预处理器，分配目标 DMA 池（640x640 RGB）。
		 * @param pool_capacity 目标池容量（建议等于 NPU 线程数+4）
		 * @return true 成功
		 */
		bool init(size_t pool_capacity = 8);

		/**
		 * @brief 核心接口：DMA→DMA 零拷贝预处理。
		 * @param src_buf 源 DMA 缓冲（任意分辨率 BGR）
		 * @return 目标 DMA 缓冲（640x640 RGB），可直接喂给 NPU
		 * @note 全程硬件加速，零 CPU 负载。
		 */
		DmaBufferPtr preprocess_dma_to_dma(const DmaBufferPtr& src_buf);

		/**
		 * @brief 兼容接口：cv::Mat→DMA 预处理。
		 * @param src 输入图像（BGR）
		 * @return 目标 DMA 缓冲（640x640 RGB）
		 * @note 内部先将 cv::Mat 拷贝到临时 DMA，再走 DMA→DMA 路径，性能低于主路径。
		 */
		DmaBufferPtr preprocess_mat_to_dma(const cv::Mat& src);

		/**
		 * @brief 桥接接口：将 cv::Mat 拷贝到 DMA 缓冲（用于视频文件等场景）。
		 * @param src       输入图像
		 * @param src_pool  源尺寸 DMA 池（由调用方提供）
		 * @return DMA 缓冲，之后可复用 preprocess_dma_to_dma
		 */
		DmaBufferPtr bridge_mat_to_dma(const cv::Mat& src, DmaBufferPool& src_pool);

		/** @brief 释放所有资源。 */
		void release();

		/** @brief 获取源 DMA 池（用于桥接模式）。 */
		DmaBufferPool& src_pool()
		{
			return *src_pool_;
		}
		/** @brief 获取目标 DMA 池。 */
		DmaBufferPool& dst_pool()
		{
			return *dst_pool_;
		}

		/**
		 * @brief 获取指定尺寸和格式的源 DMA 池（懒创建）。
		 * @param width   宽度
		 * @param height  高度
		 * @param format  RGA 格式
		 * @return 引用到 DmaBufferPool
		 */
		DmaBufferPool& get_src_pool(int width, int height, int format);

	private:
		std::unique_ptr<DmaBufferPool> dst_pool_;   // 640x640 RGB 输出池
		std::unique_ptr<DmaBufferPool> src_pool_;   // 源尺寸 BGR 输入池（桥接模式用）
		bool inited_ = false;
};

/**
 * @brief 全局 RGA 预处理器单例（线程安全，懒加载）。
 * @return 单例引用
 */
RgaPreprocessor& rga_preprocessor();
