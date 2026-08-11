#include "rga_utils.h"

#include <iostream>
#include <cstring>
#include <chrono>

// 声明全局性能计数器（定义在 npu_pipeline.cc 中）
extern PerfCounter g_perf;

// ============================================================================
// RGA
// ============================================================================
#include <RgaUtils.h>
#include <im2d.hpp>
#include <rga.h>

// ============================================================================
// RgaPreprocessor 实现
// ============================================================================
RgaPreprocessor::RgaPreprocessor() {}
RgaPreprocessor::~RgaPreprocessor()
{
	release();
}

bool RgaPreprocessor::init(size_t pool_capacity)
{
	if (inited_) return true;

	// 初始化 RGA 设备（调用 imcheckHeader 触发初始化）
	IM_STATUS init_status = imcheckHeader(RGA_CURRENT_API_HEADER_VERSION);
	if (init_status != IM_STATUS_NOERROR)
	{
		std::cerr << "[RGA] imcheckHeader returned " << init_status
		          << ", but continuing (may cause issues)\n";
	}
	else
	{
		std::cerr << "[RGA] RGA initialized, API version "
		          << RGA_API_VERSION << "\n";
	}

	dst_pool_ = std::make_unique<DmaBufferPool>(
	                INPUT_WIDTH, INPUT_HEIGHT, RK_FORMAT_RGB_888, pool_capacity);

	src_pool_ = nullptr;
	inited_ = true;
	std::cerr << "[RGA] Preprocessor initialized, dst pool capacity="
	          << pool_capacity << "\n";
	return true;
}

void RgaPreprocessor::release()
{
	if (!inited_) return;
	src_pool_.reset();
	dst_pool_.reset();
	inited_ = false;
}

// ============================================================================
// DMA→DMA 零拷贝预处理
// ============================================================================
DmaBufferPtr RgaPreprocessor::preprocess_dma_to_dma(const DmaBufferPtr& src_buf)
{
	if (!inited_ || !src_buf)
	{
		std::cerr << "[RGA] preprocess_dma_to_dma: not initialized or null src\n";
		return nullptr;
	}

	DmaBufferPtr dst = dst_pool_->alloc();
	if (!dst)
	{
		std::cerr << "[RGA] dst pool alloc failed\n";
		return nullptr;
	}

	rga_buffer_t src_rga = {};
	rga_buffer_t dst_rga = {};
	bool src_ok = false, dst_ok = false;

	// ---------- 尝试方法1: importbuffer_fd + wrapbuffer_handle_t 【目前不可行】----------
	/*
	    rga_buffer_handle_t src_handle = importbuffer_fd(src_buf->fd, src_buf->width,
	                                                     src_buf->height, src_buf->format);
	    if (src_handle) {
	        src_rga = wrapbuffer_handle_t(src_handle, src_buf->width, src_buf->height,
	                                      src_buf->width, src_buf->height,  // wstride=width, hstride=height
	                                      src_buf->format);
	        if (src_rga.vir_addr != nullptr || src_rga.fd > 0) {
	            src_ok = true;
	            std::cerr << "[RGA] Method1 src OK\n";
	        } else {
	            releasebuffer_handle(src_handle);
	        }
	    }
	*/

	// ---------- 尝试方法2: wrapbuffer_fd_t (直接调用C函数) ----------
	if (!src_ok)
	{
		src_rga = wrapbuffer_fd_t(src_buf->fd, src_buf->width, src_buf->height,
		                          src_buf->width, src_buf->height,  // wstride=width, hstride=height
		                          src_buf->format);
		if (src_rga.vir_addr != nullptr || src_rga.fd > 0)
		{
			src_ok = true;
			// std::cerr << "[RGA] Method2 src OK\n";
		}
	}

	// ---------- 尝试方法3: wrapbuffer_virtualaddr (备选) ----------
	if (!src_ok)
	{
		src_rga = wrapbuffer_virtualaddr_t(src_buf->ptr, src_buf->width, src_buf->height,
		                                   src_buf->width, src_buf->height,
		                                   src_buf->format);
		if (src_rga.vir_addr != nullptr)
		{
			src_ok = true;
			std::cerr << "[RGA] Method3 src OK\n";
		}
	}

	if (!src_ok)
	{
		std::cerr << "[RGA] All src wrapping methods failed\n";
		return nullptr;
	}

	// ---------- 目标 buffer 同样尝试三种方法 ----------
	int dst_format = RK_FORMAT_BGR_888;  // resize 时格式统一为 BGR

	// 方法1 - importbuffer_fd + wrapbuffer_handle_t 【目前不可行】
	/*
	    rga_buffer_handle_t dst_handle = importbuffer_fd(dst->fd, INPUT_WIDTH, INPUT_HEIGHT, dst_format);
	    if (dst_handle) {
	        dst_rga = wrapbuffer_handle_t(dst_handle, INPUT_WIDTH, INPUT_HEIGHT,
	                                      INPUT_WIDTH, INPUT_HEIGHT,  // wstride=width, hstride=height
	                                      dst_format);
	        if (dst_rga.vir_addr != nullptr || dst_rga.fd > 0) {
	            dst_ok = true;
	            std::cerr << "[RGA] Method1 dst OK\n";
	        } else {
	            releasebuffer_handle(dst_handle);
	        }
	    }
	*/

	// 方法2
	if (!dst_ok)
	{
		dst_rga = wrapbuffer_fd_t(dst->fd, INPUT_WIDTH, INPUT_HEIGHT,
		                          INPUT_WIDTH, INPUT_HEIGHT,
		                          dst_format);
		if (dst_rga.vir_addr != nullptr || dst_rga.fd > 0)
		{
			dst_ok = true;
			// std::cerr << "[RGA] Method2 dst OK\n";
		}
	}

	// 方法3
	if (!dst_ok)
	{
		dst_rga = wrapbuffer_virtualaddr_t(dst->ptr, INPUT_WIDTH, INPUT_HEIGHT,
		                                   INPUT_WIDTH, INPUT_HEIGHT,
		                                   dst_format);
		if (dst_rga.vir_addr != nullptr)
		{
			dst_ok = true;
			std::cerr << "[RGA] Method3 dst OK\n";
		}
	}

	if (!dst_ok)
	{
		std::cerr << "[RGA] All dst wrapping methods failed\n";
		return nullptr;
	}

	// ---------- 执行 resize (BGR -> BGR) ----------
	IM_STATUS status = imresize(src_rga, dst_rga);
	if (status != IM_STATUS_SUCCESS)
	{
		std::cerr << "[RGA] imresize failed: " << status << "\n";
		return nullptr;
	}

	// ---------- 颜色转换 BGR -> RGB ----------
	status = imcvtcolor(dst_rga, dst_rga, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
	if (status != IM_STATUS_SUCCESS)
	{
		std::cerr << "[RGA] imcvtcolor failed: " << status << "\n";
		return nullptr;
	}

	// 更新 dst 格式为 RGB
	dst->format = RK_FORMAT_RGB_888;

	g_perf.total_dma_to_dma++;
	return dst;
}

// ============================================================================
// cv::Mat→DMA 预处理
// ============================================================================
DmaBufferPtr RgaPreprocessor::preprocess_mat_to_dma(const cv::Mat& src)
{
	if (!inited_ || src.empty())
	{
		std::cerr << "[RGA] preprocess_mat_to_dma: not initialized or empty src\n";
		return nullptr;
	}

	// 避免使用局部临时池，改用全局池
	DmaBufferPool& temp_src_pool = get_src_pool(src.cols, src.rows, RK_FORMAT_BGR_888);
	DmaBufferPtr src_dma = bridge_mat_to_dma(src, temp_src_pool);
	if (!src_dma)
	{
		std::cerr << "[RGA] bridge_mat_to_dma failed\n";
		return nullptr;
	}

	// 调用 DMA→DMA 硬件加速路径
	return preprocess_dma_to_dma(src_dma);
}

// ============================================================================
// 桥接接口
// ============================================================================
DmaBufferPtr RgaPreprocessor::bridge_mat_to_dma(const cv::Mat& src,
        DmaBufferPool& src_pool)
{
	if (src.empty()) return nullptr;

	DmaBufferPtr dma_buf = src_pool.alloc();
	if (!dma_buf)
	{
		std::cerr << "[RGA] bridge: failed to alloc src DMA buffer\n";
		return nullptr;
	}

	int bpp = 3;
	size_t src_row_bytes = (size_t)src.cols * bpp;
	size_t dst_stride = dma_buf->stride;

	if (dst_stride == src_row_bytes)
	{
		memcpy(dma_buf->ptr, src.data, src_row_bytes * src.rows);
	}
	else
	{
		uint8_t* dst_ptr = (uint8_t*)dma_buf->ptr;
		const uint8_t* src_ptr = src.data;
		for (int y = 0; y < src.rows; ++y)
		{
			memcpy(dst_ptr, src_ptr, src_row_bytes);
			dst_ptr += dst_stride;
			src_ptr += src.step;
		}
	}

	return dma_buf;
}


DmaBufferPool& RgaPreprocessor::get_src_pool(int width, int height, int format)
{
	if (!src_pool_)
	{
		src_pool_ = std::make_unique<DmaBufferPool>(width, height, format, 8);
	}
	else
	{
		// 可选：检查尺寸格式是否匹配，若不匹配则重新创建或调整
		// 简单起见，假设调用时参数一致，否则重建
		if (src_pool_->width() != width || src_pool_->height() != height || src_pool_->format() != format)
		{
			src_pool_ = std::make_unique<DmaBufferPool>(width, height, format, 8);
		}
	}
	return *src_pool_;
}


// ============================================================================
// 全局单例
// ============================================================================
RgaPreprocessor& rga_preprocessor()
{
	static RgaPreprocessor inst;
	return inst;
}
