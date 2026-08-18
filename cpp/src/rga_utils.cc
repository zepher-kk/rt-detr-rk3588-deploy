#include "rga_utils.h"
#include "logger.h"

#include <iostream>
#include <cstring>
#include <chrono>
#include <mutex>
#include <algorithm>

// 声明全局性能计数器（定义在 npu_pipeline.cc 中）
extern PerfCounter g_perf;

// RGA 调用互斥锁：librga 2.2 在并发调用（多预处理线程）下偶发
// "RGA_BLIT fail: Invalid argument"，串行化 RGA 硬件调用可根治。
static std::mutex g_rga_mutex;

// ============================================================================
// YUYV(4:2:2) → BGR 转换（与 OpenCV COLOR_YUV2BGR_YUY2 一致，BT.601 limited range）
// 用于相机 YUYV DMA 帧在 RGA 不可用时的 CPU 回退路径。
// ============================================================================
void yuyv422_to_bgr(const uint8_t* src, size_t src_stride,
                    uint8_t* dst, size_t dst_stride,
                    int w, int h)
{
	for (int y = 0; y < h; ++y)
	{
		const uint8_t* s = src + (size_t)y * src_stride;
		uint8_t* d = dst + (size_t)y * dst_stride;
		for (int x = 0; x < w; x += 2)
		{
			int y0 = s[0], u = s[1], y1 = s[2], v = s[3];
			s += 4;
			for (int k = 0; k < 2; ++k)
			{
				int yy = (k == 0) ? y0 : y1;
				int c  = (yy - 16) * 1192 / 1024;
				int db = (u - 128) * 2066 / 1024;
				int dr = (v - 128) * 1634 / 1024;
				int dg = (u - 128) * 400 / 1024 + (v - 128) * 832 / 1024;
				*d++ = (uint8_t)std::max(0, std::min(255, c + db));
				*d++ = (uint8_t)std::max(0, std::min(255, c - dg));
				*d++ = (uint8_t)std::max(0, std::min(255, c + dr));
			}
		}
	}
}

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
		LOG(MOD_RGA, LOG_INFO) << "imcheckHeader returned " << init_status
		          << ", but continuing (may cause issues)\n";
	}
	else
	{
		LOG(MOD_RGA, LOG_INFO) << "RGA initialized, API version "
		          << RGA_API_VERSION << "\n";
	}

	dst_pool_ = std::make_unique<DmaBufferPool>(
	                INPUT_WIDTH, INPUT_HEIGHT, RK_FORMAT_RGB_888, pool_capacity);

	src_pool_ = nullptr;
	inited_ = true;
	LOG(MOD_RGA, LOG_INFO) << "Preprocessor initialized, dst pool capacity="
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
		LOG(MOD_RGA, LOG_ERROR) << "preprocess_dma_to_dma: not initialized or null src\n";
		return nullptr;
	}
	std::lock_guard<std::mutex> lock(g_rga_mutex);

	DmaBufferPtr dst = dst_pool_->alloc();
	if (!dst)
	{
		LOG(MOD_RGA, LOG_ERROR) << "dst pool alloc failed\n";
		return nullptr;
	}

	// CPU 回退：RGA 无法包装/转换源时，按实际 stride 逐行读取 DMA 内存完成
	// resize+cvtColor。支持 BGR/RGB/YUYV（相机 YUYV 帧走此路径或 RGA YUYV 路径）。
	auto cpu_fallback = [this](const DmaBufferPtr& src_buf) -> DmaBufferPtr
	{
		const int fmt = src_buf->format;
		if (fmt != RK_FORMAT_BGR_888 && fmt != RK_FORMAT_RGB_888 &&
		    fmt != RK_FORMAT_YUYV_422)
		{
			LOG(MOD_RGA, LOG_ERROR) << "CPU fallback only supports BGR/RGB/YUYV src, format="
			          << fmt << "\n";
			return nullptr;
		}

		DmaBufferPtr fdst = dst_pool_->alloc();
		if (!fdst)
		{
			LOG(MOD_RGA, LOG_ERROR) << "dst pool alloc failed (cpu fallback)\n";
			return nullptr;
		}

		cv::Mat src_mat;
		if (fmt == RK_FORMAT_YUYV_422)
		{
			src_mat.create(src_buf->height, src_buf->width, CV_8UC3);
			yuyv422_to_bgr((const uint8_t*)src_buf->ptr, src_buf->stride,
			               src_mat.data, (size_t)src_mat.step,
			               src_buf->width, src_buf->height);
		}
		else
		{
			src_mat = cv::Mat(src_buf->height, src_buf->width, CV_8UC3);
			uint8_t* mrow = src_mat.data;
			const uint8_t* srow = (const uint8_t*)src_buf->ptr;
			const size_t row_bytes = (size_t)src_buf->width * 3;
			for (int y = 0; y < src_buf->height; ++y)
			{
				memcpy(mrow, srow, row_bytes);
				mrow += src_mat.step;
				srow += src_buf->stride;
			}
		}

		cv::resize(src_mat, src_mat, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
		if (fmt != RK_FORMAT_RGB_888)
			cv::cvtColor(src_mat, src_mat, cv::COLOR_BGR2RGB);

		uint8_t* d = (uint8_t*)fdst->ptr;
		const uint8_t* s = src_mat.data;
		for (int y = 0; y < INPUT_HEIGHT; ++y)
		{
			memcpy(d, s, INPUT_WIDTH * 3);
			d += fdst->stride;
			s += src_mat.step;
		}
		fdst->format = RK_FORMAT_RGB_888;
		g_perf.total_virt_to_dma++;
		return fdst;
	};

	rga_buffer_t src_rga = {};
	rga_buffer_t dst_rga = {};
	bool src_ok = false, dst_ok = false;

	// 源有效 wstride = 实际 stride / bpp。
	// 3 字节格式下 DRM 对齐 stride（如 1360 宽 → 4096，4080%3!=0）无法被 RGA wstride(像素) 表达，
	// 此时 src_wstride=0，跳过 RGA 包装，走下方 CPU 回退，杜绝逐行偏移污染。
	int src_bpp = rga_format_bpp(src_buf->format);
	int src_wstride = (src_buf->stride > 0 && src_bpp > 0 && src_buf->stride % src_bpp == 0)
	                  ? (int)(src_buf->stride / src_bpp) : 0;

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
	            LOG(MOD_RGA, LOG_INFO) << "Method1 src OK\n";
	        } else {
	            releasebuffer_handle(src_handle);
	        }
	    }
	*/

	// ---------- 尝试方法2: wrapbuffer_fd_t（wstride 用实际 stride，避免对齐污染）----------
	if (src_wstride > 0 && !src_ok)
	{
		src_rga = wrapbuffer_fd_t(src_buf->fd, src_buf->width, src_buf->height,
		                          src_wstride, src_buf->height,  // wstride=实际像素行宽
		                          src_buf->format);
		if (src_rga.vir_addr != nullptr || src_rga.fd > 0)
		{
			src_ok = true;
		}
	}

	// ---------- 尝试方法3: wrapbuffer_virtualaddr (备选) ----------
	if (src_wstride > 0 && !src_ok)
	{
		src_rga = wrapbuffer_virtualaddr_t(src_buf->ptr, src_buf->width, src_buf->height,
		                                   src_wstride, src_buf->height,
		                                   src_buf->format);
		if (src_rga.vir_addr != nullptr)
		{
			src_ok = true;
		}
	}

	if (!src_ok)
	{
		// 安全回退：CPU 按实际 stride 逐行读取 DMA 内存完成 resize+cvtColor。
		// 适用：源 stride 与 width*bpp 不对齐（如 1360 宽 BGR → stride 4096）时，
		// RGA 无法以像素 wstride 表达该内存布局，走此路径杜绝逐行偏移污染。
		dst.reset();
		return cpu_fallback(src_buf);
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
	            LOG(MOD_RGA, LOG_INFO) << "Method1 dst OK\n";
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
			LOG(MOD_RGA, LOG_INFO) << "Method3 dst OK\n";
		}
	}

	if (!dst_ok)
	{
		LOG(MOD_RGA, LOG_ERROR) << "All dst wrapping methods failed\n";
		return nullptr;
	}

	// ---------- 执行 resize (BGR -> BGR) ----------
	IM_STATUS status = imresize(src_rga, dst_rga);
	if (status != IM_STATUS_SUCCESS)
	{
		LOG(MOD_RGA, LOG_WARN) << "imresize failed: " << status << ", fallback CPU\n";
		dst.reset();
		return cpu_fallback(src_buf);
	}

	// ---------- 颜色转换 BGR -> RGB ----------
	status = imcvtcolor(dst_rga, dst_rga, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
	if (status != IM_STATUS_SUCCESS)
	{
		LOG(MOD_RGA, LOG_WARN) << "imcvtcolor failed: " << status << ", fallback CPU\n";
		dst.reset();
		return cpu_fallback(src_buf);
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
		LOG(MOD_RGA, LOG_ERROR) << "preprocess_mat_to_dma: not initialized or empty src\n";
		return nullptr;
	}

	// 首选 RGA virt→DMA 硬件路径：
	//   源 = cv::Mat 用户态连续内存（紧致 stride=width*3，RGA wstride 可精确表达），
	//   目标 = 640x640 RGB DMA（stride=1920 == 640*3）。
	//   消除 CPU resize/cvtColor 与 Mat→DMA 桥接 memcpy，降低 CPU 占用；
	//   同时规避 DRM 源缓冲 stride 对齐污染问题（非对齐宽度如 1360→4096）。
	if (src.isContinuous())
	{
		std::lock_guard<std::mutex> lock(g_rga_mutex);
		DmaBufferPtr dst = dst_pool_->alloc();
		if (dst)
		{
			rga_buffer_t src_rga = wrapbuffer_virtualaddr_t((void*)src.data, src.cols, src.rows,
			                                                src.cols, src.rows, RK_FORMAT_BGR_888);
			rga_buffer_t dst_rga = wrapbuffer_fd_t(dst->fd, INPUT_WIDTH, INPUT_HEIGHT,
			                                       INPUT_WIDTH, INPUT_HEIGHT, RK_FORMAT_BGR_888);
			if (src_rga.vir_addr != nullptr && dst_rga.fd > 0)
			{
				IM_STATUS st = imresize(src_rga, dst_rga);
				if (st == IM_STATUS_SUCCESS)
				{
					st = imcvtcolor(dst_rga, dst_rga, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
					if (st == IM_STATUS_SUCCESS)
					{
						dst->format = RK_FORMAT_RGB_888;
						g_perf.total_virt_to_dma++;
						return dst;
					}
				}
			}
			static bool rga_virt_warned = false;
			if (!rga_virt_warned)
			{
				LOG(MOD_RGA, LOG_WARN) << "RGA virt→DMA failed, fallback to CPU path\n";
				rga_virt_warned = true;
			}
			dst.reset();
		}
	}

	// 回退路径：CPU resize + BGR→RGB 后紧致写入 640x640 DMA（stride 安全）
	cv::Mat resized;
	cv::resize(src, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
	cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

	DmaBufferPtr dst = dst_pool_->alloc();
	if (!dst)
	{
		LOG(MOD_RGA, LOG_ERROR) << "dst pool alloc failed\n";
		return nullptr;
	}

	// 紧致拷贝（dst stride 恒为 640*3=1920，与 resized 行宽一致）
	uint8_t* dst_ptr = (uint8_t*)dst->ptr;
	const uint8_t* src_ptr = resized.data;
	const size_t row_bytes = (size_t)resized.cols * resized.elemSize();
	for (int y = 0; y < resized.rows; ++y)
	{
		memcpy(dst_ptr, src_ptr, row_bytes);
		dst_ptr += dst->stride;
		src_ptr += resized.step;
	}

	dst->format = RK_FORMAT_RGB_888;
	g_perf.total_virt_to_dma++;
	return dst;
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
		LOG(MOD_RGA, LOG_ERROR) << "bridge: failed to alloc src DMA buffer\n";
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
