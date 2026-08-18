#include "drm_alloc.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <xf86drm.h>
#include <cstring>
#include <iostream>
#include <chrono>
#include <errno.h>
#include "logger.h"

// ============================================================================
// RGA 格式常量（来自 rga.h）
// ============================================================================
#include <rga.h>
// #include <RgaUtils.h>

// ============================================================================
// 全局 DRM render node fd（懒加载单例）
// ============================================================================
static int g_drm_fd = -1;

int get_drm_fd()
{
	if (g_drm_fd >= 0) return g_drm_fd;

	// 尝试 renderD128 → renderD129 → renderD130 → card0
	const char* paths[] =
	{
		"/dev/dri/renderD128",
		"/dev/dri/renderD129",
		"/dev/dri/renderD130",
		"/dev/dri/card0",
		nullptr
	};

	for (int i = 0; paths[i]; ++i)
	{
		int flags = O_RDWR | O_CLOEXEC;
		// card0 需要 root 权限，render node 不需要
		if (std::string(paths[i]).find("card") != std::string::npos)
		{
			flags |= O_RDWR;
		}
		g_drm_fd = open(paths[i], flags);
		if (g_drm_fd >= 0)
		{
			// 验证是否支持 dumb buffer
			uint64_t has_dumb = 0;
			if (drmGetCap(g_drm_fd, DRM_CAP_DUMB_BUFFER, &has_dumb) == 0 && has_dumb)
			{
				LOG(MOD_DRM, LOG_INFO) << "Opened " << paths[i]
				          << " fd=" << g_drm_fd << "\n";
				return g_drm_fd;
			}
			close(g_drm_fd);
			g_drm_fd = -1;
		}
	}

	LOG(MOD_DRM, LOG_ERROR) << "FATAL: cannot open any /dev/dri/renderD12X or card0\n"
	          << "Please ensure:\n"
	          << "  1. Device is RK3588 with DRM support\n"
	          << "  2. User has permission to access /dev/dri/*\n"
	          << "  3. Run: sudo usermod -aG render $USER\n";
	return -1;
}

// ============================================================================
// 工具函数：根据 RGA 格式计算每像素字节数
// ============================================================================
int rga_format_bpp(int format)
{
	switch (format)
	{
		case RK_FORMAT_RGB_888:
		case RK_FORMAT_BGR_888:
			return 3;

		case RK_FORMAT_RGBA_8888:
			return 4;

		case RK_FORMAT_YUYV_422:
			return 2;

		case RK_FORMAT_YCbCr_420_SP:   // case RK_FORMAT_NV12: NV12 对应 YUV420 半平面
			return 1;                  // Y plane 每像素1字节，UV合起来也平均1字节

		default:
			return 3;                  // 默认按 3 字节每像素处理（RGB888/BGR888 等）
	}
}

// ============================================================================
// 工具函数：根据宽度与每像素字节数计算 DRM 对齐的 stride
// RK3588 DRM dumb buffer 默认对齐到 16 字节（某些场景 64 字节）
// ============================================================================
size_t drm_aligned_stride(int width, int bpp)
{
	size_t raw = (size_t)width * bpp;
	// 对齐到 64 字节（RK3588 DRM 默认对齐）
	const size_t alignment = 64;
	return (raw + alignment - 1) & ~(alignment - 1);
}

// ============================================================================
// DmaBuffer 析构与移动语义
// ============================================================================
DmaBuffer::~DmaBuffer()
{
	if (ptr && ptr != MAP_FAILED && ptr != nullptr)
	{
		munmap(ptr, size);
		ptr = nullptr;
	}
	if (fd >= 0)
	{
		close(fd);
		fd = -1;
	}
	// 释放 DRM GEM handle（防止内核内存泄漏）
	if (handle != 0 && drm_fd >= 0)
	{
		struct drm_mode_destroy_dumb destroy = {};
		destroy.handle = handle;
		ioctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
		handle = 0;
	}
}

DmaBuffer::DmaBuffer(DmaBuffer&& o) noexcept
	: fd(o.fd), ptr(o.ptr), size(o.size),
	  width(o.width), height(o.height), format(o.format),
	  stride(o.stride), handle(o.handle), drm_fd(o.drm_fd)
{
	o.fd = -1;
	o.ptr = nullptr;
	o.size = 0;
	o.handle = 0;
	o.drm_fd = -1;
}

DmaBuffer& DmaBuffer::operator=(DmaBuffer&& o) noexcept
{
	if (this != &o)
	{
		// 先释放自己的资源
		if (ptr && ptr != MAP_FAILED) munmap(ptr, size);
		if (fd >= 0) close(fd);
		if (handle != 0 && drm_fd >= 0)
		{
			struct drm_mode_destroy_dumb destroy = {};
			destroy.handle = handle;
			ioctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
		}

		// 接管 o 的资源
		fd = o.fd;
		ptr = o.ptr;
		size = o.size;
		width = o.width;
		height = o.height;
		format = o.format;
		stride = o.stride;
		handle = o.handle;
		drm_fd = o.drm_fd;

		// 清空 o
		o.fd = -1;
		o.ptr = nullptr;
		o.size = 0;
		o.handle = 0;
		o.drm_fd = -1;
	}
	return *this;
}

// ============================================================================
// DmaBufferPool 实现
// ============================================================================
DmaBufferPool::DmaBufferPool(int width, int height, int format, size_t capacity)
	: width_(width), height_(height), format_(format), capacity_(capacity)
{
	int bpp = rga_format_bpp(format);
	stride_ = drm_aligned_stride(width, bpp);
	LOG(MOD_DRM, LOG_INFO) << "Pool created: " << width << "x" << height
	          << " bpp=" << bpp << " stride=" << stride_
	          << " capacity=" << capacity << "\n";
}

DmaBufferPool::~DmaBufferPool()
{
	std::lock_guard<std::mutex> lock(mtx_);
	for (auto* buf : free_list_)
	{
		delete buf;
	}
	free_list_.clear();
}

// ============================================================================
// 内部分配：DRM dumb buffer + PRIME fd + mmap
// ============================================================================
bool DmaBufferPool::alloc_one_drm(DmaBuffer& out)
{
	int drm_fd = get_drm_fd();
	if (drm_fd < 0) return false;

	// 获取每像素字节数
	int bpp = rga_format_bpp(format_);
	if (bpp <= 0) bpp = 3;  // fallback

	// 1. 创建 dumb buffer
	struct drm_mode_create_dumb create_req = {};
	create_req.width  = width_;
	create_req.height = height_;
	create_req.bpp    = bpp * 8;  // 每像素位数

	if (drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) < 0)
	{
		LOG(MOD_DRM, LOG_ERROR) << "CREATE_DUMB failed: " << strerror(errno) << "\n";
		return false;
	}

	// 2. 导出 PRIME fd
	struct drm_prime_handle prime_req = {};
	prime_req.handle = create_req.handle;
	prime_req.flags  = DRM_CLOEXEC | O_RDWR;

	if (drmIoctl(drm_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime_req) < 0)
	{
		LOG(MOD_DRM, LOG_ERROR) << "PRIME_HANDLE_TO_FD failed: " << strerror(errno) << "\n";
		struct drm_mode_destroy_dumb destroy = {};
		destroy.handle = create_req.handle;
		drmIoctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
		return false;
	}
	int prime_fd = prime_req.fd;

	// 3. mmap 映射 CPU 虚拟地址
	struct drm_mode_map_dumb map_req = {};
	map_req.handle = create_req.handle;
	if (drmIoctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req) < 0)
	{
		LOG(MOD_DRM, LOG_ERROR) << "MAP_DUMB failed: " << strerror(errno) << "\n";
		close(prime_fd);
		struct drm_mode_destroy_dumb destroy = {};
		destroy.handle = create_req.handle;
		drmIoctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
		return false;
	}

	void* map_ptr = mmap(nullptr, create_req.size,
	                     PROT_READ | PROT_WRITE, MAP_SHARED,
	                     drm_fd, map_req.offset);
	if (map_ptr == MAP_FAILED)
	{
		LOG(MOD_DRM, LOG_ERROR) << "mmap failed: " << strerror(errno) << "\n";
		close(prime_fd);
		struct drm_mode_destroy_dumb destroy = {};
		destroy.handle = create_req.handle;
		drmIoctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
		return false;
	}

	// 4. 填充 DmaBuffer
	out.fd      = prime_fd;
	out.ptr     = map_ptr;
	out.size    = create_req.size;
	out.width   = width_;
	out.height  = height_;
	out.format  = format_;
	out.stride  = create_req.pitch;  // DRM 返回的实际 stride（对齐后）
	out.handle  = create_req.handle;
	out.drm_fd  = drm_fd;

	// 记录分配信息（含 DRM 实际 stride）
	LOG(MOD_DRM, LOG_INFO) << "Allocated: size=" << out.size << " stride=" << out.stride << "\n";

	return true;
}

DmaBufferPtr DmaBufferPool::alloc()
{
	std::unique_lock<std::mutex> lock(mtx_);

	// 优先从空闲队列取
	if (!free_list_.empty())
	{
		DmaBuffer* raw = free_list_.back();
		free_list_.pop_back();
		// 返回 shared_ptr，自定义 deleter 归还到池
		return DmaBufferPtr(raw, [this](DmaBuffer* b)
		{
			recycle(b);
		});
	}

	// 池空：新建一个
	DmaBuffer* raw = new DmaBuffer();
	if (!alloc_one_drm(*raw))
	{
		delete raw;
		return nullptr;
	}
	return DmaBufferPtr(raw, [this](DmaBuffer* b)
	{
		recycle(b);
	});
}

void DmaBufferPool::recycle(DmaBuffer* buf)
{
	if (!buf) return;
	std::lock_guard<std::mutex> lock(mtx_);
	free_list_.push_back(buf);
}

size_t DmaBufferPool::size() const
{
	std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mtx_));
	return free_list_.size();
}
