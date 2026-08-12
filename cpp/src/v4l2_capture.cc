#include "v4l2_capture.h"
#include "logger.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <linux/dma-buf.h>
#include <cstring>
#include <iostream>
#include <chrono>
#include <errno.h>

// RGA 格式常量
#include <rga.h>

// V4L2 四字符码（FourCC）
#ifndef V4L2_PIX_FMT_BGR24
#define V4L2_PIX_FMT_BGR24 v4l2_fourcc('B', 'G', 'R', '3')
#endif
#ifndef V4L2_PIX_FMT_RGB24
#define V4L2_PIX_FMT_RGB24 v4l2_fourcc('R', 'G', 'B', '3')
#endif

// ============================================================================
// V4l2ZeroCopyCapture 实现
// ============================================================================
V4l2ZeroCopyCapture::V4l2ZeroCopyCapture() {}
V4l2ZeroCopyCapture::~V4l2ZeroCopyCapture()
{
	stop();
}

bool V4l2ZeroCopyCapture::open(const std::string& device,
                               int width, int height, int fps)
{
	// 1. 打开 V4L2 设备
	fd_ = ::open(device.c_str(), O_RDWR | O_CLOEXEC);
	if (fd_ < 0)
	{
		LOG(MOD_V4L2, LOG_ERROR) << "Cannot open " << device << ": "
		          << strerror(errno) << "\n";
		return false;
	}

	// 2. 查询设备能力
	struct v4l2_capability cap = {};
	if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0)
	{
		LOG(MOD_V4L2, LOG_ERROR) << "VIDIOC_QUERYCAP failed: " << strerror(errno) << "\n";
		stop();
		return false;
	}

	if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE))
	{
		LOG(MOD_V4L2, LOG_ERROR) << "Device does not support video capture\n";
		stop();
		return false;
	}

	if (!(cap.capabilities & V4L2_CAP_STREAMING))
	{
		LOG(MOD_V4L2, LOG_ERROR) << "Device does not support streaming\n";
		stop();
		return false;
	}

	LOG(MOD_V4L2, LOG_INFO) << "Device: " << cap.card
	          << " driver: " << cap.driver << "\n";

	width_  = width;
	height_ = height;

	// 3. 协商格式
	if (!negotiate_format())
	{
		stop();
		return false;
	}

	// 4. 设置帧率
	struct v4l2_streamparm parm = {};
	parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	parm.parm.capture.timeperframe.numerator   = 1;
	parm.parm.capture.timeperframe.denominator = fps;
	ioctl(fd_, VIDIOC_S_PARM, &parm);

	// 5. 请求 MMAP buffer
	if (!request_mmap_buffers())
	{
		stop();
		return false;
	}

	// 6. 导出 DMA fd
	if (!export_dma_fds())
	{
		stop();
		return false;
	}

	LOG(MOD_V4L2, LOG_INFO) << "Opened: " << width_ << "x" << height_
	          << " stride=" << stride_
	          << " buffers=" << mmap_buffers_.size() << "\n";

	return true;
}

// ----------------------------------------------------------------------------
// 协商格式
// ----------------------------------------------------------------------------
bool V4l2ZeroCopyCapture::negotiate_format()
{
	struct v4l2_format fmt = {};
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width       = width_;
	fmt.fmt.pix.height      = height_;
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24;  // 优先 BGR3
	fmt.fmt.pix.field       = V4L2_FIELD_NONE;

	if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0)
	{
		LOG(MOD_V4L2, LOG_ERROR) << "VIDIOC_S_FMT failed: " << strerror(errno) << "\n";
		return false;
	}

	// 检查实际协商结果
	width_  = fmt.fmt.pix.width;
	height_ = fmt.fmt.pix.height;
	stride_ = fmt.fmt.pix.bytesperline;

	// 映射 V4L2 格式到 RGA 格式
	if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_BGR24)
	{
		format_ = RK_FORMAT_BGR_888;
	}
	else if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_RGB24)
	{
		format_ = RK_FORMAT_RGB_888;
	}
	else
	{
		LOG(MOD_V4L2, LOG_ERROR) << "Unsupported format: "
		          << char(fmt.fmt.pix.pixelformat & 0xFF)
		          << char((fmt.fmt.pix.pixelformat >> 8) & 0xFF)
		          << char((fmt.fmt.pix.pixelformat >> 16) & 0xFF)
		          << char((fmt.fmt.pix.pixelformat >> 24) & 0xFF)
		          << "\n";
		return false;
	}

	return true;
}

// ----------------------------------------------------------------------------
// 请求 MMAP buffer
// ----------------------------------------------------------------------------
bool V4l2ZeroCopyCapture::request_mmap_buffers()
{
	struct v4l2_requestbuffers req = {};
	req.count  = buffer_count_;
	req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;

	if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0)
	{
		LOG(MOD_V4L2, LOG_ERROR) << "VIDIOC_REQBUFS failed: " << strerror(errno) << "\n";
		return false;
	}

	buffer_count_ = req.count;
	mmap_buffers_.resize(buffer_count_);

	for (int i = 0; i < buffer_count_; ++i)
	{
		struct v4l2_buffer buf = {};
		buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index  = i;

		if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0)
		{
			LOG(MOD_V4L2, LOG_ERROR) << "VIDIOC_QUERYBUF failed: " << strerror(errno) << "\n";
			return false;
		}

		mmap_buffers_[i].start      = nullptr;
		mmap_buffers_[i].length     = buf.length;
		mmap_buffers_[i].v4l2_index = i;
		mmap_buffers_[i].dma_fd     = -1;

		// mmap
		mmap_buffers_[i].start = mmap(nullptr, buf.length,
		                              PROT_READ | PROT_WRITE, MAP_SHARED,
		                              fd_, buf.m.offset);
		if (mmap_buffers_[i].start == MAP_FAILED)
		{
			LOG(MOD_V4L2, LOG_ERROR) << "mmap failed: " << strerror(errno) << "\n";
			return false;
		}
	}

	return true;
}

// ----------------------------------------------------------------------------
// 导出 DMA fd（PRIME fd）
// ----------------------------------------------------------------------------
bool V4l2ZeroCopyCapture::export_dma_fds()
{
	for (int i = 0; i < buffer_count_; ++i)
	{
		struct v4l2_exportbuffer expbuf = {};
		expbuf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		expbuf.index  = i;
		expbuf.plane   = 0;
		expbuf.flags  = O_CLOEXEC;

		if (ioctl(fd_, VIDIOC_EXPBUF, &expbuf) < 0)
		{
			LOG(MOD_V4L2, LOG_ERROR) << "VIDIOC_EXPBUF failed: " << strerror(errno) << "\n";
			return false;
		}

		mmap_buffers_[i].dma_fd = expbuf.fd;
	}
	return true;
}

// ----------------------------------------------------------------------------
// 启动采集
// ----------------------------------------------------------------------------
bool V4l2ZeroCopyCapture::start()
{
	// 入队所有 buffer
	for (int i = 0; i < buffer_count_; ++i)
	{
		struct v4l2_buffer buf = {};
		buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index  = i;
		if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0)
		{
			LOG(MOD_V4L2, LOG_ERROR) << "VIDIOC_QBUF failed: " << strerror(errno) << "\n";
			return false;
		}
	}

	// 启动流
	enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0)
	{
		LOG(MOD_V4L2, LOG_ERROR) << "VIDIOC_STREAMON failed: " << strerror(errno) << "\n";
		return false;
	}

	streaming_ = true;
	return true;
}

// ----------------------------------------------------------------------------
// 取一帧：返回 DMA buffer（零拷贝）
// ----------------------------------------------------------------------------
DmaBufferPtr V4l2ZeroCopyCapture::read_frame()
{
	if (!streaming_) return nullptr;

	// 先归还所有待回收的 buffer
	{
		std::lock_guard<std::mutex> lock(recycle_mtx_);
		while (!recycle_queue_.empty())
		{
			do_qbuf(recycle_queue_.front());
			recycle_queue_.pop();
		}
	}

	// 取一帧
	struct v4l2_buffer buf = {};
	buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;

	if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0)
	{
		if (errno == EAGAIN) return nullptr;
		LOG(MOD_V4L2, LOG_ERROR) << "VIDIOC_DQBUF failed: " << strerror(errno) << "\n";
		return nullptr;
	}

	int idx = buf.index;
	if (idx < 0 || idx >= (int)mmap_buffers_.size())
	{
		LOG(MOD_V4L2, LOG_ERROR) << "Invalid buffer index: " << idx << "\n";
		return nullptr;
	}

	// 创建 DmaBuffer 包装
	DmaBuffer* raw = new DmaBuffer();
	raw->fd      = mmap_buffers_[idx].dma_fd;
	raw->ptr     = mmap_buffers_[idx].start;
	raw->size    = mmap_buffers_[idx].length;
	raw->width   = width_;
	raw->height  = height_;
	raw->format  = format_;
	raw->stride  = stride_;
	raw->handle  = 0;  // V4L2 buffer 无 DRM handle
	raw->drm_fd  = -1;

	// 返回 shared_ptr，析构时归还 V4L2 buffer
	return DmaBufferPtr(raw, [this, idx](DmaBuffer* b)
	{
		delete b;
		enqueue_recycle(idx);
	});
}

// ----------------------------------------------------------------------------
// 归还 V4L2 buffer
// ----------------------------------------------------------------------------
void V4l2ZeroCopyCapture::enqueue_recycle(int v4l2_index)
{
	{
		std::lock_guard<std::mutex> lock(recycle_mtx_);
		recycle_queue_.push(v4l2_index);
	}
	recycle_cv_.notify_one();
}

void V4l2ZeroCopyCapture::do_qbuf(int v4l2_index)
{
	struct v4l2_buffer buf = {};
	buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.index  = v4l2_index;
	ioctl(fd_, VIDIOC_QBUF, &buf);
}

// ----------------------------------------------------------------------------
// 停止采集并释放资源
// ----------------------------------------------------------------------------
void V4l2ZeroCopyCapture::stop()
{
	if (streaming_)
	{
		enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		ioctl(fd_, VIDIOC_STREAMOFF, &type);
		streaming_ = false;
	}

	// 处理完所有待归还的 buffer
	{
		std::lock_guard<std::mutex> lock(recycle_mtx_);
		while (!recycle_queue_.empty())
		{
			do_qbuf(recycle_queue_.front());
			recycle_queue_.pop();
		}
	}

	// 注意：dma_fd 由 V4L2 内核管理，close 后内核会自动回收
	// mmap 的内存也由内核管理，munmap 后自动回收
	// 这里只需关闭设备 fd
	for (auto& mb : mmap_buffers_)
	{
		if (mb.start && mb.start != MAP_FAILED)
		{
			munmap(mb.start, mb.length);
			mb.start = nullptr;
		}
		// dma_fd 是 EXPBUF 导出的，需要 close
		if (mb.dma_fd >= 0)
		{
			close(mb.dma_fd);
			mb.dma_fd = -1;
		}
	}
	mmap_buffers_.clear();

	if (fd_ >= 0)
	{
		close(fd_);
		fd_ = -1;
	}
}
