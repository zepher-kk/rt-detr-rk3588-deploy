#include "drm_alloc.h"

#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <rga.h>

static int g_drm_fd = -1;

int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static bool probe_node(int fd) {
    uint64_t has_dumb = 0;
    if (drmGetCap(fd, DRM_CAP_DUMB_BUFFER, &has_dumb) != 0 || !has_dumb) return false;

    struct drm_mode_create_dumb cr = {};
    cr.width = 64; cr.height = 64; cr.bpp = 24;
    if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &cr) < 0) return false;

    struct drm_mode_map_dumb mr = {};
    mr.handle = cr.handle;
    bool ok = false;
    if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &mr) == 0) {
        void* p = mmap(nullptr, cr.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mr.offset);
        if (p != MAP_FAILED) { munmap(p, cr.size); ok = true; }
    }
    struct drm_mode_destroy_dumb d = {};
    d.handle = cr.handle;
    drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
    return ok;
}

int get_drm_fd() {
    if (g_drm_fd >= 0) return g_drm_fd;
    // Some boards expose several DRM nodes; only some support dumb-buffer mmap.
    // Probe create+map instead of trusting DRM_CAP_DUMB_BUFFER alone.
    const char* paths[] = {"/dev/dri/renderD128", "/dev/dri/renderD129",
                           "/dev/dri/card1",     "/dev/dri/card0",
                           "/dev/dri/card2",     nullptr};
    for (int i = 0; paths[i]; ++i) {
        int fd = open(paths[i], O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        if (probe_node(fd)) {
            g_drm_fd = fd;
            fprintf(stderr, "[drm] using %s fd=%d\n", paths[i], fd);
            return fd;
        }
        close(fd);
    }
    fprintf(stderr, "[drm] FATAL: no usable dumb-buffer node under /dev/dri\n");
    return -1;
}

int rga_format_bpp(int format) {
    switch (format) {
        case RK_FORMAT_RGB_888:
        case RK_FORMAT_BGR_888: return 3;
        case RK_FORMAT_RGBA_8888: return 4;
        case RK_FORMAT_YUYV_422: return 2;
        default: return 3;
    }
}

size_t drm_aligned_stride(int width, int bpp) {
    size_t raw = (size_t)width * bpp;
    const size_t align = 64;
    return (raw + align - 1) & ~(align - 1);
}

DmaBuffer::~DmaBuffer() {
    if (ptr && ptr != MAP_FAILED) {
        if (heap) ::free(ptr);
        else munmap(ptr, size);
    }
    if (fd >= 0) close(fd);
    if (handle != 0 && drm_fd >= 0) {
        struct drm_mode_destroy_dumb d = {};
        d.handle = handle;
        ioctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
    }
}

DmaBuffer::DmaBuffer(DmaBuffer&& o) noexcept
    : fd(o.fd), ptr(o.ptr), size(o.size), width(o.width), height(o.height),
      format(o.format), stride(o.stride), handle(o.handle), drm_fd(o.drm_fd),
      heap(o.heap) {
    o.fd = -1; o.ptr = nullptr; o.size = 0; o.handle = 0; o.drm_fd = -1;
}

DmaBuffer& DmaBuffer::operator=(DmaBuffer&& o) noexcept {
    if (this != &o) {
        if (ptr && ptr != MAP_FAILED) {
            if (heap) ::free(ptr);
            else munmap(ptr, size);
        }
        if (fd >= 0) close(fd);
        if (handle != 0 && drm_fd >= 0) {
            struct drm_mode_destroy_dumb d = {};
            d.handle = handle;
            ioctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
        }
        fd = o.fd; ptr = o.ptr; size = o.size; width = o.width; height = o.height;
        format = o.format; stride = o.stride; handle = o.handle; drm_fd = o.drm_fd;
        heap = o.heap;
        o.fd = -1; o.ptr = nullptr; o.size = 0; o.handle = 0; o.drm_fd = -1;
    }
    return *this;
}

DmaBufferPool::DmaBufferPool(int width, int height, int format, size_t capacity,
                             bool heap)
    : width_(width), height_(height), format_(format), capacity_(capacity),
      heap_(heap) {
    stride_ = drm_aligned_stride(width, rga_format_bpp(format));
}

DmaBufferPool::~DmaBufferPool() {
    std::lock_guard<std::mutex> lk(mtx_);
    for (auto* b : free_list_) delete b;
    free_list_.clear();
}

bool DmaBufferPool::alloc_one_drm(DmaBuffer& out) {
    int drm_fd = get_drm_fd();
    if (drm_fd < 0) return false;

    int bpp = rga_format_bpp(format_);

    struct drm_mode_create_dumb create_req = {};
    create_req.width = width_;
    create_req.height = height_;
    create_req.bpp = bpp * 8;
    if (drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) < 0) {
        fprintf(stderr, "[drm] CREATE_DUMB failed: %s\n", strerror(errno));
        return false;
    }

    struct drm_prime_handle prime_req = {};
    prime_req.handle = create_req.handle;
    prime_req.flags = DRM_CLOEXEC | O_RDWR;
    if (drmIoctl(drm_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime_req) < 0) {
        fprintf(stderr, "[drm] PRIME_HANDLE_TO_FD failed: %s\n", strerror(errno));
        struct drm_mode_destroy_dumb d = {};
        d.handle = create_req.handle;
        drmIoctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
        return false;
    }

    struct drm_mode_map_dumb map_req = {};
    map_req.handle = create_req.handle;
    if (drmIoctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req) < 0) {
        fprintf(stderr, "[drm] MAP_DUMB failed: %s\n", strerror(errno));
        close(prime_req.fd);
        struct drm_mode_destroy_dumb d = {};
        d.handle = create_req.handle;
        drmIoctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
        return false;
    }

    void* p = mmap(nullptr, create_req.size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   drm_fd, map_req.offset);
    if (p == MAP_FAILED) {
        fprintf(stderr, "[drm] mmap failed: %s\n", strerror(errno));
        close(prime_req.fd);
        struct drm_mode_destroy_dumb d = {};
        d.handle = create_req.handle;
        drmIoctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
        return false;
    }

    out.fd = prime_req.fd;
    out.ptr = p;
    out.size = create_req.size;
    out.width = width_;
    out.height = height_;
    out.format = format_;
    out.stride = create_req.pitch;
    out.handle = create_req.handle;
    out.drm_fd = drm_fd;
    out.heap = false;
    // zero once: the letterbox padding relies on untouched zero memory
    memset(p, 0, create_req.size);
    return true;
}

DmaBufferPtr DmaBufferPool::alloc() {
    std::unique_lock<std::mutex> lk(mtx_);
    if (!free_list_.empty()) {
        DmaBuffer* raw = free_list_.back();
        free_list_.pop_back();
        return DmaBufferPtr(raw, [this](DmaBuffer* b) { recycle(b); });
    }
    lk.unlock();

    auto* raw = new DmaBuffer();
    bool ok = heap_ ? alloc_one_heap(*raw) : alloc_one_drm(*raw);
    if (!ok) { delete raw; return nullptr; }
    return DmaBufferPtr(raw, [this](DmaBuffer* b) { recycle(b); });
}

bool DmaBufferPool::alloc_one_heap(DmaBuffer& out) {
    size_t bytes = stride_ * (size_t)height_;
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) return false;
    memset(p, 0, bytes);
    out.fd = -1;
    out.ptr = p;
    out.size = bytes;
    out.width = width_;
    out.height = height_;
    out.format = format_;
    out.stride = stride_;
    out.handle = 0;
    out.drm_fd = -1;
    out.heap = true;
    return true;
}

void DmaBufferPool::recycle(DmaBuffer* buf) {
    if (!buf) return;
    // drop excess buffers instead of growing without bound
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (free_list_.size() < capacity_) {
            free_list_.push_back(buf);
            return;
        }
    }
    delete buf;
}
