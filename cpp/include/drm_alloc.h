#pragma once

#include "types.h"

#include <condition_variable>
#include <mutex>
#include <vector>

class DmaBufferPool {
  public:
    DmaBufferPool(int width, int height, int format, size_t capacity = 8,
                  bool heap = false);
    ~DmaBufferPool();

    DmaBufferPtr alloc();
    void recycle(DmaBuffer* buf);

    size_t capacity() const { return capacity_; }
    size_t stride() const { return stride_; }
    bool   is_heap() const { return heap_; }

  private:
    bool alloc_one_drm(DmaBuffer& out);
    bool alloc_one_heap(DmaBuffer& out);

    int      width_, height_, format_;
    size_t   capacity_, stride_;
    bool     heap_ = false;
    std::vector<DmaBuffer*> free_list_;
    mutable std::mutex      mtx_;
};

int get_drm_fd();
