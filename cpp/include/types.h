#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <cstdint>
#include <functional>
#include <opencv2/opencv.hpp>

// ============================================================================
// 模型输入输出常量（RT-DETR-R18）
// ============================================================================
constexpr int INPUT_WIDTH    = 640;
constexpr int INPUT_HEIGHT   = 640;
constexpr int INPUT_CHANNELS = 3;
constexpr int NUM_CLASSES    = 10;   //!< VisDrone 10 类
constexpr int NUM_BOXES      = 300;  //!< RT-DETR 固定输出 300 个候选框

// RT-DETR 输入归一化参数（与 Python 训练时一致）
// RT-DETR 通常不做均值减法，仅归一化到 [0,1]
constexpr float RTDETR_MEAN[3] = {0.0f, 0.0f, 0.0f};
constexpr float RTDETR_STD[3]  = {255.0f, 255.0f, 255.0f};

// ============================================================================
// 检测结果
// ============================================================================
/**
 * @brief 单个检测目标的结果。
 */
struct DetectResult
{
	int       class_id;   //!< 类别 ID（0~9）
	float     score;      //!< 置信度 (0~1)
	cv::Rect  box;        //!< 边界框（像素坐标）
};

//! VisDrone 类别名称（与训练标签一致）
const std::vector<std::string> CLASSES =
{
	"Pedestrian", "People", "Bicycle", "Car", "Van",
	"Truck", "Tricycle", "Awning-tricycle", "Bus", "Motor"
};

// ============================================================================
// DmaBuffer：CPU/NPU/RGA 三方共享的物理连续内存
//
// 实现方式（RGA_DMA 核心技术）：
//   通过 /dev/dri/renderD128 的 DRM_IOCTL_MODE_CREATE_DUMB 分配物理连续内存，
//   再用 DRM_IOCTL_PRIME_HANDLE_TO_FD 导出为 PRIME fd。
//   该 fd 可在 RGA（wrapbuffer_fd）、NPU（rknn_create_mem_from_fd）、
//   V4L2（VIDIOC_EXPBUF）三个硬件单元间零拷贝传递。
//
// 关键优势：
//   1. 物理连续：DMA 引擎可直接访问，无需 IOMMU 页表查找
//   2. 共享 fd：跨硬件零拷贝，无需 memcpy
//   3. mmap 映射：CPU 需要时可通过虚拟地址访问（热路径不读写）
// ============================================================================
struct DmaBuffer
{
	int       fd        = -1;     //!< PRIME fd，跨硬件传递的"内存句柄"
	void*     ptr       = nullptr;//!< mmap 后的虚拟地址（CPU 访问用）
	size_t    size      = 0;      //!< 总字节数
	int       width     = 0;      //!< 像素宽度
	int       height    = 0;      //!< 像素高度
	int       format    = 0;      //!< RGA 格式，如 RK_FORMAT_BGR_888
	size_t    stride    = 0;      //!< 行跨度（DRM 对齐后，可能 > width*bpp）
	uint32_t  handle    = 0;      //!< DRM GEM handle（内部使用）
	int       drm_fd    = -1;     //!< 关联的 DRM 设备 fd（用于销毁时 ioctl）

	DmaBuffer() = default;
	~DmaBuffer();
	DmaBuffer(const DmaBuffer&) = delete;
	DmaBuffer& operator=(const DmaBuffer&) = delete;
	DmaBuffer(DmaBuffer&& o) noexcept;
	DmaBuffer& operator=(DmaBuffer&& o) noexcept;
};
using DmaBufferPtr = std::shared_ptr<DmaBuffer>;

// ============================================================================
// FrameBundle：贯穿三段流水线的帧数据包
//
// 设计要点：
//   1. src_buf：源 DMA buffer（V4L2 相机零拷贝路径），预处理完成后提前归还采集队列
//   2. input_buf：RGA 输出 DMA buffer（640x640 RGB），NPU 直接零拷贝读取
//   3. orig_img：源图像副本（入队时 clone），用于后处理画框与视频输出
//   4. 帧包以 shared_ptr 在队列间传递，仅移动所有权，不复制图像数据
// ============================================================================
struct FrameBundle
{
	int             frame_id    = -1;          //!< 帧序号（用于输出排序）

	// 源图像：DMA buffer 优先（RGA_DMA 路径），cv::Mat 回退（兼容路径）
	DmaBufferPtr    src_buf;                    //!< 源 DMA buffer（任意分辨率 BGR）
	cv::Mat         orig_img;                   //!< 源 cv::Mat（仅 src_buf 不可用时使用，或用于画框）

	// RGA 输出 / NPU 输入：640x640 RGB DMA buffer
	DmaBufferPtr    input_buf;                  //!< 预处理后的 DMA 缓冲

	// NPU 输出（预分配，避免每帧 malloc/free）
	std::vector<float>  pred_boxes;    //!< [300, 4] 归一化框坐标
	std::vector<float>  pred_logits;   //!< [300, NUM_CLASSES] 分数
	int                 num_boxes = 0;
	int                 num_classes = NUM_CLASSES;   // 当前模型实际类别数 - 固定

	// 标记：是否使用 DMA 源（决定 RGA 走 DMA→DMA 还是 virt→DMA）
	bool            use_dma_src = false;

	// 性能追踪时间戳（单位：微秒）
	int64_t t_enqueue   = 0;
	int64_t t_pre_done  = 0;
	int64_t t_npu_done  = 0;
	int64_t t_post_done = 0;
};
using FrameBundlePtr = std::shared_ptr<FrameBundle>;

// ============================================================================
// 全局性能计数器（原子操作，线程安全）
// ============================================================================
struct PerfCounter
{
	std::atomic<int64_t> total_frames{0};
	std::atomic<int64_t> total_pre_us{0};
	std::atomic<int64_t> total_npu_us{0};
	std::atomic<int64_t> total_post_us{0};
	std::atomic<int64_t> total_e2e_us{0};
	std::atomic<int64_t> total_dma_to_dma{0};   // DMA→DMA 路径帧数
	std::atomic<int64_t> total_virt_to_dma{0};  // virt→DMA 回退路径帧数
};
extern PerfCounter g_perf;
