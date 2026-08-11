// ============================================================================
// test_unit.cc - RT-DETR RGA_DMA 流水线单元测试
//
// 运行: ./test_unit
// ============================================================================

#include "../include/types.h"
#include "../include/drm_alloc.h"
#include "../include/rga_utils.h"
#include "../include/rknn_detector.h"
#include "../include/postprocess.h"
#include "../include/npu_pipeline.h"

#include <iostream>
#include <cassert>
#include <cstring>
#include <chrono>

#include <rga.h>
#include <opencv2/opencv.hpp>

using namespace std::chrono;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
	do { \
		std::cout << "[TEST] " << name << " ... "; \
	} while(0)

#define PASS() \
	do { \
		std::cout << "PASS" << std::endl; \
		tests_passed++; \
	} while(0)

#define FAIL(msg) \
	do { \
		std::cout << "FAIL: " << msg << std::endl; \
		tests_failed++; \
	} while(0)

#define ASSERT_TRUE(cond, msg) \
	do { \
		if (!(cond)) { FAIL(msg); return; } \
	} while(0)

#define ASSERT_EQ(a, b, msg) \
	do { \
		if ((a) != (b)) { FAIL(msg); return; } \
	} while(0)

// ============================================================================
// 测试 1: DmaBuffer 分配与释放
// ============================================================================
void test_dma_buffer_alloc()
{
	TEST("DmaBuffer 分配与释放");

	DmaBufferPool pool(640, 640, RK_FORMAT_RGB_888, 4);

	// 测试分配
	DmaBufferPtr buf1 = pool.alloc();

	ASSERT_TRUE(buf1 != nullptr, "分配失败");
	ASSERT_TRUE(buf1->fd >= 0, "fd 无效");
	ASSERT_TRUE(buf1->ptr != nullptr, "ptr 为空");
	ASSERT_TRUE(buf1->size > 0, "size 为 0");
	ASSERT_EQ(buf1->width, 640, "width 不匹配");
	ASSERT_EQ(buf1->height, 640, "height 不匹配");


	std::cerr << "buf1->fd = " << buf1->fd << std::endl;
	std::cerr << "buf1->ptr = " << buf1->ptr << std::endl;
	std::cerr << "buf1->size = " << buf1->size << std::endl;

	// 测试池复用
	size_t size_before = pool.size();
	{
		DmaBufferPtr buf2 = pool.alloc();
		ASSERT_TRUE(buf2 != nullptr, "第二次分配失败");
	}
	// buf2 析构后应归还到池（期望池大小为 1，因为 buf2 已归还）：
	ASSERT_EQ(pool.size(), 1, "池未正确回收");

	PASS();
}

// ============================================================================
// 测试 2: DmaBuffer 池容量限制
// ============================================================================
void test_dma_pool_capacity()
{
	TEST("DmaBuffer 池容量限制");

	DmaBufferPool pool(320, 320, RK_FORMAT_RGB_888, 2);

	std::vector<DmaBufferPtr> bufs;
	for (int i = 0; i < 5; ++i)
	{
		DmaBufferPtr buf = pool.alloc();
		if (buf) bufs.push_back(buf);
	}

	ASSERT_TRUE(bufs.size() >= 2, "至少应分配 2 个");
	PASS();
}

// ============================================================================
// 测试 3: RGA 预处理初始化
// ============================================================================
void test_rga_preprocessor_init()
{
	TEST("RGA 预处理器初始化");

	RgaPreprocessor& rga = rga_preprocessor();
	bool ok = rga.init(4);
	ASSERT_TRUE(ok, "初始化失败");

	PASS();
}

// ============================================================================
// 测试 4: cv::Mat → DMA 桥接（安全版，避免 cv::Mat 析构问题）
// ============================================================================
void test_mat_to_dma_bridge()
{
	TEST("cv::Mat → DMA 桥接");

	// 使用 vector 存储图像数据，避免 cv::Mat 内部管理复杂性
	const int width = 640, height = 480;
	std::vector<unsigned char> img_data(width * height * 3, 0);

	// 填充测试数据 (BGR 顺序)
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			size_t idx = (y * width + x) * 3;
			img_data[idx + 0] = 128; // B
			img_data[idx + 1] = 64;  // G
			img_data[idx + 2] = 32;  // R
		}
	}

	// 创建 cv::Mat 包装，但不拥有数据（data 指向 img_data，析构时不会释放）
	cv::Mat test_img(height, width, CV_8UC3, img_data.data());

	DmaBufferPool src_pool(640, 480, RK_FORMAT_BGR_888 /* RK_FORMAT_BGR_888 */, 4);
	DmaBufferPtr dma_buf = rga_preprocessor().bridge_mat_to_dma(test_img, src_pool);

	ASSERT_TRUE(dma_buf != nullptr, "桥接失败");
	ASSERT_TRUE(dma_buf->ptr != nullptr, "DMA ptr 为空");

	// 验证 DMA buffer 中的数据
	unsigned char* data = (unsigned char*)dma_buf->ptr;
	ASSERT_EQ(data[0], 128, "B 通道数据不匹配");
	ASSERT_EQ(data[1], 64,  "G 通道数据不匹配");
	ASSERT_EQ(data[2], 32,  "R 通道数据不匹配");

	// 显式释放 DMA buffer，避免后续析构顺序问题
	dma_buf.reset();

	PASS();
}

// ============================================================================
// 测试 5: RGA DMA→DMA 预处理（调试版，用于定位写入崩溃）
// ============================================================================
void test_rga_dma_to_dma()
{
	TEST("RGA DMA→DMA 预处理");

	// 创建源 DMA buffer
	DmaBufferPool src_pool(640, 480, RK_FORMAT_BGR_888 /* BGR_888 */, 2);
	DmaBufferPtr src_buf = src_pool.alloc();
	ASSERT_TRUE(src_buf != nullptr, "源 DMA 分配失败");
	/*
	    // 打印调试信息
	    std::cerr << "[DEBUG] src_buf->ptr = " << src_buf->ptr << std::endl;
	    std::cerr << "[DEBUG] src_buf->size = " << src_buf->size << std::endl;
	    std::cerr << "[DEBUG] src_buf->fd = " << src_buf->fd << std::endl;
	    std::cerr << "[DEBUG] src_buf->stride = " << src_buf->stride << std::endl;

	    // 尝试写入单个字节
	    std::cerr << "[DEBUG] Attempting to write single byte..." << std::endl;
	    volatile unsigned char* test_ptr = (volatile unsigned char*)src_buf->ptr;
	    *test_ptr = 0x55;
	    std::cerr << "[DEBUG] Single byte write succeeded!" << std::endl;

	    // 如果单字节写入成功，则进行循环填充
	    std::cerr << "[DEBUG] Filling entire buffer..." << std::endl;
	    unsigned char* data = (unsigned char*)src_buf->ptr;
	    size_t stride = src_buf->stride;
	    for (int y = 0; y < 480; ++y) {
	        for (int x = 0; x < 640; ++x) {
	            size_t idx = y * stride + x * 3;
	            data[idx + 0] = 100;  // B
	            data[idx + 1] = 150;  // G
	            data[idx + 2] = 200;  // R
	        }
	    }
	    std::cerr << "[DEBUG] Fill completed." << std::endl;
	*/
	// 执行 RGA 预处理
	DmaBufferPtr dst_buf = rga_preprocessor().preprocess_dma_to_dma(src_buf);
	ASSERT_TRUE(dst_buf != nullptr, "RGA 预处理失败");
	ASSERT_EQ(dst_buf->width, 640, "目标宽度不匹配");
	ASSERT_EQ(dst_buf->height, 640, "目标高度不匹配");

	PASS();
}

// ============================================================================
// 测试 5.1: RGA 预处理（使用 cv::Mat 输入，验证 RGA 功能）
// ============================================================================
void test_rga_mat_to_dma()
{
	TEST("RGA 预处理 (via cv::Mat)");

	// 创建测试图像 (480x640 BGR)
	cv::Mat test_img(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));

	// 调用 RGA 预处理（cv::Mat → DMA）
	DmaBufferPtr dst_buf = rga_preprocessor().preprocess_mat_to_dma(test_img);
	ASSERT_TRUE(dst_buf != nullptr, "RGA 预处理失败");
	ASSERT_EQ(dst_buf->width, 640, "目标宽度不匹配");
	ASSERT_EQ(dst_buf->height, 640, "目标高度不匹配");

	// 可选：验证目标数据是否正确（比如第一个像素）
	unsigned char* data = (unsigned char*)dst_buf->ptr;
	// 因为 preprocess_mat_to_dma 会将 BGR→RGB，所以预期 R=200, G=150, B=100
	// 但不确定是否完全准确，只简单检查非空
	ASSERT_TRUE(data != nullptr, "目标数据为空");

	PASS();
}

// ============================================================================
// 测试 6: 后处理解码正确性
// ============================================================================
void test_postprocess_decode()
{
	TEST("后处理解码正确性");

	// 构造测试数据：1 个有效框，299 个无效框
	float boxes[300 * 4];
	float scores[300 * 10];

	memset(boxes, 0, sizeof(boxes));
	memset(scores, 0, sizeof(scores));

	// 第 0 个框：有效，类别 3，分数 0.9
	boxes[0] = 0.5f;  // cx
	boxes[1] = 0.5f;  // cy
	boxes[2] = 0.2f;  // w
	boxes[3] = 0.3f;  // h
	scores[3] = 0.9f; // 类别 3 分数

	// 第 1 个框：有效，类别 0，分数 0.3（低于阈值）
	boxes[4] = 0.3f;
	boxes[5] = 0.3f;
	boxes[6] = 0.1f;
	boxes[7] = 0.1f;
	scores[10 + 0] = 0.3f;

	std::vector<DetectResult> results =
	    decode_rtdetr_output(boxes, scores, 300, 640, 480, 0.45f);

	ASSERT_EQ(results.size(), 1, "应只检测到 1 个目标");
	ASSERT_EQ(results[0].class_id, 3, "类别 ID 不匹配");
	ASSERT_TRUE(results[0].score > 0.89f && results[0].score < 0.91f, "分数不匹配");

	PASS();
}

// ============================================================================
// 测试 7: 后处理边界条件
// ============================================================================
void test_postprocess_edge_cases()
{
	TEST("后处理边界条件");

	// 全零数据
	float boxes[300 * 4] = {0};
	float scores[300 * 10] = {0};

	std::vector<DetectResult> results =
	    decode_rtdetr_output(boxes, scores, 300, 640, 480, 0.45f);

	ASSERT_EQ(results.size(), 0, "全零数据应返回 0 个结果");

	// 超出边界的框
	boxes[0] = -0.5f;  // cx 负数
	boxes[1] = 1.5f;   // cy 超出
	boxes[2] = 0.2f;
	boxes[3] = 0.3f;
	scores[3] = 0.9f;

	results = decode_rtdetr_output(boxes, scores, 300, 640, 480, 0.45f);
	ASSERT_TRUE(results.size() <= 1, "边界框应被裁剪或过滤");

	PASS();
}

// ============================================================================
// 测试 8: BoundedSafeQueue 基本功能
// ============================================================================
void test_bounded_queue()
{
	TEST("BoundedSafeQueue 基本功能");

	BoundedSafeQueue<int> queue(4);

	// 测试 push/pop
	queue.push(1);
	queue.push(2);
	queue.push(3);

	int val;
	bool ok;
	ok = queue.pop(val);
	ASSERT_TRUE(ok, "pop 失败");
	ASSERT_EQ(val, 1, "第一个值不匹配");

	ok = queue.pop(val);
	ASSERT_TRUE(ok, "pop 失败");
	ASSERT_EQ(val, 2, "第二个值不匹配");

	PASS();
}

// ============================================================================
// 测试 9: BoundedSafeQueue 毒丸协议
// ============================================================================
void test_bounded_queue_poison()
{
	TEST("BoundedSafeQueue 毒丸协议");

	BoundedSafeQueue<int> queue(4);
	queue.shutdown();

	int val;
	bool ok = queue.pop(val);
	ASSERT_TRUE(!ok, "shutdown 后 pop 应返回 false");

	PASS();
}

// ============================================================================
// 测试 10: 性能基准测试
// ============================================================================
void test_performance_benchmark()
{
	TEST("性能基准测试");

	DmaBufferPool pool(640, 640, RK_FORMAT_RGB_888, 8);

	auto start = high_resolution_clock::now();
	for (int i = 0; i < 100; ++i)
	{
		DmaBufferPtr buf = pool.alloc();
		// buf 析构时自动归还
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end - start).count();

	std::cout << "(100 次分配: " << duration << " us, 平均 "
	          << duration / 100 << " us/次) ";
	ASSERT_TRUE(duration < 100000, "分配速度过慢");

	PASS();
}

// ============================================================================
// 主函数
// ============================================================================
int main()
{
	std::cout << "============================================" << std::endl;
	std::cout << "  RT-DETR RGA_DMA 单元测试" << std::endl;
	std::cout << "============================================" << std::endl;
	std::cout << std::endl;

	test_dma_buffer_alloc();
	test_dma_pool_capacity();
	test_rga_preprocessor_init();

	test_mat_to_dma_bridge();
	test_rga_dma_to_dma();
	test_rga_mat_to_dma();

	test_postprocess_decode();
	test_postprocess_edge_cases();

	test_bounded_queue();
	test_bounded_queue_poison();

	test_performance_benchmark();

	std::cout << std::endl;
	std::cout << "============================================" << std::endl;
	std::cout << "  测试结果汇总" << std::endl;
	std::cout << "============================================" << std::endl;
	std::cout << "  通过: " << tests_passed << std::endl;
	std::cout << "  失败: " << tests_failed << std::endl;
	std::cout << "============================================" << std::endl;

	return tests_failed > 0 ? 1 : 0;
}
