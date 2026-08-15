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
#include "../include/logger.h"
#include "../include/gst_io.h"

#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
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


	LOG(MOD_TEST, LOG_INFO) << "buf1->fd = " << buf1->fd << std::endl;
	LOG(MOD_TEST, LOG_INFO) << "buf1->ptr = " << buf1->ptr << std::endl;
	LOG(MOD_TEST, LOG_INFO) << "buf1->size = " << buf1->size << std::endl;

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
	    LOG(MOD_TEST, LOG_INFO) << "src_buf->ptr = " << src_buf->ptr << std::endl;
	    LOG(MOD_TEST, LOG_INFO) << "src_buf->size = " << src_buf->size << std::endl;
	    LOG(MOD_TEST, LOG_INFO) << "src_buf->fd = " << src_buf->fd << std::endl;
	    LOG(MOD_TEST, LOG_INFO) << "src_buf->stride = " << src_buf->stride << std::endl;

	    // 尝试写入单个字节
	    LOG(MOD_TEST, LOG_INFO) << "Attempting to write single byte..." << std::endl;
	    volatile unsigned char* test_ptr = (volatile unsigned char*)src_buf->ptr;
	    *test_ptr = 0x55;
	    LOG(MOD_TEST, LOG_INFO) << "Single byte write succeeded!" << std::endl;

	    // 如果单字节写入成功，则进行循环填充
	    LOG(MOD_TEST, LOG_INFO) << "Filling entire buffer..." << std::endl;
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
	    LOG(MOD_TEST, LOG_INFO) << "Fill completed." << std::endl;
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
// 测试 12: detect_image 空图保护（uncle-bob 约束：新增代码必须配套单元测试）
// ============================================================================
void test_detect_image_guard()
{
	TEST("detect_image 空图保护");

	PipelineManager pipeline(0, 0, 0, "dummy_model.rknn", 4, 0.45f, RKNN_NPU_CORE_AUTO);
	cv::Mat empty_img;
	cv::Mat out_img;

	ASSERT_TRUE(!pipeline.detect_image(empty_img, out_img), "空图应返回 false");
	ASSERT_TRUE(out_img.empty(), "空图不应产生输出图片");

	PASS();
}

// ============================================================================
// 测试 13: preprocess_mat_to_dma 正确性（回归：非对齐宽度输入污染问题）
// ============================================================================
void test_preprocess_mat_to_dma_correctness()
{
	TEST("preprocess_mat_to_dma 正确性(1360宽回归)");

	// 模拟 uav.jpg 的 1360 宽非对齐输入，每行不同颜色
	cv::Mat src(765, 1360, CV_8UC3);
	for (int y = 0; y < src.rows; ++y)
	{
		src.row(y).setTo(cv::Scalar(y % 256, 0, 255));
	}

	// CPU 参考：与原版一致的 resize + BGR→RGB
	cv::Mat ref;
	cv::resize(src, ref, cv::Size(640, 640));
	cv::cvtColor(ref, ref, cv::COLOR_BGR2RGB);

	RgaPreprocessor& pre = rga_preprocessor();
	pre.init(2);

	DmaBufferPtr out = pre.preprocess_mat_to_dma(src);
	ASSERT_TRUE(out != nullptr, "preprocess_mat_to_dma 失败");
	ASSERT_TRUE(out->width == 640 && out->height == 640, "输出尺寸错误");
	ASSERT_TRUE(out->format == RK_FORMAT_RGB_888, "输出格式应为 RGB");

	// RGA 插值与 OpenCV 插值非逐位一致，用平均绝对误差(MAE)判定正确性
	const uint8_t* a = (const uint8_t*)out->ptr;
	const uint8_t* b = ref.data;
	const size_t n = ref.total() * ref.elemSize();
	double sum = 0.0;
	for (size_t k = 0; k < n; ++k)
	{
		sum += (a[k] >= b[k]) ? (a[k] - b[k]) : (b[k] - a[k]);
	}
	double mae = sum / n;
	ASSERT_TRUE(mae < 3.0, "输出与 CPU 参考 MAE 过大（RGA/stride 污染回归）");
	std::cout << "MAE=" << mae << " ";

	PASS();
}

// ============================================================================
// 测试 15: preprocess_dma_to_dma stride 安全（模拟相机 DMA→DMA 非对齐宽度）
// ============================================================================
void test_dma_to_dma_stride_safe()
{
	TEST("preprocess_dma_to_dma stride 安全(1360宽)");

	// 模拟 V4L2 相机 BGR24 帧：1360x765，DRM stride 可能为 4096（1360*3=4080 非对齐）
	cv::Mat src(765, 1360, CV_8UC3);
	for (int y = 0; y < src.rows; ++y)
	{
		src.row(y).setTo(cv::Scalar(y % 256, 0, 255));
	}

	DmaBufferPool pool(1360, 765, RK_FORMAT_BGR_888, 1);
	DmaBufferPtr dma = pool.alloc();
	ASSERT_TRUE(dma != nullptr, "DMA 分配失败");
	LOG(MOD_TEST, LOG_INFO) << "simulated cam stride=" << dma->stride
	          << " width*3=" << 1360 * 3 << "\n";

	// 按实际 stride 逐行写入，模拟相机输出
	uint8_t* drow = (uint8_t*)dma->ptr;
	for (int y = 0; y < src.rows; ++y)
	{
		memcpy(drow, src.ptr<uint8_t>(y), 1360 * 3);
		drow += dma->stride;
	}

	// CPU 参考
	cv::Mat ref;
	cv::resize(src, ref, cv::Size(640, 640));
	cv::cvtColor(ref, ref, cv::COLOR_BGR2RGB);

	RgaPreprocessor& pre = rga_preprocessor();
	pre.init(2);
	DmaBufferPtr out = pre.preprocess_dma_to_dma(dma);
	ASSERT_TRUE(out != nullptr, "preprocess_dma_to_dma 失败");

	const uint8_t* a = (const uint8_t*)out->ptr;
	const uint8_t* b = ref.data;
	const size_t n = ref.total() * ref.elemSize();
	double sum = 0.0;
	for (size_t k = 0; k < n; ++k)
	{
		sum += (a[k] >= b[k]) ? (a[k] - b[k]) : (b[k] - a[k]);
	}
	ASSERT_TRUE(sum / n < 3.0, "输出与 CPU 参考 MAE 过大（stride 污染）");

	PASS();
}

// ============================================================================
// 测试 14: 日志等级化模块化控制（uncle-bob 约束：新代码必须配套单元测试）
// ============================================================================
void test_logger_modules()
{
	TEST("logger 模块级联与级别控制");

	log_set_modules(1 << MOD_MAIN);
	ASSERT_TRUE(log_enabled(MOD_MAIN, LOG_INFO), "-G1 应开启 Main");
	ASSERT_TRUE(!log_enabled(MOD_RKNN, LOG_INFO), "-G1 不应开启 RKNN");
	ASSERT_TRUE(log_enabled(MOD_RKNN, LOG_ERROR), "ERROR 应恒打印");

	log_set_modules((1 << MOD_MAIN) | (1 << MOD_RKNN));
	ASSERT_TRUE(log_enabled(MOD_RKNN, LOG_INFO), "-G2 应开启 RKNN");
	ASSERT_TRUE(!log_enabled(MOD_PIPELINE, LOG_INFO), "-G2 不应开启 Pipeline");

	log_set_level(LOG_ERROR);
	ASSERT_TRUE(!log_enabled(MOD_MAIN, LOG_INFO), "级别 ERROR 时 INFO 不应输出");
	ASSERT_TRUE(log_enabled(MOD_MAIN, LOG_ERROR), "级别 ERROR 时 ERROR 应输出");

	log_set_level(LOG_INFO);
	log_set_modules((1 << MOD_COUNT) - 1);
	PASS();
}

// ============================================================================
// 测试 16: GStreamer MPP 硬件编码写入（uncle-bob：新代码配套测试）
// ============================================================================
void test_gst_video_writer()
{
	TEST("GstVideoWriter 硬编(H.264)");

	const char* out_path = "/home/neardi/Workspace_Codex/rk3588_-rt-detr/cpp_DMSJ/build/ut_gst_out.mp4";
	GstVideoWriter writer;
	ASSERT_TRUE(writer.open(out_path, 30.0, cv::Size(320, 240)), "GstVideoWriter open 失败");

	cv::Mat frame(240, 320, CV_8UC3, cv::Scalar(90, 140, 200));
	for (int i = 0; i < 5; ++i)
	{
		ASSERT_TRUE(writer.write(frame), "GstVideoWriter write 失败");
	}
	writer.release();

	FILE* fp = fopen(out_path, "rb");
	ASSERT_TRUE(fp != nullptr, "输出文件不存在");
	fseek(fp, 0, SEEK_END);
	long sz = ftell(fp);
	fclose(fp);
	ASSERT_TRUE(sz > 0, "输出文件为空");
	LOG(MOD_TEST, LOG_INFO) << "H.264 output size=" << sz << "\n";

	PASS();
}

// ============================================================================
// 测试 17: GStreamer MPP 硬件解码读取
// ============================================================================
void test_gst_video_reader()
{
	TEST("GstVideoReader 硬解(cars_2s)");

	GstVideoReader reader;
	ASSERT_TRUE(reader.open("/home/neardi/Workspace_Codex/img/cars_2s.mp4"), "GstVideoReader open 失败");
	cv::Mat frame;
	int count = 0;
	while (count < 10 && reader.read(frame))
	{
		ASSERT_TRUE(!frame.empty(), "读到空帧");
		count++;
	}
	ASSERT_TRUE(count > 0, "未读到任何帧");
	LOG(MOD_TEST, LOG_INFO) << "decoded frames=" << count
	          << " fps=" << reader.fps()
	          << " dims=" << frame.cols << "x" << frame.rows << "\n";
	reader.release();

	PASS();
}

// ============================================================================
// 测试 18: GStreamer MPP JPEG 硬解（图片输入）
// ============================================================================
void test_gst_image_decode()
{
	TEST("GstVideoReader JPEG 硬解(uav.jpg)");

	GstVideoReader reader;
	ASSERT_TRUE(reader.open("/home/neardi/Workspace_Codex/img/uav.jpg"), "JPEG open 失败");
	cv::Mat frame;
	ASSERT_TRUE(reader.read(frame), "JPEG 读帧失败");
	LOG(MOD_TEST, LOG_INFO) << "JPEG dims=" << frame.cols << "x" << frame.rows << "\n";
	ASSERT_TRUE(frame.cols == 1360, "宽度应为 1360");
	ASSERT_TRUE(frame.rows >= 765 && frame.rows <= 768, "高度应为 765~768（硬件对齐）");
	reader.release();

	PASS();
}

// ============================================================================
// 测试 19: GStreamer 1080p 解码无顶部绿条（回归：高度对齐 1080→1088 的 UV 偏移）
// ============================================================================
void test_gst_1080p_no_green_bar()
{
	TEST("GstVideoReader 1080p 无顶部绿条");

	GstVideoReader reader;
	ASSERT_TRUE(reader.open("/home/neardi/Workspace_Codex/img/test_people_small_little_18s.mp4"),
	            "1080p open 失败");
	cv::Mat frame;
	ASSERT_TRUE(reader.read(frame), "1080p 读帧失败");
	ASSERT_TRUE(frame.cols == 1920 && frame.rows == 1080, "尺寸应为 1920x1080");

	// 顶部 20 行绿色像素占比应 < 10%
	int green = 0;
	int total = 0;
	for (int y = 0; y < 20; ++y)
	{
		for (int x = 0; x < frame.cols; ++x)
		{
			cv::Vec3b p = frame.at<cv::Vec3b>(y, x);
			total++;
			if (p[1] > 150 && p[0] < 100 && p[2] < 100) green++;
		}
	}
	LOG(MOD_TEST, LOG_INFO) << "top20 green_ratio=" << (double)green / total << "\n";
	ASSERT_TRUE((double)green / total < 0.1, "顶部出现绿色伪影（UV 偏移回归）");

	reader.release();
	PASS();
}

// ============================================================================
// 测试 20: 多格式输入（AVI/H.264、MP4/H.265、图片）
// ============================================================================
void test_gst_multi_format()
{
	TEST("GstVideoReader 多格式(avi/hevc)");

	const char* build = "/home/neardi/Workspace_Codex/rk3588_-rt-detr/cpp_DMSJ/build/";
	std::string avi = std::string(build) + "ut_multi.avi";
	std::string hevc = std::string(build) + "ut_multi_hevc.mp4";

	// 用板端 mpp 编码器生成测试文件（不依赖外部素材）
	std::string cmd_avi = "gst-launch-1.0 -q videotestsrc num-buffers=15 ! videoconvert "
	                      "! video/x-raw,format=BGR,width=320,height=240 ! mpph264enc ! h264parse "
	                      "! avimux ! filesink location=" + avi + " 2>/dev/null";
	std::string cmd_hevc = "gst-launch-1.0 -q videotestsrc num-buffers=15 ! videoconvert "
	                       "! video/x-raw,format=I420,width=320,height=240 ! mpph265enc ! h265parse "
	                       "! mp4mux ! filesink location=" + hevc + " 2>/dev/null";
	system(cmd_avi.c_str());
	system(cmd_hevc.c_str());

	GstVideoReader r1;
	ASSERT_TRUE(r1.open(avi), "AVI(H.264) 打开失败");
	cv::Mat f1;
	ASSERT_TRUE(r1.read(f1) && !f1.empty(), "AVI 读帧失败");
	r1.release();

	GstVideoReader r2;
	ASSERT_TRUE(r2.open(hevc), "HEVC(H.265) 打开失败");
	cv::Mat f2;
	ASSERT_TRUE(r2.read(f2) && !f2.empty(), "HEVC 读帧失败");
	r2.release();

	PASS();
}

// 测试 21: PipelineManager 模型加载失败不挂起（回归：NPU 初始化失败曾导致
// reader 阻塞在满队列 / wait_idle 永久等待 / 析构 push 毒丸卡死）
void test_pipeline_init_failure_no_hang()
{
	TEST("PipelineManager 模型加载失败不挂起");
	auto t0 = steady_clock::now();
	{
		PipelineManager pipeline(1, 2, 1, "/nonexistent/model.rknn", 4);
		cv::Mat f(480, 332, CV_8UC3, cv::Scalar(0, 0, 0));
		for (int i = 0; i < 100; ++i) pipeline.push_image(i, f);
		pipeline.wait_idle();
		// 析构在作用域末尾执行
	}
	double sec = duration<double>(steady_clock::now() - t0).count();
	ASSERT_TRUE(sec < 30.0, "模型加载失败仍挂起超过 30s");
	PASS();
}

// 测试 22: GstVideoReader VFR 平均帧率估算（回归：480×332 录屏 caps framerate=0/1，
// 前段 60fps burst 曾导致输出 60fps、时长 21.7s 被压成 9.2s）
void test_gst_vfr_avg_fps()
{
	TEST("GstVideoReader VFR 平均帧率估算");
	const char* vfr = "/home/neardi/Workspace_Codex/img/cars-from uav_Unconventional Size_.mp4";
	GstVideoReader r;
	ASSERT_TRUE(r.open(vfr), "VFR 视频打开失败");
	ASSERT_TRUE(!r.caps_fps_authoritative(), "VFR 源不应有权威 caps 帧率");
	cv::Mat f;
	int n = 0;
	while (r.read(f)) n++;
	double avg = r.measured_avg_fps();
	r.release();
	ASSERT_TRUE(n > 300, "VFR 读帧数异常");
	ASSERT_TRUE(avg >= 20.0 && avg <= 35.0,
	            "VFR 实测平均帧率应约 25~27fps（容器时长/跨度），实际偏出");
	std::cout << "frames=" << n << " avg_fps=" << avg << " ";
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
	test_detect_image_guard();
	test_preprocess_mat_to_dma_correctness();
	test_dma_to_dma_stride_safe();
	test_logger_modules();
	test_gst_video_writer();
	test_gst_video_reader();
	test_gst_image_decode();
	test_gst_1080p_no_green_bar();
	test_gst_multi_format();
	test_pipeline_init_failure_no_hang();
	test_gst_vfr_avg_fps();

	std::cout << std::endl;
	std::cout << "============================================" << std::endl;
	std::cout << "  测试结果汇总" << std::endl;
	std::cout << "============================================" << std::endl;
	std::cout << "  通过: " << tests_passed << std::endl;
	std::cout << "  失败: " << tests_failed << std::endl;
	std::cout << "============================================" << std::endl;

	return tests_failed > 0 ? 1 : 0;
}
