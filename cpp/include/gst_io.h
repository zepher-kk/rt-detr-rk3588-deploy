#pragma once

#include <string>
#include <opencv2/opencv.hpp>

// ============================================================================
// gst_io.h - GStreamer + Rockchip MPP 硬件编解码封装
//
// GstVideoReader：filesrc → demux → (h264/h265)parse → mppvideodec → appsink
//                 硬解输出 NV12 后由调用方转 BGR（规避 videoconvert 高 CPU 开销）；
//                 图片走 mppjpegdec 硬解，PNG/BMP/WebP 软解，失败回退 OpenCV
// GstVideoWriter：appsrc(BGR) → mpph264enc → h264parse → mp4mux → filesink
//                 （硬件 H.264 编码，替代 OpenCV mp4v 软编）
// ============================================================================

/**
 * @brief MPP 硬件解码视频读取器（接口对齐 cv::VideoCapture 常用子集）。
 */
class GstVideoReader
{
	public:
		GstVideoReader();
		~GstVideoReader();

		/** @brief 打开视频文件（H.264 MP4），失败返回 false。 */
		bool open(const std::string& path);

		/** @brief 读取下一帧（BGR），EOF/失败返回 false。 */
		bool read(cv::Mat& frame);

		bool   isOpened() const;
		double fps() const;
		int    width() const;
		int    height() const;
		/**
		 * @brief 容器/流是否提供了权威帧率（caps framerate 有效且 > 0）。
		 * @note VFR 或帧率元数据缺失（如 0/1）的源返回 false，需用两遍法实测。
		 */
		bool caps_fps_authoritative() const;
		/**
		 * @brief 实测平均帧率：
		 * 优先容器时长（count/duration），否则 PTS 首尾跨度（(count-1)/span）。
		 * 需先读完整段；无效返回 0。
		 */
		double measured_avg_fps() const;
		void   release();

	private:
		bool try_start_pipeline(const std::string& desc, bool verify_frame);
		struct Impl;
		Impl* impl_;
};

/**
 * @brief MPP 硬件编码视频写入器（H.264 MP4，接口对齐 cv::VideoWriter 常用子集）。
 */
class GstVideoWriter
{
	public:
		GstVideoWriter();
		~GstVideoWriter();

		/** @brief 打开输出文件（H.264 MP4），失败返回 false。 */
		bool open(const std::string& path, double fps, cv::Size size);

		bool          isOpened() const;
		bool          write(const cv::Mat& frame);
		void          release();

		/**
		 * @brief 配置硬件编码器质量参数。
		 * @param rc_mode  "fixqp"（固定 QP 质量优先）/ "vbr" / "cbr"
		 * @param qp_init  rc_mode=fixqp 时的初始 QP（越小越清晰，建议 24~30）
		 * @param profile  "high"（CABAC，推荐）/ "main" / "baseline"
		 */
		void set_encoder_params(const std::string& rc_mode, int qp_init,
		                        const std::string& profile);

	private:
		struct Impl;
		Impl* impl_;
	};
