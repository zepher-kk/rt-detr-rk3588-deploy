#include "gst_io.h"
#include "logger.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>

#include <cstdio>
#include <cstring>
#include <cctype>
#include <vector>
#include <mutex>

// ============================================================================
// GStreamer 一次性初始化
// ============================================================================
static std::once_flag g_gst_init_flag;

static void ensure_gst_init()
{
	std::call_once(g_gst_init_flag, []()
	{
		gst_init(nullptr, nullptr);
	});
}

// ============================================================================
// GstVideoReader：MPP 硬件解码
// ============================================================================
struct GstVideoReader::Impl
{
	GstElement* pipeline    = nullptr;
	GstElement* sink        = nullptr;
	double      fps         = 30.0;
	int         width       = 0;
	int         height      = 0;
	bool        opened      = false;
	guint64     prev_pts    = 0;
	bool        have_prev_pts = false;
	// 实测平均帧率统计（两遍法用）
	guint64     first_pts   = GST_CLOCK_TIME_NONE;
	guint64     last_pts    = GST_CLOCK_TIME_NONE;
	int         frame_count = 0;
	bool        caps_fps_valid = false;
	// PTS 间隔众数直方图（VFR 视频用众数估计帧率，抗单帧抖动）
	int         pts_hist[241] = {0};
};

GstVideoReader::GstVideoReader() : impl_(new Impl()) {}
GstVideoReader::~GstVideoReader() { release(); delete impl_; }

bool GstVideoReader::open(const std::string& path)
{
	ensure_gst_init();
	release();

	std::string ext = path.substr(path.find_last_of('.') + 1);
	for (auto& c : ext) c = (char)tolower((unsigned char)c);

	// 按扩展名选择候选链路（容器 + 编解码；H.264 失败自动重试 H.265）
	std::vector<std::string> candidates;
	const std::string src = "filesrc location=\"" + path + "\" ! ";
	if (ext == "jpg" || ext == "jpeg")
	{
		// 图片为单帧：不探测（探测会消费掉唯一一帧）
		if (try_start_pipeline(src + "jpegparse ! mppjpegdec ! appsink name=sink", false))
		{
			impl_->opened = true;
			return true;
		}
	}
	else if (ext == "png")
	{
		if (try_start_pipeline(src + "pngdec ! appsink name=sink", false))
		{
			impl_->opened = true;
			return true;
		}
	}
	else if (ext == "bmp")
	{
		if (try_start_pipeline(src + "bmpdec ! appsink name=sink", false))
		{
			impl_->opened = true;
			return true;
		}
	}
	else if (ext == "webp")
	{
		if (try_start_pipeline(src + "webpdec ! appsink name=sink", false))
		{
			impl_->opened = true;
			return true;
		}
	}
	else if (ext == "hevc" || ext == "h265" || ext == "265")
	{
		candidates.push_back(src + "h265parse ! mppvideodec arm-afbc=false ! appsink name=sink");
		candidates.push_back(src + "qtdemux ! h265parse ! mppvideodec arm-afbc=false ! appsink name=sink");
	}
	else if (ext == "h264" || ext == "264")
	{
		candidates.push_back(src + "h264parse ! mppvideodec arm-afbc=false ! appsink name=sink");
	}
	else
	{
		std::string demuxer = "qtdemux";
		if (ext == "avi") demuxer = "avidemux";
		else if (ext == "mkv" || ext == "webm") demuxer = "matroskademux";
		else if (ext == "ts" || ext == "m2ts") demuxer = "tsdemux";
		std::string head = src + demuxer + " ! ";
		candidates.push_back(head + "h264parse ! mppvideodec arm-afbc=false ! appsink name=sink");
		candidates.push_back(head + "h265parse ! mppvideodec arm-afbc=false ! appsink name=sink");
	}

	for (const auto& desc : candidates)
	{
		if (try_start_pipeline(desc, true))
		{
			impl_->opened = true;
			return true;
		}
	}
	LOG(MOD_PIPELINE, LOG_ERROR) << "GstVideoReader: all pipelines failed for " << path << "\n";
	return false;
}

bool GstVideoReader::try_start_pipeline(const std::string& desc, bool verify_frame)
{
	GError* err = nullptr;
	GstElement* pipe = gst_parse_launch(desc.c_str(), &err);
	if (!pipe)
	{
		if (err) g_error_free(err);
		return false;
	}
	GstElement* sink = gst_bin_get_by_name(GST_BIN(pipe), "sink");
	if (!sink)
	{
		gst_object_unref(pipe);
		return false;
	}
	g_object_set(G_OBJECT(sink), "sync", FALSE, "drop", FALSE, "max-buffers", 4, NULL);
	if (gst_element_set_state(pipe, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
	{
		gst_object_unref(sink);
		gst_object_unref(pipe);
		return false;
	}
	// 数据流验证（仅视频）：候选链路必须实际产出帧才算匹配
	// （否则 HEVC 流走 h264parse 候选时 PLAYING 也会"成功"但永无帧输出）
	if (verify_frame)
	{
		GstSample* probe = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 800 * GST_MSECOND);
		if (!probe)
		{
			gst_element_set_state(pipe, GST_STATE_NULL);
			gst_object_unref(sink);
			gst_object_unref(pipe);
			return false;
		}
		// open 阶段即从出帧 sample 的 caps 解析帧率：
		// 此前 caps_fps_valid 仅在 read() 里置位，而 main 的两遍法判断发生在 open 后、
		// 首次 read 前 → 所有视频（含 CFR）都被误判为 VFR 走全量两遍解码
		// （长视频如 300s/6000 帧需先等 probe 解码完才开 VideoWriter）。
		GstCaps* pcaps = gst_sample_get_caps(probe);
		if (pcaps)
		{
			GstStructure* s = gst_caps_get_structure(pcaps, 0);
			int fr_n = 0, fr_d = 0;
			if (gst_structure_get_fraction(s, "framerate", &fr_n, &fr_d) &&
			    fr_n > 0 && fr_d > 0)
			{
				impl_->fps = (double)fr_n / fr_d;
				impl_->caps_fps_valid = true;
			}
		}
		gst_sample_unref(probe);
	}
	impl_->pipeline = pipe;
	impl_->sink = sink;
	return true;
}

bool GstVideoReader::read(cv::Mat& frame)
{
	if (!impl_->opened || !impl_->sink) return false;

	GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(impl_->sink), 2 * GST_SECOND);
	if (!sample) return false;

	GstBuffer* buf = gst_sample_get_buffer(sample);
	GstCaps* caps = gst_sample_get_caps(sample);
	GstMapInfo map;
	if (!gst_buffer_map(buf, &map, GST_MAP_READ))
	{
		gst_sample_unref(sample);
		return false;
	}

	int w = 0, h = 0;
	const char* fmt = nullptr;
	if (caps)
	{
		GstStructure* s = gst_caps_get_structure(caps, 0);
		static bool caps_logged = false;
		if (!caps_logged)
		{
			gchar* caps_str = gst_caps_to_string(caps);
			LOG(MOD_PIPELINE, LOG_INFO) << "appsink caps=" << (caps_str ? caps_str : "?") << "\n";
			if (caps_str) g_free(caps_str);
			caps_logged = true;
		}
		gst_structure_get_int(s, "width", &w);
		gst_structure_get_int(s, "height", &h);
		fmt = gst_structure_get_string(s, "format");
		int fr_n = 0, fr_d = 0;
		if (gst_structure_get_fraction(s, "framerate", &fr_n, &fr_d) && fr_n > 0 && fr_d > 0)
		{
			impl_->fps = (double)fr_n / fr_d;
			impl_->caps_fps_valid = true;
		}
	}
	// 帧率估算：容器元数据缺失（如 0/1）时用 PTS 间隔众数推算
	// （VFR 视频 PTS 不规则，取众数比取"最后一帧间隔"更稳）
	guint64 pts = GST_BUFFER_PTS(buf);
	if (!GST_CLOCK_TIME_IS_VALID(pts)) pts = GST_BUFFER_DTS(buf);
	if (GST_CLOCK_TIME_IS_VALID(pts))
	{
		if (impl_->have_prev_pts && pts > impl_->prev_pts)
		{
			double dt = (double)(pts - impl_->prev_pts) / GST_SECOND;
			if (dt > 0.001 && dt < 1.0)
			{
				double est = 1.0 / dt;
				int e = (int)(est + 0.5);
				if (e >= 1 && e <= 240) impl_->pts_hist[e]++;
				// 众数（全区间扫描 240 次，每帧开销可忽略）
				int best = 30, bc = 0;
				for (int i = 1; i <= 240; ++i)
				{
					if (impl_->pts_hist[i] > bc)
					{
						bc = impl_->pts_hist[i];
						best = i;
					}
				}
				impl_->fps = best;
			}
		}
		impl_->prev_pts = pts;
		impl_->have_prev_pts = true;
	}
	if (w > 0 && h > 0)
	{
		impl_->width = w;
		impl_->height = h;
	}

	if (w > 0 && h > 0 && fmt)
	{
		cv::Mat bgr;
		std::string f = fmt;
		if (f == "NV12" && map.size >= (size_t)w * h * 3 / 2)
		{
			// mppvideodec 会把高度按 16 对齐（如 1080→1088），caps 只报 1080；
			// 若按 1080 找 UV 平面，会把 Y 的 8 行当色度 → 顶部 16 行绿条。
			// 优先使用 GstVideoMeta 的真实平面偏移/stride，否则按缓冲区大小推断。
			const unsigned char* yp = map.data;
			const unsigned char* uvp = map.data + (size_t)w * h;
			int stride_y = w;
			int stride_uv = w;
			GstVideoMeta* vmeta = gst_buffer_get_video_meta(buf);
			if (vmeta)
			{
				yp = map.data + vmeta->offset[0];
				uvp = map.data + vmeta->offset[1];
				stride_y = vmeta->stride[0];
				stride_uv = vmeta->stride[1];
			}
			else if (map.size > (size_t)w * h * 3 / 2)
			{
				int total_rows = (int)(map.size / w);
				if (total_rows > h * 3 / 2)
				{
					uvp = map.data + (size_t)((total_rows * 2) / 3) * w;
				}
			}
			// 组装紧凑 NV12（Y + UV 按真实 stride 逐行拷贝）
			std::vector<unsigned char> nv12buf((size_t)w * h * 3 / 2);
			unsigned char* dst = nv12buf.data();
			for (int r = 0; r < h; ++r)
			{
				std::memcpy(dst + (size_t)r * w, yp + (size_t)r * stride_y, w);
			}
			unsigned char* uvdst = dst + (size_t)w * h;
			for (int r = 0; r < h / 2; ++r)
			{
				std::memcpy(uvdst + (size_t)r * w, uvp + (size_t)r * stride_uv, w);
			}
			cv::Mat nv12(h * 3 / 2, w, CV_8UC1, nv12buf.data());
			cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
		}
		else if (f == "BGR" && map.size >= (size_t)w * h * 3)
		{
			cv::Mat tmp(h, w, CV_8UC3, map.data);
			tmp.copyTo(bgr);
		}
		else if (f == "RGB" && map.size >= (size_t)w * h * 3)
		{
			cv::Mat tmp(h, w, CV_8UC3, map.data);
			cv::cvtColor(tmp, bgr, cv::COLOR_RGB2BGR);
		}
	if (!bgr.empty())
	{
		bgr.copyTo(frame);
		// 实测统计（两遍法用）：帧数 + 首尾 PTS
		impl_->frame_count++;
		if (GST_CLOCK_TIME_IS_VALID(pts))
		{
			if (!GST_CLOCK_TIME_IS_VALID(impl_->first_pts)) impl_->first_pts = pts;
			impl_->last_pts = pts;
		}
	}
	}
	else
	{
		frame.release();
	}

	gst_buffer_unmap(buf, &map);
	gst_sample_unref(sample);
	return !frame.empty();
}

bool GstVideoReader::isOpened() const { return impl_->opened; }
double GstVideoReader::fps() const    { return impl_->fps; }
int GstVideoReader::width() const     { return impl_->width; }
int GstVideoReader::height() const    { return impl_->height; }

bool GstVideoReader::caps_fps_authoritative() const
{
	return impl_->caps_fps_valid;
}

double GstVideoReader::measured_avg_fps() const
{
	if (impl_->frame_count < 2) return 0.0;
	// 优先容器时长：count/duration 更贴近"原长"（含最后一帧显示时长）
	gint64 dur = 0;
	if (impl_->pipeline &&
	    gst_element_query_duration(impl_->pipeline, GST_FORMAT_TIME, &dur) &&
	    dur > 0)
	{
		double d = (double)dur / GST_SECOND;
		double f = impl_->frame_count / d;
		if (d > 0.0 && f >= 1.0 && f <= 240.0) return f;
	}
	// 回退：PTS 首尾跨度
	if (GST_CLOCK_TIME_IS_VALID(impl_->first_pts) &&
	    GST_CLOCK_TIME_IS_VALID(impl_->last_pts) &&
	    impl_->last_pts > impl_->first_pts)
	{
		double span = (double)(impl_->last_pts - impl_->first_pts) / GST_SECOND;
		if (span > 0.0) return (impl_->frame_count - 1) / span;
	}
	return 0.0;
}

void GstVideoReader::release()
{
	if (impl_->pipeline)
	{
		gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
		if (impl_->sink) gst_object_unref(impl_->sink);
		gst_object_unref(impl_->pipeline);
		impl_->pipeline = nullptr;
		impl_->sink = nullptr;
	}
	impl_->opened = false;
	impl_->width = 0;
	impl_->height = 0;
	impl_->have_prev_pts = false;
	impl_->first_pts = GST_CLOCK_TIME_NONE;
	impl_->last_pts = GST_CLOCK_TIME_NONE;
	impl_->frame_count = 0;
	impl_->caps_fps_valid = false;
}

// ============================================================================
// GstVideoWriter：MPP 硬件编码（H.264 MP4）
// ============================================================================
struct GstVideoWriter::Impl
{
	GstElement* pipeline = nullptr;
	GstElement* src      = nullptr;
	double      fps      = 30.0;
	cv::Size    size     = {0, 0};
	guint64     frame_index = 0;
	bool        opened   = false;
	// 编码质量参数
	std::string rc_mode  = "fixqp";   // 固定 QP：质量优先，避免 CBR 低码率块状模糊
	int         qp_init  = 26;        // QP 越小越清晰（24~30 为常用清晰区间）
	std::string profile  = "high";    // CABAC，同码率下画质优于 baseline
};

GstVideoWriter::GstVideoWriter() : impl_(new Impl()) {}
GstVideoWriter::~GstVideoWriter() { release(); delete impl_; }

void GstVideoWriter::set_encoder_params(const std::string& rc_mode, int qp_init,
                                        const std::string& profile)
{
	impl_->rc_mode = rc_mode;
	impl_->qp_init = qp_init;
	impl_->profile = profile;
}

bool GstVideoWriter::open(const std::string& path, double fps, cv::Size size)
{
	ensure_gst_init();
	release();

	impl_->fps = fps > 0 ? fps : 30.0;
	impl_->size = size;
	impl_->frame_index = 0;

	// 编码码率按分辨率×帧率自适应（VBR/CBR 回退档；fixqp 模式下码率由 QP 决定）
	guint fps_int = (guint)(impl_->fps + 0.5);
	guint bps = (guint)((double)size.width * size.height * fps_int * 0.35);
	if (bps < 2000000) bps = 2000000;
	if (bps > 20000000) bps = 20000000;

	char desc[1536];
	snprintf(desc, sizeof(desc),
	         "appsrc name=src ! video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1 "
	         "! mpph264enc name=enc ! h264parse ! mp4mux ! filesink location=\"%s\"",
	         size.width, size.height, fps_int, path.c_str());

	GError* err = nullptr;
	GstElement* pipe = gst_parse_launch(desc, &err);
	if (!pipe)
	{
		LOG(MOD_PIPELINE, LOG_ERROR) << "GstVideoWriter: parse failed: "
		          << (err ? err->message : "unknown") << "\n";
		if (err) g_error_free(err);
		return false;
	}

	GstElement* src = gst_bin_get_by_name(GST_BIN(pipe), "src");
	if (!src)
	{
		LOG(MOD_PIPELINE, LOG_ERROR) << "GstVideoWriter: no appsrc\n";
		gst_object_unref(pipe);
		return false;
	}
	impl_->pipeline = pipe;
	impl_->src = src;

	GstElement* enc = gst_bin_get_by_name(GST_BIN(pipe), "enc");
	if (enc)
	{
		// profile：high（CABAC）优先，兼容性由调用方保证
		// 注意：g_object_set 对枚举属性必须传枚举值（int），传字符串会静默失败
		int profile_val = 0;   // baseline
		if (impl_->profile == "main") profile_val = 1;
		else if (impl_->profile == "high") profile_val = 2;
		if (impl_->profile == "high" || impl_->profile == "main" ||
		    impl_->profile == "baseline")
		{
			g_object_set(G_OBJECT(enc), "profile", profile_val, NULL);
		}
		if (impl_->rc_mode == "fixqp")
		{
			// 固定 QP：质量恒定、抗复杂纹理块状伪影；bps 自动由 QP 推导
			g_object_set(G_OBJECT(enc), "rc-mode", 2,
			             "qp-init", impl_->qp_init, NULL);
		}
		else
		{
			int rc = (impl_->rc_mode == "vbr") ? 0 : 1;
			g_object_set(G_OBJECT(enc), "rc-mode", rc,
			             "bps", bps, "bps-max", bps * 3 / 2, "bps-min", bps / 2, NULL);
		}
		LOG(MOD_PIPELINE, LOG_INFO) << "GstVideoWriter encoder: rc="
		          << impl_->rc_mode << " qp-init=" << impl_->qp_init
		          << " profile=" << impl_->profile << " bps=" << bps << "\n";
		gst_object_unref(enc);
	}

	// 显式声明 BT.709 limited 色彩学：
	// 否则 mpph264enc 会把 appsrc 默认 sRGB 色彩学直接写进 H.264 VUI
	// （color_space=gbr / color_range=pc），解码器按错误矩阵还原导致对比度被压。
	GstCaps* caps = gst_caps_new_simple("video/x-raw",
	                                    "format", G_TYPE_STRING, "BGR",
	                                    "width", G_TYPE_INT, size.width,
	                                    "height", G_TYPE_INT, size.height,
	                                    "framerate", GST_TYPE_FRACTION, fps_int, 1,
	                                    "colorimetry", G_TYPE_STRING, "bt709", NULL);
	g_object_set(G_OBJECT(src), "caps", caps, "is-live", FALSE,
	             "format", GST_FORMAT_TIME, "stream-type", GST_APP_STREAM_TYPE_STREAM, NULL);
	gst_caps_unref(caps);

	if (gst_element_set_state(pipe, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
	{
		LOG(MOD_PIPELINE, LOG_ERROR) << "GstVideoWriter: PLAYING failed\n";
		release();
		return false;
	}
	impl_->opened = true;
	return true;
}

bool GstVideoWriter::isOpened() const { return impl_->opened; }

bool GstVideoWriter::write(const cv::Mat& frame)
{
	if (!impl_->opened || !impl_->src) return false;

	cv::Mat cont = frame.isContinuous() ? frame : frame.clone();
	GstBuffer* buf = gst_buffer_new_allocate(NULL, cont.total() * cont.elemSize(), NULL);
	GstMapInfo map;
	gst_buffer_map(buf, &map, GST_MAP_WRITE);
	std::memcpy(map.data, cont.data, map.size);
	gst_buffer_unmap(buf, &map);

	guint64 f = (guint64)(impl_->fps + 0.5);
	GST_BUFFER_PTS(buf) = gst_util_uint64_scale(impl_->frame_index, GST_SECOND, f);
	GST_BUFFER_DTS(buf) = GST_BUFFER_PTS(buf);
	GST_BUFFER_DURATION(buf) = gst_util_uint64_scale(GST_SECOND, 1, f);
	impl_->frame_index++;

	GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(impl_->src), buf);
	return ret == GST_FLOW_OK;
}

void GstVideoWriter::release()
{
	if (impl_->pipeline)
	{
		if (impl_->src)
		{
			gst_app_src_end_of_stream(GST_APP_SRC(impl_->src));
			GstBus* bus = gst_element_get_bus(impl_->pipeline);
			GstMessage* msg = gst_bus_timed_pop_filtered(bus, 5 * GST_SECOND,
			                (GstMessageType)(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
			if (msg) gst_message_unref(msg);
			gst_object_unref(bus);
		}
		gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
		if (impl_->src) gst_object_unref(impl_->src);
		gst_object_unref(impl_->pipeline);
		impl_->pipeline = nullptr;
		impl_->src = nullptr;
	}
	impl_->opened = false;
	impl_->frame_index = 0;
}
