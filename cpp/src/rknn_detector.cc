#include "rknn_detector.h"
#include "logger.h"
#include "drm_alloc.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/ioctl.h>

// ============================================================================
// 辅助函数：加载模型文件
// ============================================================================
unsigned char* RKNNDetector::load_model(const char* filename, int* model_size)
{
	FILE* fp = fopen(filename, "rb");
	if (!fp)
	{
		LOG(MOD_RKNN, LOG_ERROR) << "Cannot open model: " << filename << "\n";
		return nullptr;
	}
	fseek(fp, 0, SEEK_END);
	int size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	unsigned char* data = (unsigned char*)malloc(size);
	if (data)
	{
		fread(data, 1, size, fp);
	}
	fclose(fp);
	*model_size = size;
	return data;
}

// ============================================================================
// 构造函数 / 析构函数
// ============================================================================
RKNNDetector::RKNNDetector()
{
	memset(&boxes_attr_, 0, sizeof(boxes_attr_));
	memset(&logits_attr_, 0, sizeof(logits_attr_));
}

RKNNDetector::~RKNNDetector()
{
	if (ctx_)
	{
		rknn_destroy(ctx_);
		ctx_ = 0;
	}
	if (model_data_)
	{
		free(model_data_);
		model_data_ = nullptr;
	}
}

// ============================================================================
// 查询输入输出属性（按元素数形状识别 boxes/logits）
// ============================================================================
bool RKNNDetector::query_io_attrs()
{
	rknn_input_output_num io_num;
	int ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if (ret < 0)
	{
		LOG(MOD_RKNN, LOG_ERROR) << "query IN_OUT_NUM failed: " << ret << "\n";
		return false;
	}
	n_input_  = io_num.n_input;
	n_output_ = io_num.n_output;
	LOG(MOD_RKNN, LOG_INFO) << "n_input=" << n_input_ << " n_output=" << n_output_ << "\n";

	// 查询输入（仅第一个，用于后续零拷贝设置）
	if (n_input_ > 0)
	{
		input_attr_.index = 0;
		int ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));
		if (ret == 0)
		{
			LogStream ls(MOD_RKNN, LOG_INFO);
			ls << "input[0] dims=" << input_attr_.n_dims << " [";
			for (int d = 0; d < (int)input_attr_.n_dims; ++d)
			{
				ls << input_attr_.dims[d] << (d + 1 < (int)input_attr_.n_dims ? "," : "");
			}
			ls << "] fmt=" << input_attr_.fmt << " type=" << input_attr_.type << "\n";
		}
	}

	// 遍历所有输出，收集全部张量属性，稍后按元素数形状识别 boxes 和 logits
	boxes_idx_ = -1;
	logits_idx_ = -1;
	output_attrs_.clear();
	output_attrs_.reserve(n_output_);
	for (int i = 0; i < n_output_; ++i)
	{
		rknn_tensor_attr attr;
		attr.index = i;
		ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
		if (ret < 0)
		{
			LOG(MOD_RKNN, LOG_ERROR) << "query output[" << i << "] failed: " << ret << "\n";
			continue;
		}
		output_attrs_.push_back(attr);
		LOG(MOD_RKNN, LOG_INFO) << "output[" << i << "] n_elems=" << attr.n_elems
		          << " type=" << attr.type << " fmt=" << attr.fmt
		          << " scale=" << attr.scale << " zp=" << attr.zp
		          << " name=" << attr.name << "\n";

	}

	resolve_rtdetr_output_indices(output_attrs_, boxes_idx_, logits_idx_);
	if (boxes_idx_ >= 0) boxes_attr_ = output_attrs_[boxes_idx_];
	if (logits_idx_ >= 0) logits_attr_ = output_attrs_[logits_idx_];

	if (boxes_idx_ < 0 || logits_idx_ < 0)
	{
		LOG(MOD_RKNN, LOG_ERROR) << "FATAL: cannot identify boxes/logits output\n";
		return false;
	}

	// 从 logits 推断类别数(日志)
	int detected_classes = logits_attr_.n_elems / 300;
	LOG(MOD_RKNN, LOG_INFO) << "Detected num_classes from model = " << detected_classes
	          << " (aligned with NUM_CLASSES=" << NUM_CLASSES << ")\n";

	return true;
}

// ============================================================================
// 初始化
// ============================================================================
bool RKNNDetector::init(const std::string& model_path, rknn_core_mask core_mask)
{
	int model_size;
	model_data_ = load_model(model_path.c_str(), &model_size);
	if (!model_data_)
	{
		LOG(MOD_RKNN, LOG_ERROR) << "load_model failed\n";
		return false;
	}

	int ret = rknn_init(&ctx_, model_data_, model_size, 0, nullptr);
	if (ret < 0)
	{
		LOG(MOD_RKNN, LOG_ERROR) << "rknn_init failed: " << ret << "\n";
		return false;
	}

	// 绑定 NPU 核心（可选）
	if (core_mask != RKNN_NPU_CORE_AUTO)
	{
		ret = rknn_set_core_mask(ctx_, core_mask);
		if (ret < 0)
		{
			LOG(MOD_RKNN, LOG_ERROR) << "rknn_set_core_mask failed: " << ret << "\n";
			// 不致命，继续
		}
		else
		{
			LOG(MOD_RKNN, LOG_INFO) << "Set core mask to " << core_mask << "\n";
		}
	}
	else
	{
		LOG(MOD_RKNN, LOG_INFO) << "Using auto NPU core selection\n";
	}

	if (!query_io_attrs())
	{
		return false;
	}

	output_buffers_.resize(n_output_);
	for (int i = 0; i < n_output_; ++i)
	{
		// 按张量元素数预分配 float 输出缓冲：推理时 is_prealloc 复用，避免每帧 malloc/free
		output_buffers_[i].assign(output_attrs_[i].n_elems, 0.0f);
	}

	is_init_ = true;
	return true;
}

// ============================================================================
// 零拷贝推理（输入为 DMA buffer）
// ============================================================================
bool RKNNDetector::infer_zero_copy(const DmaBufferPtr& input_buf,
                                   std::vector<float>& out_boxes,
                                   std::vector<float>& out_logits,
                                   int& num_boxes,
                                   int& num_classes)
{
	if (!is_init_ || !input_buf) return false;

	rknn_input inputs[1];
	memset(inputs, 0, sizeof(inputs));
	inputs[0].index = 0;
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].fmt = RKNN_TENSOR_NHWC;
	inputs[0].size = input_buf->size;
	inputs[0].buf = input_buf->ptr;
	inputs[0].pass_through = 0;

	if (rknn_inputs_set(ctx_, 1, inputs) < 0) return false;
	if (rknn_run(ctx_, NULL) < 0) return false;

	std::vector<rknn_output> outputs(n_output_);
	for (int i = 0; i < n_output_; ++i)
	{
		outputs[i].want_float = 1;   // 请求 float 反量化输出
		outputs[i].index = i;
		outputs[i].is_prealloc = 1;  // 写入 init 时预分配的缓冲，免动态分配
		outputs[i].buf = output_buffers_[i].data();
		outputs[i].size = output_attrs_[i].n_elems * sizeof(float);
	}
	if (rknn_outputs_get(ctx_, n_output_, outputs.data(), NULL) < 0) return false;

	// 从预分配缓冲中拷贝 boxes / logits 到输出向量
	const size_t boxes_count = (size_t)boxes_attr_.n_elems;
	const size_t logits_count = (size_t)logits_attr_.n_elems;
	if (out_boxes.size() != boxes_count) out_boxes.resize(boxes_count);
	if (out_logits.size() != logits_count) out_logits.resize(logits_count);
	if (boxes_count > 0)
	{
		memcpy(out_boxes.data(), output_buffers_[boxes_idx_].data(), boxes_count * sizeof(float));
	}
	if (logits_count > 0)
	{
		memcpy(out_logits.data(), output_buffers_[logits_idx_].data(), logits_count * sizeof(float));
	}
	num_boxes = (int)(boxes_count / 4);
	num_classes = (int)(logits_count / 300);

	rknn_outputs_release(ctx_, n_output_, outputs.data());
	return true;
}
// ============================================================================
// 传统推理（输入为 cv::Mat）
// ============================================================================
bool RKNNDetector::infer_only(const cv::Mat& preprocessed_img,
                              std::vector<float>& out_boxes,
                              std::vector<float>& out_logits,
                              int& num_boxes,
                              int& num_classes)
{
	if (!is_init_ || preprocessed_img.empty())
	{
		LOG(MOD_RKNN, LOG_ERROR) << "infer_only: not initialized or empty image\n";
		return false;
	}

	// 1. 设置输入（使用 rknn_inputs_set）
	rknn_input inputs[1];
	memset(inputs, 0, sizeof(inputs));
	inputs[0].index = 0;
	inputs[0].type  = RKNN_TENSOR_UINT8;
	inputs[0].fmt   = RKNN_TENSOR_NHWC;
	inputs[0].size  = preprocessed_img.total() * preprocessed_img.elemSize();
	inputs[0].buf   = preprocessed_img.data;
	inputs[0].pass_through = 0;
	if (rknn_inputs_set(ctx_, 1, inputs) < 0)
	{
		LOG(MOD_RKNN, LOG_ERROR) << "rknn_inputs_set failed\n";
		return false;
	}

	// 2. 推理
	int ret = rknn_run(ctx_, nullptr);
	if (ret < 0)
	{
		LOG(MOD_RKNN, LOG_ERROR) << "rknn_run failed: " << ret << "\n";
		return false;
	}

	// 3. 获取输出
	std::vector<rknn_output> outputs(n_output_);
	for (int i = 0; i < n_output_; ++i)
	{
		outputs[i].want_float  = 1;
		outputs[i].is_prealloc = 0;
		outputs[i].buf         = nullptr;
		outputs[i].size        = 0;
	}
	ret = rknn_outputs_get(ctx_, n_output_, outputs.data(), nullptr);
	if (ret < 0)
	{
		LOG(MOD_RKNN, LOG_ERROR) << "rknn_outputs_get failed: " << ret << "\n";
		return false;
	}

	// 4. 提取
	if (boxes_idx_ >= 0 && logits_idx_ >= 0)
	{
		float* b_data = (float*)outputs[boxes_idx_].buf;
		float* l_data = (float*)outputs[logits_idx_].buf;
		size_t b_count = outputs[boxes_idx_].size / sizeof(float);
		size_t l_count = outputs[logits_idx_].size / sizeof(float);
		out_boxes.assign(b_data, b_data + b_count);
		out_logits.assign(l_data, l_data + l_count);
		num_boxes  = b_count / 4;
		num_classes = NUM_CLASSES;
	}
	else
	{
		LOG(MOD_RKNN, LOG_ERROR) << "No boxes/logits indices found\n";
		ret = -1;
	}

	rknn_outputs_release(ctx_, n_output_, outputs.data());
	return ret == 0;
}
