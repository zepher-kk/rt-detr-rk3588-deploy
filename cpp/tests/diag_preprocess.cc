// ============================================================================
// diag_preprocess.cc - 预处理变体诊断程序（临时）
//
// 用法: ./diag_preprocess <model.rknn> <image.jpg>
// 目的: 对比 4 种预处理变体对同一模型输出的最高 logit 分数，
//       用于定位检测结果不一致时 RGA 与 CPU 预处理的输入差异。
// ============================================================================
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"

static unsigned char* load_model(const char* path, int* size)
{
	FILE* fp = fopen(path, "rb");
	if (!fp) return nullptr;
	fseek(fp, 0, SEEK_END);
	*size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	unsigned char* d = (unsigned char*)malloc(*size);
	if (d) fread(d, 1, *size, fp);
	fclose(fp);
	return d;
}

static void run_once(rknn_context ctx, const cv::Mat& in, const char* label)
{
	rknn_input inputs[1];
	memset(inputs, 0, sizeof(inputs));
	inputs[0].index = 0;
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].fmt = RKNN_TENSOR_NHWC;
	inputs[0].size = in.total() * in.elemSize();
	inputs[0].buf = in.data;
	inputs[0].pass_through = 0;
	if (rknn_inputs_set(ctx, 1, inputs) < 0)
	{
		printf("%s: inputs_set fail\n", label);
		return;
	}
	if (rknn_run(ctx, NULL) < 0)
	{
		printf("%s: run fail\n", label);
		return;
	}

	rknn_input_output_num io_num;
	if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) < 0)
	{
		printf("%s: query fail\n", label);
		return;
	}
	std::vector<rknn_output> outputs(io_num.n_output);
	for (int i = 0; i < io_num.n_output; ++i)
	{
		outputs[i].want_float = 1;
		outputs[i].is_prealloc = 0;
		outputs[i].buf = nullptr;
		outputs[i].size = 0;
	}
	if (rknn_outputs_get(ctx, io_num.n_output, outputs.data(), NULL) < 0)
	{
		printf("%s: outputs_get fail\n", label);
		return;
	}

	float best = -1e30f;
	int best_i = -1;
	for (int i = 0; i < io_num.n_output; ++i)
	{
		float* data = (float*)outputs[i].buf;
		size_t cnt = outputs[i].size / sizeof(float);
		if (cnt == 300 * 10)
		{
			for (size_t k = 0; k < cnt; ++k)
			{
				if (data[k] > best)
				{
					best = data[k];
					best_i = i;
				}
			}
		}
	}
	printf("%-24s max_logit=%.4f (out_idx=%d)\n", label, best, best_i);
	rknn_outputs_release(ctx, io_num.n_output, outputs.data());
}

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		printf("usage: %s <model.rknn> <image.jpg>\n", argv[0]);
		return 1;
	}

	int msize = 0;
	unsigned char* mdata = load_model(argv[1], &msize);
	if (!mdata)
	{
		printf("model load fail\n");
		return 1;
	}
	rknn_context ctx = 0;
	if (rknn_init(&ctx, mdata, msize, 0, NULL) < 0)
	{
		printf("rknn_init fail\n");
		return 1;
	}

	cv::Mat bgr = cv::imread(argv[2]);
	if (bgr.empty())
	{
		printf("imread fail\n");
		return 1;
	}
	printf("img=%dx%d\n", bgr.cols, bgr.rows);

	// 变体1: 直接拉伸 + BGR→RGB（当前 workspace 行为）
	{
		cv::Mat v;
		cv::resize(bgr, v, cv::Size(640, 640));
		cv::cvtColor(v, v, cv::COLOR_BGR2RGB);
		run_once(ctx, v, "v1_direct_bgr2rgb");
	}

	// 变体2: 直接拉伸，不做颜色转换（BGR 原样送入）
	{
		cv::Mat v;
		cv::resize(bgr, v, cv::Size(640, 640));
		run_once(ctx, v, "v2_direct_bgr_raw");
	}

	// 变体3: letterbox 保边填充 + BGR→RGB
	{
		float scale = std::min(640.0f / bgr.cols, 640.0f / bgr.rows);
		int nw = std::round(bgr.cols * scale);
		int nh = std::round(bgr.rows * scale);
		cv::Mat tmp;
		cv::resize(bgr, tmp, cv::Size(nw, nh));
		cv::Mat padded(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
		tmp.copyTo(padded(cv::Rect((640 - nw) / 2, (640 - nh) / 2, nw, nh)));
		cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);
		run_once(ctx, padded, "v3_letterbox_bgr2rgb");
	}

	// 变体4: letterbox 保边填充，不做颜色转换
	{
		float scale = std::min(640.0f / bgr.cols, 640.0f / bgr.rows);
		int nw = std::round(bgr.cols * scale);
		int nh = std::round(bgr.rows * scale);
		cv::Mat tmp;
		cv::resize(bgr, tmp, cv::Size(nw, nh));
		cv::Mat padded(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
		tmp.copyTo(padded(cv::Rect((640 - nw) / 2, (640 - nh) / 2, nw, nh)));
		run_once(ctx, padded, "v4_letterbox_bgr_raw");
	}

	rknn_destroy(ctx);
	free(mdata);
	return 0;
}
