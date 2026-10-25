#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "rknn_api.h"

/**
 * @brief 在全部输出张量中按元素数识别 RT-DETR 的 boxes 与 logits。
 * @param attrs       查询到的全部输出张量属性
 * @param boxes_idx   输出：boxes 张量索引（未找到为 -1）
 * @param logits_idx  输出：logits 张量索引（未找到为 -1）
 * @param boxes_elems  boxes 张量元素数；0 表示从同 query 数的 logits 自动推断
 * @param logits_elems logits 张量元素数；0 表示自动推断
 * @note 自动模式兼容 150/300 query，最后一个匹配者胜。
 */
inline void resolve_rtdetr_output_indices(
    const std::vector<rknn_tensor_attr>& attrs,
    int& boxes_idx,
    int& logits_idx,
    int boxes_elems = 0,
    int logits_elems = 0)
{
	boxes_idx = -1;
	logits_idx = -1;
	if (boxes_elems > 0 && logits_elems > 0)
	{
		for (size_t i = 0; i < attrs.size(); ++i)
		{
			if (attrs[i].n_elems == (uint32_t)boxes_elems) boxes_idx = (int)i;
			if (attrs[i].n_elems == (uint32_t)logits_elems) logits_idx = (int)i;
		}
		return;
	}

	for (size_t i = 0; i < attrs.size(); ++i)
	{
		if (attrs[i].n_elems % 4 != 0) continue;
		for (size_t j = i + 1; j < attrs.size(); ++j)
		{
			if (attrs[j].n_elems % NUM_CLASSES != 0) continue;
			const size_t boxes_queries = attrs[i].n_elems / 4;
			const size_t logits_queries = attrs[j].n_elems / NUM_CLASSES;
			if (boxes_queries > 0 && boxes_queries == logits_queries)
			{
				boxes_idx = (int)i;
				logits_idx = (int)j;
			}
		}
	}
}

/**
 * @brief 从 boxes/logits 元素数推导 query 数和类别数。
 * @return 类别数；形状不一致时返回 -1。
 */
inline int infer_rtdetr_num_classes(size_t boxes_elems, size_t logits_elems)
{
	if (boxes_elems == 0 || boxes_elems % 4 != 0) return -1;
	const size_t query_count = boxes_elems / 4;
	if (logits_elems % query_count != 0) return -1;
	return (int)(logits_elems / query_count);
}

/**
 * @brief RKNN 推理封装类，支持零拷贝 DMA 输入和传统 cv::Mat 输入。
 */
class RKNNDetector
{
	public:
		RKNNDetector();
		~RKNNDetector();

		/**
		 * @brief 加载模型并查询输入输出属性。
		 * @param model_path RKNN 模型文件路径
		 * @param core_mask  NPU 核心掩码（多核分配）
		 * @return true 成功
		 */
		bool init(const std::string& model_path, rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO);

		/**
		 * @brief 零拷贝推理（输入为 DMA 缓冲）。
 * @param input_buf  输入 DMA 缓冲（INPUT_WIDTH x INPUT_HEIGHT RGB，UINT8）
 * @param out_boxes  输出预测框 [num_boxes,4] 归一化坐标
 * @param out_logits 输出置信度 [num_boxes,num_classes]
 * @param num_boxes  输出框数量（支持 150/300 query）
 * @param num_classes 输出类别数
		 * @return true 成功
		 */
		bool infer_zero_copy(const DmaBufferPtr& input_buf,
		                     std::vector<float>& out_boxes,
		                     std::vector<float>& out_logits,
		                     int& num_boxes,
		                     int& num_classes);

		/**
		 * @brief 传统推理（输入为 cv::Mat，通常用于测试）。
		 * @param preprocessed_img INPUT_WIDTH×INPUT_HEIGHT RGB 图像
		 * @param out_boxes  输出预测框
		 * @param out_logits 输出置信度
		 * @param num_boxes  框数量
		 * @param num_classes 类别数
		 * @return true 成功
		 */
		bool infer_only(const cv::Mat& preprocessed_img,
		                std::vector<float>& out_boxes,
		                std::vector<float>& out_logits,
		                int& num_boxes,
		                int& num_classes);

	private:
		rknn_context  ctx_ = 0;
		unsigned char* model_data_ = nullptr;
		bool is_init_ = false;

		int n_input_  = 0;
		int n_output_ = 0;

		std::vector<rknn_tensor_attr> output_attrs_;  // 全部输出张量属性
		rknn_tensor_attr boxes_attr_;
		rknn_tensor_attr logits_attr_;
		int boxes_idx_  = -1;
		int logits_idx_ = -1;
		std::vector<std::vector<float>> output_buffers_;  // 输出预分配缓冲（避免每帧动态分配）

		// 加载模型文件
		unsigned char* load_model(const char* filename, int* model_size);

		rknn_tensor_attr input_attr_;   // 输入 tensor 属性，用于零拷贝设置

		// 查询并保存输入输出属性
		bool query_io_attrs();
};
