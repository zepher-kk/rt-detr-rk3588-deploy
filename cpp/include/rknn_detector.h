#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "rknn_api.h"

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
		 * @param input_buf  输入 DMA 缓冲（640x640 RGB，UINT8）
		 * @param out_boxes  输出预测框 [300,4] 归一化坐标
		 * @param out_logits 输出置信度 [300,NUM_CLASSES]
		 * @param num_boxes  输出框数量（固定 300）
		 * @param num_classes 输出类别数（固定 NUM_CLASSES）
		 * @return true 成功
		 */
		bool infer_zero_copy(const DmaBufferPtr& input_buf,
		                     std::vector<float>& out_boxes,
		                     std::vector<float>& out_logits,
		                     int& num_boxes,
		                     int& num_classes);

		/**
		 * @brief 传统推理（输入为 cv::Mat，通常用于测试）。
		 * @param preprocessed_img 640x640 RGB 图像
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

		// 仅保存 boxes 和 logits 的属性（用于零拷贝设置）
		rknn_tensor_attr boxes_attr_;
		rknn_tensor_attr logits_attr_;
		int boxes_idx_  = -1;
		int logits_idx_ = -1;

		// 加载模型文件
		unsigned char* load_model(const char* filename, int* model_size);

		rknn_tensor_attr input_attr_;   // 输入 tensor 属性，用于零拷贝设置

		// 查询并保存输入输出属性
		bool query_io_attrs();
};
