#pragma once

#include <vector>
#include "types.h"

// ============================================================================
// RT-DETR 后处理（NEON SIMD 加速）
// ============================================================================

/**
 * @brief 解码 RT-DETR 网络输出，过滤并转换检测框。
 * @param boxes_data  [300,4] 归一化坐标 (cx, cy, w, h)
 * @param scores_data [300,NUM_CLASSES] 各类别分数
 * @param num_boxes   盒子数量（固定 300）
 * @param orig_w      原始图像宽度（用于反归一化）
 * @param orig_h      原始图像高度（用于反归一化）
 * @param conf_thres  置信度阈值
 * @param num_classes 类别数（由模型实际决定，但此处固定使用 NUM_CLASSES）
 * @return 过滤后的检测结果列表
 * @note 已集成 NEON 加速（若编译启用），速度优于纯 C++。
 */
std::vector<DetectResult> decode_rtdetr_output(float* boxes_data,
                                                float* scores_data,
                                                int num_boxes,
                                                int orig_w, int orig_h,
                                                float conf_thres,
                                                int num_classes = NUM_CLASSES);  

/**
 * @brief 在图像上绘制检测框和标签（用于可视化）。
 * @param image   待绘制图像（会被修改）
 * @param results 检测结果列表
 */
void draw_results(cv::Mat& image, const std::vector<DetectResult>& results);
