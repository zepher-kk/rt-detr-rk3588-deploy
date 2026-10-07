#pragma once

#include <vector>
#include <string>
#include "types.h"

// ============================================================================
// RT-DETR 后处理（NEON SIMD 加速）
// ============================================================================

/**
 * @brief 置信度阈值配置：全局阈值 + 可选按类阈值。
 *
 * 背景：VisDrone 10 类的分数分布差异极大（Car 偏高、Bicycle/Awning 偏低），
 * 单一全局阈值无法同时服务大目标与小目标/稀有类。本结构支持为每个类别
 * 单独指定生效阈值；未启用按类阈值时全部回落到 global。
 */
struct ConfConfig
{
	float global = 0.45f;                 //!< 全局阈值（对应 -c/--conf）
	float per_class[NUM_CLASSES];         //!< 各类别阈值
	bool  per_class_set = false;          //!< true 时按类别取阈值，false 时全部用 global

	ConfConfig()
	{
		for (int i = 0; i < NUM_CLASSES; ++i) per_class[i] = global;
	}

	//!< 把全部类别阈值重置为 base（不改 per_class_set 标志）
	void reset(float base)
	{
		global = base;
		for (int i = 0; i < NUM_CLASSES; ++i) per_class[i] = base;
	}

	//!< 取某个类别实际生效的阈值（类别越界或未启用按类阈值时返回 global）
	float threshold_for(int cls) const
	{
		return (per_class_set && cls >= 0 && cls < NUM_CLASSES) ? per_class[cls] : global;
	}
};

/**
 * @brief 解析按类阈值规格串，例如 "0:-0.40,3:0.00,9:-0.55"。
 *
 * 语法：`<class_id>:<threshold>` 以 `,` 或 `;` 分隔；未列出的类别保持 cfg 原值
 * （通常先调用 cfg.reset(全局阈值)）。解析成功会把 cfg.per_class_set 置为 true。
 * @param spec 规格串
 * @param cfg  输入输出配置
 * @param err  失败原因（可为 nullptr）
 * @return true 解析成功；false 语法错误或类别/数值非法
 */
bool parse_conf_class(const std::string& spec, ConfConfig& cfg, std::string* err = nullptr);

/**
 * @brief 应用置信度阈值预设档位。
 *
 * - `balanced`：仅用全局阈值（合并 F1 最优）；
 * - `recall`  ：按类标定向量（VisDrone-2019-DET val 全量 548 张、IoU≥0.5、
 *               以宏平均 F1 最大为准则逐类标定）。
 * @param name 档位名（balanced / recall）
 * @param base 全局阈值（-c）
 * @param cfg  输出配置
 * @param err  失败原因（可为 nullptr）
 * @return true 成功；false 未知档位
 */
bool apply_conf_profile(const std::string& name, float base, ConfConfig& cfg, std::string* err = nullptr);

/**
 * @brief 解码 RT-DETR 网络输出，过滤并转换检测框。
 * @param boxes_data  [300,4] 归一化坐标 (cx, cy, w, h)
 * @param scores_data [300,NUM_CLASSES] 各类别分数
 * @param num_boxes   盒子数量（固定 300）
 * @param orig_w      原始图像宽度（用于反归一化）
 * @param orig_h      原始图像高度（用于反归一化）
 * @param conf_thres  置信度阈值
 * @param num_classes 类别数（由模型实际决定，但此处固定使用 NUM_CLASSES）
 * @param conf_cfg    可选的按类阈值配置；为 nullptr 时全部使用 conf_thres
 * @return 过滤后的检测结果列表
 * @note 已集成 NEON 加速（若编译启用），速度优于纯 C++。
 */
std::vector<DetectResult> decode_rtdetr_output(float* boxes_data,
                                                float* scores_data,
                                                int num_boxes,
                                                int orig_w, int orig_h,
                                                float conf_thres,
                                                int num_classes = NUM_CLASSES,
                                                const ConfConfig* conf_cfg = nullptr);

/**
 * @brief 画框样式（随输入分辨率自适应）。
 */
struct DrawStyle
{
	int    thickness      = 2;    //!< 检测框线宽（px）
	double font_scale     = 0.6;  //!< putText 字号
	int    font_thickness = 2;    //!< 文字描边粗细（px）
	double small_box_th   = 0.0;  //!< 小目标判定阈值（min(w,h) px）：小于则只画细框、不画文字
};

/**
 * @brief 根据输入分辨率计算自适应画框样式。
 * @param frame_w 帧宽（px）
 * @param frame_h 帧高（px）
 * @return DrawStyle；以 720p 为基准（2px 线宽 / 0.6 字号），随分辨率线性缩放并限幅
 *         （线宽 1~4px、字号 0.35~1.2），小目标阈值随分辨率等比放大。
 */
DrawStyle compute_draw_style(int frame_w, int frame_h);

/**
 * @brief 在图像上绘制检测框和标签（用于可视化）。
 * @param image   待绘制图像（会被修改）
 * @param results 检测结果列表
 * @note 颜色按类别固定（确定性调色板）；线宽/字号随分辨率自适应；
 *       小目标只画细框不画文字，避免遮挡密集目标。
 */
void draw_results(cv::Mat& image, const std::vector<DetectResult>& results);
