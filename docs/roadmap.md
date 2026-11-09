# 路线图

## v4.0.x（维护）

- 长稳压测（10 分钟以上，记录降频曲线）
- 视频路径 Mat 池，去掉每帧 `clone()`
- 输出视频 / 画框（当前只统计检出数量）
- 共享输入缓冲，降低多实例内存争用

## v4.1.0（计划）：backbone/neck 混合量化

目标：backbone + neck 走 INT8，GridSample 保持 FP16，decoder 视精度决定。

预期：

| 项 | v4.0.0 | v4.1.0（目标） |
|---|---|---|
| backbone 45.1 ms | FP16 | INT8，约 2× |
| neck 27.3 ms | FP16 | INT8，约 2× |
| decoder ~46 ms | FP16 | FP16 / 部分 INT8 |
| 单实例 run | 124 ms | ~90 ms |
| 3 实例吞吐 | 18.1 FPS | ~22 FPS |

风险：

1. GridSample 只接受 FP16，混合图可能触发额外 fallback
2. RKNN 的 `hybrid_quantization` 流程需要校准集，且不一定支持「指定某算子保持 FP16」
3. INT8 精度损失需要按类阈值重新标定

验收口径与 v4.0.0 相同：同一套 C++ 工程、同一组图片/视频、recall/precision/IoU
对比 PyTorch FP32 参考。

## 更长期

- 150 query / 更浅 decoder 的模型（需重训），预期单实例 ~80 ms
- 多板 / 多 NPU 横向扩展
