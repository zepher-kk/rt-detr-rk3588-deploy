# 路线图

## v4.0.x（维护）

- 长稳压测（10 分钟以上，记录降频曲线）
- 视频路径 Mat 池，去掉每帧 `clone()`
- 输出视频 / 画框（当前只统计检出数量）
- 共享输入缓冲，降低多实例内存争用

## v4.1：裁剪模型（150 query / 2 层 decoder）

已实测（`--num-queries 150 --decoder-layers 2`）：

| 项 | 300q / 3L | 150q / 2L |
|---|---|---|
| 推理 | 131.0 ms | **104.9 ms** |
| 3 实例 | 17.99 FPS | **22.63 FPS** |
| recall（对 PyTorch 300q） | 98.22% | 88.36% |
| meanIoU | 0.9805 | 0.9530 |

代价是 recall 从 98.2% 降到 88.4%，是否采用取决于业务对召回的要求。

## INT8 / 混合量化：已评估，当前不可行

三条路都实测过（数据见 [`mul-grid-bug.md`](mul-grid-bug.md) §4）：

1. **纯 INT8**：GridSample 的 GPU 实现不接受 INT8，运行时崩溃
2. **auto-hybrid**：179 个 FP16 边界，300q 410 ms（比 FP16 慢 3 倍）
3. **区域混合**（backbone+neck INT8 / decoder FP16）：`hybrid_quantization_step1`
   不接受 `op_target`，step2 不允许再 `config`，最终 GridSample 落到 CPU 报
   `Unsupport CPU op`

结论：在 native-GridSample 架构下，RKNN 2.5 上没有可用的 INT8 路径。
若要 INT8，只能回到 v3 的 Gather 展开图（无 GPU 算子），但那条路 decoder 单帧
255 ms，得不偿失。

## 更长期

- **双模型拆分**：backbone+neck 单独编 INT8，decoder 单独编 FP16，
  C++ 里串联两个 context。可绕开混合量化的 op_target 限制，但要拆 ONNX、
  多一次跨模型拷贝，收益与精度都未验证。
- 多板 / 多 NPU 横向扩展
