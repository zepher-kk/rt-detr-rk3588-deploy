# v4.0.1 基准数据

平台：RK3588（LubanCat，Linux 6.1.84）
模型：`rtdetr_tpopt2_4d_mulnpu.rknn`（FP16 / 300 query / 3 层 decoder / native GridSample，46.6 MB）
工具链：rknn-toolkit2 2.5.0 / librknnrt 2.5.0
频率：CPU governor=performance，NPU 1 GHz，GPU 1 GHz

> v4.0.0 的旧数据是在 decoder 缺陷（6 维 Mul 被编译成全 0）下测得的，
> `out[2]` 分数通路是错的，不能引用。下表全部为 4-D 修复后的复测。

## 1. 吞吐（图片列表）

输入 `uav.jpg` 1360×765，每组 30 帧 warmup + 300 帧测量。

| 配置 | 聚合 FPS | pre | npu | post |
|---|---|---|---|---|
| 单实例（三核，`-n 1`） | **8.11** | 3.14 ms | 123.32 ms | 0.01 ms |
| 2 实例各绑 1 核 | 12.87 | 4.27 ms | 155.01 ms | 0.01 ms |
| **3 实例各绑 1 核** | **17.99** | 4.77 ms | 166.31 ms | 0.01 ms |
| **4 实例（3 核）** | **18.53** | 4.04 ms | 215.30 ms | 0.01 ms |

## 2. 吞吐（视频）

`cars.mp4`：399 帧，30 fps，1360×765，取 200 帧。

| 配置 | 聚合 FPS | decode | pre | npu |
|---|---|---|---|---|
| 单实例（三核） | 8.10 | 3.19 ms | 4.60 ms | 123.12 ms |
| **3 实例各绑 1 核** | **17.63** | 3.31 ms | 5.21 ms | 168.35 ms |
| 4 实例（3 核） | 18.26 | 5.91 ms | 4.76 ms | 217.09 ms |

视频比图片慢约 2%，解码已完全被流水线掩盖。

## 3. 单帧时延

`--latency`（单帧串行、无排队，25 次）：

| 指标 | 值 |
|---|---|
| mean | 120.2 ms |
| best | **116.8 ms** |

单模型 RKNN 推理（`board_eval.py`，150 次）：p50 **131.0** / p95 **133.1 ms**。

## 4. 精度

11 张 VisDrone 图，conf≥0.45，与 PyTorch 300q/3L 参考对齐：

| 指标 | 值 |
|---|---|
| recall | 98.22% |
| precision | 98.62% |
| meanIoU | 0.9805 |

高置信 query 与 PyTorch 的逐元素差异 < 1e-3（框 1e-4 量级），
差异集中在低分/背景 query。

## 5. 模型内部分段

| 阶段 | 耗时 |
|---|---|
| Backbone（model.0–8） | ~45 ms |
| AIFI（model.9） | ~5 ms |
| Neck（model.10–25） | ~27 ms |
| Decoder/Head（model.26） | ~46 ms |
| **合计（run）** | **~123 ms** |

Decoder 的关键改动：6 维 `Mul` 改 4 维（修复全 0 缺陷，且让 `Mul` 留在 NPU，
比放 CPU 省 26 ms）、消 12 宽 transpose（-13.1 ms）、消 value 布局往返（-16.6 ms）。

## 6. 复测要求

固定模型文件（md5 见 README）、输入内容、`-n/-p/-P/--cores`、是否节流、
输出方式。更换板卡或镜像后需重新测。长稳（10 分钟以上）尚未纳入本表。
