# v4 基准数据

平台：RK3588（LubanCat，Linux 6.1.84）
模型：`rtdetr_native_tpopt2_mulnpu_fp16.rknn`（FP16 / 300 query / native GridSample，44.4 MB）
工具链：rknn-toolkit2 2.5.0 / librknnrt 2.5.0
频率：CPU governor=performance，NPU 1 GHz，GPU 1 GHz

## 1. 吞吐（图片列表）

输入 `uav.jpg` 1360×765，每组 20 帧 warmup + 200~300 帧测量。

| 配置 | 聚合 FPS | pre | npu | post |
|---|---|---|---|---|
| 单实例（三核，`-n 1`） | **7.96** | 5.89 ms | 125.67 ms | 0.01 ms |
| 2 实例各绑 1 核 | 12.70 | 4.00 ms | 156.94 ms | 0.01 ms |
| **3 实例各绑 1 核** | **18.06** | 5.31 ms | 165.68 ms | 0.01 ms |
| **4 实例（3 核）** | **18.66** | 4.71 ms | 213.69 ms | 0.01 ms |
| 3 实例共享三核 | 10.17 | — | 294 ms | — |

## 2. 吞吐（视频）

`cars.mp4`：399 帧，30 fps，1360×765。

| 配置 | 聚合 FPS | decode | pre | npu |
|---|---|---|---|---|
| 单实例（三核） | 7.85 | 3.15 ms | 4.11 ms | 127.23 ms |
| **3 实例各绑 1 核** | **17.65** | 4.33 ms | 6.34 ms | 167.66 ms |
| 4 实例（3 核） | 15.77 | 5.87 ms | 5.97 ms | 252.24 ms |

视频比图片慢约 1.3%，解码已完全被流水线掩盖。

## 3. 单帧时延

`--latency`（单帧串行、无排队）：

| 指标 | 值 |
|---|---|
| mean | 134.3 ms |
| best | **128.4 ms** |
| 300 帧分布 | p50 129.2 / p90 131.8 / p95 152.7 / max 157.1 ms |

## 4. 精度

11 张 VisDrone 图，conf≥0.45：

| 指标 | 值 |
|---|---|
| recall | 100% |
| precision | 100% |
| meanIoU | 0.9912 |
| 检出总数 | 220 / 220（与 PyTorch 一致） |

uav.jpg 单图：24 检出，与 Python 管线逐帧一致。

## 5. 模型内部分段（优化前 → 优化后）

| 阶段 | 优化前 | 现在（推算） |
|---|---|---|
| Backbone（model.0–8） | 45.1 ms | 45.1 ms |
| AIFI（model.9） | 5.3 ms | 5.3 ms |
| Neck（model.10–25） | 27.3 ms | 27.3 ms |
| Decoder/Head（model.26） | 108.5 ms | ~46 ms |
| **合计（run）** | 186.2 ms | ~124 ms |

Decoder 的改进：Mul→NPU（-24.9 ms）、消 12 宽 transpose（-13.1 ms）、
消 value 布局往返（-16.6 ms）。

## 6. 复测要求

固定模型文件（md5 见 README）、输入内容、`-n/-p/-P/--cores`、是否节流、
输出方式。更换板卡或镜像后需重新测。长稳（10 分钟以上）尚未纳入本表。
