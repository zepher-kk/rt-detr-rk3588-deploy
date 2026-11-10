# v4.1.0 — INT8 backbone + FP16 decoder（拆分模型）

## 亮点

- **backbone+neck 走 INT8**（w8a8，per-channel），**decoder 保持 FP16**（GridSample 继续走 GPU）
- 一套代码、两种配置：不加 `--decoder` 是 FP16 单模型，加了就是 INT8 拆分
- 3 worker **29.9 FPS**、4 worker **33.3 FPS**（FP16 单模型为 18.0 / 18.5 FPS）
- 单帧 e2e **70 ms**（FP16 单模型 122 ms）
- AP **0.1882**（VisDrone val 548 张，同口径；FP16 单模型 0.1985）

## 指标

| 配置 | AP | AP50 | 单帧 e2e | 3 worker | 4 worker | 体积 |
|---|---|---|---|---|---|---|
| FP16 单模型（v4.0.1） | 0.1985 | 0.3582 | 122 ms | 17.99 FPS | 18.53 FPS | 46.6 MB |
| **INT8 拆分（v4.1.0）** | **0.1882** | 0.3359 | **70 ms** | **29.9 FPS** | **33.3 FPS** | 17.5 + 8.1 MB |

> 与 FP16 的 0.0103 AP 差距来自结构裁剪（150 query / 2 层 decoder），
> INT8 本身在同一结构下无精度损失（150q/2L FP16 与 INT8 均为 0.1882）。

## 用法

```bash
# FP16 单模型
./rtdetr_pipeline -m rtdetr_tpopt2_4d_mulnpu.rknn -i uav.jpg -n 3 -p 2 -P 2 --cores split

# INT8 拆分
./rtdetr_pipeline -m rtdetr_v41_backbone_int8.rknn \
                  --decoder rtdetr_v41_decoder_fp16.rknn \
                  -i uav.jpg -n 4 -p 2 -P 2 --cores split
```

## 模型

模型文件不随仓库提供，构建步骤见
[`docs/model.md`](https://github.com/zepher-kk/rt-detr-rk3588-deploy/blob/v4.1.0/docs/model.md) §3b：

1. `python/export/split_decoder.py` 在 neck 特征处切图
2. `python/quantize/convert_int8.py` 编译 backbone（需要校准集）
3. `python/quantize/convert_native_gpu.py` 编译 decoder（多输入、不做归一化）

两个模型必须使用相同的 `--custom-string`，运行时启动会校验。

## 注意

- **v4.0.0 不要使用**：decoder 的 6 维广播 `Mul` 被 RKNN 2.5 编译成全 0，
  模型能跑但输出错误。v4.0.1 已修复。
- **v3.0.5 的 AP 与本版本不可直接比较**：v3 使用另一套前后处理口径；
  用本仓库脚本复评 v3 模型只有 0.0968。
- 摄像头 / GStreamer+MPP / 实时显示 / 按类阈值等 v3 能力尚未迁移到 v4，
  需要时请使用 `v3.0.5` 分支。

## 文档

- [新手指引](https://github.com/zepher-kk/rt-detr-rk3588-deploy/blob/v4.1.0/README-beginner.md)
- [架构与数据通路](https://github.com/zepher-kk/rt-detr-rk3588-deploy/blob/v4.1.0/docs/architecture.md)
- [基准数据](https://github.com/zepher-kk/rt-detr-rk3588-deploy/blob/v4.1.0/docs/benchmarks.md)
- [模型与编译](https://github.com/zepher-kk/rt-detr-rk3588-deploy/blob/v4.1.0/docs/model.md)
- [常见问题](https://github.com/zepher-kk/rt-detr-rk3588-deploy/blob/v4.1.0/docs/troubleshooting.md)
- [6 维 Mul 缺陷与修复](https://github.com/zepher-kk/rt-detr-rk3588-deploy/blob/v4.1.0/docs/mul-grid-bug.md)
