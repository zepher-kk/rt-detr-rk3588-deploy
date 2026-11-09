# 模型与工具链

## 1. 产物

| 文件 | 说明 |
|---|---|
| `rtdetr_tpopt2_4d_mulnpu.rknn` | FP16 单模型，46.6 MB |
| `rtdetr_v41_backbone_int8.rknn` | v4.1 backbone+neck，INT8，17.5 MB |
| `rtdetr_v41_decoder_fp16.rknn` | v4.1 decoder，FP16，8.1 MB |

原生输入：

- FP16 单模型：FP16 NHWC `1×640×640×3`，2,457,600 字节
- v4.1 backbone：**UINT8/INT8** NHWC `1×640×640×3`，1,228,800 字节
  （编译时 `mean=0, std=255`，运行时传原始 uint8，注意按 int8 语义存放）

输出：`y = N×(4+10)`（boxes + sigmoid 分数），另有 4 个辅助输出。
N = 300（FP16 单模型）或 150（v4.1）。

模型不在仓库内，按下面步骤自行生成。

## 2. 导出 ONNX（PC / WSL）

```bash
export UAVDETR_SRC=/path/to/UAV-DETR-rt      # ultralytics 源码目录

python python/export/export_transpose_opt.py \
    --weights best.pt \
    --value-nchw \
    --fold-offset-scale \
    --msda-4d-grid \
    --output model/rtdetr_tpopt2_4d.onnx
```

`--value-nchw` 会同时应用两处等价重写：

1. deformable attention 的 `stack → flatten(-2)` 改为逐 level 累加
2. `value` 保持 channel-first，沿序列轴切片

`--fold-offset-scale --msda-4d-grid` 是 v4.0.1 的修复：

- 把 `0.5 / n_points` 折进 `sampling_offsets` 权重，去掉一个常量 Mul
- 把采样点的乘法 / `grid = 2*loc-1` 从 6 维改成 4 维。
  RKNN 2.5 对 `(1,300,8,3,4,2) × (1,300,1,1,1,2)` 这种 6 维广播 `Mul`
  **会返回全 0**，导致采样点塌到参考框中心、decoder 全错。

三处都不改变数学结果（PyTorch / ONNX 侧逐位相同）。

## 3. 编译 RKNN

```bash
python python/quantize/convert_native_gpu.py \
    model/rtdetr_tpopt2_4d.onnx \
    model/rtdetr_tpopt2_4d_mulnpu.rknn \
    --opt-level 0 \
    --disable-rules squeeze_nd_mul,fuse_mul_into_gemm \
    --op-target-map 'GridSample=gpu,Mul=npu'
```

参数说明：

| 参数 | 必须 | 原因 |
|---|---|---|
| `GridSample=gpu` | 是 | 默认 auto 会分到 CPU，报 `Unsupport CPU op` |
| `Mul=npu` | 是 | 只有配合 `--msda-4d-grid` 才安全；旧的 6 维图上 `Mul=npu` 会被编译成全 0。临时兜底可用 `--mul-cpu-scope all`（+26 ms） |
| `--opt-level 0` | 建议 | `opt-level 3` 会崩在 `fuse_mul_into_gemm`；且 opt0/opt3 规则应用数完全相同（2879），性能无差 |
| `--disable-rules ...` | 是 | opt3 下必需；opt0 下保留可复现性 |

## 3b. v4.1 拆分：backbone INT8 + decoder FP16

混合量化在 RKNN 2.5 上走不通（`hybrid_quantization_step1` 不接受 `GridSample` 的
`op_target`，step2 又禁止再 `config`，最终 GridSample 落到 CPU 报
`Unsupport CPU op`），所以改成两个模型串联。

```bash
# 1) 在 neck 输出处切图（3 张特征图 = 4.3 MB FP16）
python python/export/split_decoder.py model/rtdetr_q150_l2_4d.onnx \
    --feat /model.19/cv3/act/Mul_output_0 /model.22/cv3/act/Mul_output_0 \
           /model.25/cv3/act/Mul_output_0 \
    --out-a model/q150_backbone.onnx --out-b model/q150_decoder.onnx

# 2) backbone INT8（w8a8, per-channel，需要校准集）
python python/quantize/convert_int8.py model/q150_backbone.onnx \
    model/rtdetr_v41_backbone_int8.rknn \
    --dataset calib.txt --custom-string v41-q150l2

# 3) decoder FP16（多输入，不做图像归一化）
python python/quantize/convert_native_gpu.py model/q150_decoder.onnx \
    model/rtdetr_v41_decoder_fp16.rknn --opt-level 0 \
    --disable-rules squeeze_nd_mul,fuse_mul_into_gemm \
    --op-target-map 'GridSample=gpu,Mul=npu' \
    --inputs /model.19/cv3/act/Mul_output_0,/model.22/cv3/act/Mul_output_0,/model.25/cv3/act/Mul_output_0 \
    --input-sizes 1x256x80x80\;1x256x40x40\;1x256x20x20 --no-normalize \
    --custom-string v41-q150l2
```

两个模型必须带**相同的 `--custom-string`**：C++ 启动时会比对，
不一致直接拒绝启动（实测报 `model mismatch: backbone '...' vs decoder '...'`）。

### 交接时的三个坑（已在 `split_detector.cc` 里处理）

| 坑 | 现象 | 处理 |
|---|---|---|
| backbone 输入是 **INT8**（zp=−128, scale=1/255） | 直接拷 uint8 等于整体亮度 +0.5，特征相关性只有 0.88 | 逐字节 `^ 0x80` |
| `rknn_outputs_get(want_float=0)` 返回**连续 NCHW**（不是 native NC1HWC2），且缓冲区尾部有 padding | 按 NC1HWC2 读会完全错位 | 按 `[C][H][W]` 连续读，缓冲按 `size_with_stride` 分配 |
| decoder 输入是 **native NC1HWC2 FP16**（C2=8），backbone 输出是 NCHW INT8 | 需要反量化 + 重排 | 一次遍历 `(q−zp)·scale` → fp16，直接写 decoder 的 io_mem（<1 ms） |

## 4. 约束

| 项 | 限制 |
|---|---|
| 精度 | GridSample 所在子图必须 FP16（GPU 实现不接受 INT8） |
| GridSample | 仅 bilinear + padding_mode=zeros |
| 硬件 | RK3588（RK356X 官方不支持） |
| 系统 | 需要导出 `clImportMemoryARM` 的 OpenCL 库 |

## 5. 板端 OpenCL

Mali blob 才导出 `clImportMemoryARM`，系统自带的 ocl-icd loader 不导出：

```bash
mkdir -p /home/cat/opencl_shim
ln -sf /usr/lib/aarch64-linux-gnu/libmali-x11/libmali-valhall-g610-g13p0-x11-wayland-gbm.so \
       /home/cat/opencl_shim/libOpenCL.so
export LD_LIBRARY_PATH=/home/cat/opencl_shim:$LD_LIBRARY_PATH
```

`python/deploy/launch_native.sh` 会自动完成上述检查与设置。

## 6. 为什么不做单模型 INT8

GridSample 的 GPU 实现只接受 FP16，而 RKNN 2.5 的单模型构建只有一个全局 dtype；
混合量化的 hybrid 流程又不携带 `op_target`，GridSample 会落到 CPU 报
`Unsupport CPU op: GridSample`。所以 INT8 只能通过“拆两个模型”实现，
decoder 保持 FP16。详见 `docs/roadmap.md` 和 `docs/mul-grid-bug.md`。
