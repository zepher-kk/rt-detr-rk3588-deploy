# 模型与工具链

## 1. 产物

| 文件 | 说明 |
|---|---|
| `rtdetr_tpopt2_4d_mulnpu.rknn` | 部署模型，FP16，46.6 MB |
| 原生输入 | FP16 NHWC `1×640×640×3`，2,457,600 字节 |
| 输出 | `y = 300×(4+10)`（boxes + sigmoid 分数），另有 4 个辅助输出 |

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

## 4. 约束

| 项 | 限制 |
|---|---|
| 精度 | 仅 FP16（GridSample GPU 不接受 INT8） |
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

## 6. 为什么不用 INT8

GridSample 的 GPU 实现只接受 FP16。换 INT8 必须回退到 Gather 展开（V2 路线），
实测 255 ms / 3.91 FPS，得不偿失。backbone 级混合量化见 `docs/roadmap.md`。
