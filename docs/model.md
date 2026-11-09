# 模型与工具链

## 1. 产物

| 文件 | 说明 |
|---|---|
| `rtdetr_native_tpopt2_mulnpu_fp16.rknn` | 部署模型，FP16，44.4 MB |
| md5 | `33832CF00996A221C25DFE77C02D144B` |
| 原生输入 | FP16 NHWC `1×640×640×3`，2,457,600 字节 |
| 输出 | 300×4 boxes + 300×10 logits（另有 3 个辅助输出） |

模型不在仓库内，按下面步骤自行生成。

## 2. 导出 ONNX（PC / WSL）

```bash
export UAVDETR_SRC=/path/to/UAV-DETR-rt      # ultralytics 源码目录

python python/export/export_transpose_opt.py \
    --weights best.pt \
    --value-nchw \
    --output model/rtdetr_native_tpopt2.onnx
```

`--value-nchw` 会同时应用两处等价重写：

1. deformable attention 的 `stack → flatten(-2)` 改为逐 level 累加
2. `value` 保持 channel-first，沿序列轴切片

两处都不改变数学结果（PyTorch 侧 max diff = 0）。

## 3. 编译 RKNN

```bash
python python/quantize/convert_native_gpu.py \
    model/rtdetr_native_tpopt2.onnx \
    model/rtdetr_native_tpopt2_mulnpu_fp16.rknn \
    --opt-level 0 \
    --disable-rules squeeze_nd_mul,fuse_mul_into_gemm \
    --op-target-map 'GridSample=gpu,Mul=npu'
```

参数说明：

| 参数 | 必须 | 原因 |
|---|---|---|
| `GridSample=gpu` | 是 | 默认 auto 会分到 CPU，报 `Unsupport CPU op` |
| `Mul=npu` | 是 | 否则 6 次 decoder Mul 落在 CPU，每次约 4 ms 同步开销 |
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
