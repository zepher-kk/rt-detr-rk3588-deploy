# 🚀 RT-DETR RK3588 新手指引（v4.1）

> 面向第一次上手的人：从零到跑通第一张图。
> 项目全貌、两代版本对照见 [README.md](README.md)。

---

## 0. 先搞清楚两件事

### 两种部署配置

| 配置 | 用几个模型 | 精度 AP | 速度 | 适合谁 |
|---|---|---|---|---|
| **FP16 单模型** | 1 个 | 0.1985 | 3 实例 17.99 FPS | 想部署简单、精度优先 |
| **INT8 拆分** | 2 个（backbone + decoder） | 0.1882 | 3 实例 29.9 FPS / 4 实例 33.3 FPS | 想跑得更快 |

两份模型**用同一个二进制**，差别只是命令行加不加 `--decoder`。

### 几个名词

| 名词 | 意思 |
|---|---|
| **FP16 / INT8** | 模型权重的数值精度。INT8 更快、更小，但精度略低 |
| **query** | RT-DETR 一次输出多少个候选框（本项目 300 或 150） |
| **AP** | 目标检测的标准精度指标，越高越好（本项目在 VisDrone val 上测） |
| **`--cores split`** | 让每个 worker 独占一个 NPU 核——多实例提速的关键 |
| **backbone / decoder** | 网络前半段（提特征）/ 后半段（出框和分数） |

---

## 1. 准备环境

### 板端（RK3588，必须）

```bash
# 检查运行时版本，必须是 2.5.0
ls -l /usr/lib/librknnrt.so

# 编译依赖
sudo apt install -y build-essential cmake libopencv-dev librga-dev libdrm-dev
```

还需要一个能导出 `clImportMemoryARM` 的 Mali OpenCL 库（系统自带的 ocl-icd 不行）：

```bash
mkdir -p ~/opencl_shim
ln -sf /usr/lib/aarch64-linux-gnu/libmali-x11/libmali-valhall-g610-g13p0-x11-wayland-gbm.so \
       ~/opencl_shim/libOpenCL.so
export LD_LIBRARY_PATH=~/opencl_shim:$LD_LIBRARY_PATH
```

> 每次开新终端都要重新 `export`，或者写进 `~/.bashrc`。

### PC 端（只有“自己转模型”才需要）

- Python 3.8+
- `rknn-toolkit2==2.5.0`（版本必须和板端 `librknnrt` 一致）
- PyTorch + ultralytics（导出 ONNX 用）

---

## 2. 编译（板端，约 2 分钟）

```bash
git clone https://github.com/zepher-kk/rt-detr-rk3588-deploy.git
cd rt-detr-rk3588-deploy/cpp

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

产物：`cpp/build/rtdetr_pipeline`

```bash
# 确认链接的是系统 2.5.0 运行时
ldd build/rtdetr_pipeline | grep rknn
```

---

## 3. 准备模型

模型文件（`.rknn`）**不在仓库里**，需要自己生成或从发布页下载。

### 3a. 生成 FP16 单模型（最简单）

```bash
# 1) 导出 ONNX（注意两个参数都要带）
python python/export/export_transpose_opt.py --weights best.pt --value-nchw \
    --fold-offset-scale --msda-4d-grid --output model/rtdetr_tpopt2_4d.onnx

# 2) 编译 RKNN
python python/quantize/convert_native_gpu.py \
    model/rtdetr_tpopt2_4d.onnx model/rtdetr_tpopt2_4d_mulnpu.rknn \
    --opt-level 0 --disable-rules squeeze_nd_mul,fuse_mul_into_gemm \
    --op-target-map 'GridSample=gpu,Mul=npu'
```

> ⚠️ 千万不要漏掉 `--msda-4d-grid`：没有它模型也能跑、还更快，但 decoder 输出是错的
> （详见 `docs/mul-grid-bug.md`）。

### 3b. 生成 INT8 拆分模型（更快）

```bash
# 1) 在 neck 输出处切成两个 ONNX
python python/export/split_decoder.py model/rtdetr_q150_l2_4d.onnx \
    --feat /model.19/cv3/act/Mul_output_0 /model.22/cv3/act/Mul_output_0 \
           /model.25/cv3/act/Mul_output_0 \
    --out-a model/q150_backbone.onnx --out-b model/q150_decoder.onnx

# 2) backbone：INT8（需要一个校准图片列表 calib.txt，每行一个图片路径）
python python/quantize/convert_int8.py model/q150_backbone.onnx \
    model/rtdetr_v41_backbone_int8.rknn \
    --dataset calib.txt --custom-string v41-q150l2

# 3) decoder：FP16
python python/quantize/convert_native_gpu.py model/q150_decoder.onnx \
    model/rtdetr_v41_decoder_fp16.rknn --opt-level 0 \
    --disable-rules squeeze_nd_mul,fuse_mul_into_gemm \
    --op-target-map 'GridSample=gpu,Mul=npu' \
    --inputs /model.19/cv3/act/Mul_output_0,/model.22/cv3/act/Mul_output_0,/model.25/cv3/act/Mul_output_0 \
    --input-sizes 1x256x80x80\;1x256x40x40\;1x256x20x20 --no-normalize \
    --custom-string v41-q150l2
```

两个模型的 `--custom-string` 必须一样，否则运行时会拒绝启动（这是故意的）。

---

## 4. 跑通第一张图

```bash
export LD_LIBRARY_PATH=~/opencl_shim:$LD_LIBRARY_PATH

# FP16 单模型
./build/rtdetr_pipeline -m model/rtdetr_tpopt2_4d_mulnpu.rknn -i uav.jpg -c 0.45

# INT8 拆分
./build/rtdetr_pipeline \
    -m model/rtdetr_v41_backbone_int8.rknn \
    --decoder model/rtdetr_v41_decoder_fp16.rknn \
    -i uav.jpg -c 0.45
```

正常输出长这样：

```
[result] aggregate 8.07 FPS
[result] pre 3.22 ms | npu 123.86 ms | post 0.01 ms
[result] detections total=55 (avg 55.0/frame), rga=1 cpu_fallback=0
```

`detections` 是置信度 ≥ `-c` 的框数；`rga=1` 说明用了硬件预处理。

### 测单帧延迟

```bash
./build/rtdetr_pipeline -m model/xxx.rknn -i uav.jpg -n 1 -p 1 -P 1 --latency
# [latency] e2e mean 122.0 ms, best 118.9 ms
```

---

## 5. 想跑得更快：多实例 + 绑核

```bash
# 图片列表（每行一个路径）
ls test_images/*.jpg > list.txt

# FP16：3 实例各绑一核
./build/rtdetr_pipeline -m model/rtdetr_tpopt2_4d_mulnpu.rknn \
    --image-list list.txt -n 3 -p 2 -P 2 -N 500 --cores split

# INT8 拆分：4 个 worker（3 核超订）最快
./build/rtdetr_pipeline -m model/rtdetr_v41_backbone_int8.rknn \
    --decoder model/rtdetr_v41_decoder_fp16.rknn \
    --image-list list.txt -n 4 -p 2 -P 2 -N 500 --cores split
```

| 参数 | 建议 |
|---|---|
| `--cores split` | **必须**。不绑核的话多实例几乎不加速 |
| `-n` | FP16 用 3，INT8 拆分用 4 |
| `-p 2 -P 2` | 预处理/后处理各 2 线程，够用 |
| `-N` | 测量帧数；试跑用 100，正式压测用 500+ |

---

## 6. 视频

```bash
./build/rtdetr_pipeline -m model/xxx.rknn -v cars.mp4 \
    -n 3 -p 2 -P 2 --cores split
```

解码走 OpenCV/FFmpeg 软解（约 3 ms/帧），已被流水线掩盖。
需要 `-F 30` 可以按源帧率节流。

---

## 7. 验证精度（可选）

需要一份 VisDrone 验证集和逐图标注 txt。

```bash
# 1) 板端导出检测结果
python python/evaluation/dump_dets_board.py \
    --model model/rtdetr_tpopt2_4d_mulnpu.rknn \
    --images "visdrone_val/*.jpg" --conf 0.05 --out dets.txt

# 2) PC 端算 AP
python python/evaluation/visdrone_map.py \
    --det dets.txt --ann path/to/ann_dir --conf 0.05 --ignore-regions
```

参考值（548 张）：FP16 单模型 AP 0.1985，INT8 拆分 AP 0.1882。

---

## 8. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `Unsupport CPU op: GridSample` | 编译时没指定 GPU | 加 `--op-target-map 'GridSample=gpu'` |
| `Unsupport GPU op: GridSample` | OpenCL 库不对 | 用 Mali blob 建 shim（见 §1） |
| 启动即退出，`model mismatch` | 两个模型 build id 不同 | 用相同 `--custom-string` 重编 |
| 能跑但检出为 0 / 分数很低 | 拆分交接的布局/量化问题 | 加 `RTDETR_SPLIT_DEBUG=1` 看特征范围，见 `docs/troubleshooting.md` §11 |
| 多实例不提速 | 没绑核 | 加 `--cores split` |
| `--latency` 比 `[stage] e2e` 小很多 | 压测 e2e 含排队时间 | 单帧延迟只看 `--latency` |
| 需要摄像头 / 实时显示 / 按类阈值 | v4 尚未迁移 | 用 `v3.0.5` 分支，见 README【新增 C】 |

---

## 9. 目录速查

```
cpp/src/           C++ 推理（split_detector.cc = INT8 拆分）
python/export/     ONNX 导出（export_transpose_opt.py / split_decoder.py）
python/quantize/   RKNN 编译（convert_native_gpu.py / convert_int8.py）
python/evaluation/ 精度评估（dump_dets_*.py / visdrone_map.py）
docs/              架构、基准、模型、排错、路线图
```

---

## 10. 下一步

- 想理解为什么 18 FPS 的旧数据不能用：看 `docs/mul-grid-bug.md`
- 想调优：看 `docs/benchmarks.md` 的分段数据和 `docs/architecture.md`
- 想加摄像头/显示/按类阈值：`docs/roadmap.md` 里有未迁移的能力清单

祝跑通顺利 🎉
