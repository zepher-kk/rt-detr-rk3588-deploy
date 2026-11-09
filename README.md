# RT-DETR RK3588 部署 v4.1

RK3588 上的 RT-DETR 部署工程：原生 GridSample（GPU）+ C++ 多实例流水线。

一套代码、两种部署配置：

| 配置 | 形态 | AP (VisDrone val 548) | 单帧 | 3/4 实例 | 模型体积 |
|---|---|---|---|---|---|
| **FP16（v4.0.1）** | 单模型 | **0.1985** | 131 ms | 17.99 / 18.53 FPS | 46.6 MB |
| **INT8 拆分（v4.1）** | backbone+neck INT8 + decoder FP16 | 0.1882 | **69.8 ms**（e2e） | **29.9 / 33.3 FPS** | 17.5 + 8.1 MB |

```bash
# FP16 单模型
./rtdetr_pipeline -m rtdetr_tpopt2_4d_mulnpu.rknn -i uav.jpg -n 3 -p 2 -P 2 --cores split

# INT8 拆分（backbone + decoder）
./rtdetr_pipeline -m rtdetr_v41_backbone_int8.rknn \
                  --decoder rtdetr_v41_decoder_fp16.rknn \
                  -i uav.jpg -n 4 -p 2 -P 2 --cores split
```

## 核心结果（FP16 单模型）

| 指标 | 数值 |
|---|---|
| 模型 | `rtdetr_tpopt2_4d_mulnpu.rknn`（FP16，46.6 MB，无 INT8） |
| 单帧推理 | **131.0 ms**（p50，150 次）；p95 133.1 ms |
| 端到端时延 | mean **120.2 ms**，best **116.8 ms**（`--latency`） |
| 3 实例各绑一核 | **17.99 FPS** |
| 4 实例（3 核） | **18.53 FPS** |
| 视频（cars.mp4，3 实例） | **17.63 FPS**，解码仅 3.3 ms/帧 |
| 精度（AP，548 图） | AP **0.1985** / AP50 0.3582 / AP75 0.1872 |

## 核心结果（v4.1 INT8 拆分）

| 指标 | 数值 |
|---|---|
| 模型 | `rtdetr_v41_backbone_int8.rknn` + `rtdetr_v41_decoder_fp16.rknn` |
| 精度（AP，548 图） | AP **0.1882** / AP50 0.3359 / AP75 0.1800 |
| 单帧 e2e | **69.8 ms**（best 67.4，`--latency`） |
| 3 worker × 1 核 | **29.87 FPS**（npu 100.3 ms/帧） |
| 4 worker（超订 3 核） | **33.32 FPS**（npu 119.3 ms/帧） |
| 与 Python 链路一致性 | 548 图 conf≥0.45 检出 25,304 vs 25,282（差 0.09%） |

> 精度口径：同一套 `python/evaluation/visdrone_map.py` + 同一份 val 标注。
> v3.0.5 的历史 AP 来自另一套前后处理，**不可直接比较**（见 docs/benchmarks.md）。

> **v4.0.1 修复**：v4.0.0 里 RKNN 2.5 会把 decoder 的 6 维广播 `Mul`
> 编译成**全 0**（采样点塌到参考框中心），导致 decoder 输出错误；
> 旧 README 的 18.06 FPS / recall 100% 是在这个有缺陷的模型上测的。
> 现在把网格运算改成 4 维，decoder 正确，速度只掉 0.9 ms。
> 详见 [6 维 Mul 缺陷](docs/mul-grid-bug.md)。

## v4 的改进

### 1. 算子层（沿用 v2 的成果）

| 改动 | 收益 |
|---|---|
| GridSample 走 GPU（`op_target`） | 替代 Gather 展开，335 ms → 186 ms |
| 网格运算改 4 维（v4.0.1） | 修掉 6 维 Mul 被编译成 0 的缺陷；Mul 留在 NPU，比放 CPU 省 26 ms |
| 重写 deformable attention，消掉 12 宽 transpose | -13.1 ms |
| value 保持 channel-first，消掉布局往返 | -16.6 ms |

ONNX 输出与 PyTorch 逐位相同；板上 decoder 与 PyTorch 高置信 query 差 < 1e-3。

### 2. 工程层（v4 新增）

| 改动 | 收益 |
|---|---|
| C++ 三级流水线（pre → NPU → post，有界队列 + 背压） | 消除流水线气泡 |
| 每 worker 独立 context，**一实例绑一核** | 7.96 → 18.0 FPS |
| RGA 预处理（letterbox + BGR→RGB） | 解码/预处理不再占关键路径 |
| `rknn_set_io_mem` + NEON 自转换 | 去掉 6.8 ms/帧的通用输入转换 |
| 视频输入（OpenCV/FFmpeg 软解 + 读取线程） | 支持 mp4，仅 -1.3% 吞吐 |

### 3. v4.1：拆分模型 + INT8

| 改动 | 收益 |
|---|---|
| backbone+neck 单独 INT8（w8a8, per-channel） | 该段算力降到 1/4 |
| decoder 单独 FP16（GridSample 仍走 GPU） | 保留 GPU 采样路径 |
| 双 context 同核 + 数据并行（3/4 worker） | 22.63 → 29.87 / 33.32 FPS |
| 输入 uint8 直连（`^0x80` 转 int8）+ 输出 `want_float=0` | 去掉两次格式转换 |
| 反量化 + NCHW→NC1HWC2 融合交接 | 2.1 MB 读 + 4.3 MB 写，<1 ms |
| `custom_string` 配对校验 | 防止 backbone/decoder 版本错配 |

## 目录

```text
.
├── cpp/
│   ├── include/          types / pipeline / rknn_detector / split_detector / rga_utils / drm_alloc
│   ├── src/              对应实现 + main
│   ├── tests/            DRM 探测、RGA 自测、属性探针
│   └── CMakeLists.txt
├── python/
│   ├── export/           ONNX 导出（含注意力重写）
│   ├── quantize/         ONNX → RKNN
│   ├── runtime/          Python 推理入口（回退/对照，300 query）
│   ├── benchmark/        板端基准、多实例、帧分布
│   ├── evaluation/       PyTorch 参考输出与一致性对比
│   └── deploy/           上传、启动器
├── docs/
│   ├── architecture.md   流水线与数据通路
│   ├── benchmarks.md     全部实测数据
│   ├── model.md          导出/编译参数与约束
│   ├── troubleshooting.md 常见报错
│   ├── mul-grid-bug.md   6 维 Mul 缺陷与修复
│   └── roadmap.md        裁剪模型与 INT8 评估
└── README.md
```

## 环境要求

板端（RK3588）：

- aarch64 C++17 工具链、CMake 3.10+
- OpenCV 4.x（含 FFmpeg 后端，用于视频）
- `librga`（2.x）、`libdrm`
- RKNN Runtime **2.5.0**（`/usr/lib/librknnrt.so`）
- 导出 `clImportMemoryARM` 的 OpenCL 库（Mali blob）

PC 端（模型转换）：rknn-toolkit2 2.5.0、ONNX、PyTorch、ultralytics。

## 构建

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

> 不要把旧版 `librknnrt.so` 放进 `include/`。工程默认链接
> `/usr/lib/librknnrt.so`（2.5.0）。

## 运行

```bash
# 板端 OpenCL（必需）
export LD_LIBRARY_PATH=/home/cat/opencl_shim:$LD_LIBRARY_PATH

# 单帧时延
./build/rtdetr_pipeline -m model.rknn -i uav.jpg -n 1 -p 1 -P 1 --latency

# 图片列表 + 循环压测（3 实例各绑一核）
./build/rtdetr_pipeline -m model.rknn --image-list images.txt \
    -n 3 -p 2 -P 2 -N 500 --cores split

# v4.1 拆分：backbone(INT8) + decoder(FP16)，4 worker 可到 33 FPS
./build/rtdetr_pipeline \
    -m rtdetr_v41_backbone_int8.rknn \
    --decoder rtdetr_v41_decoder_fp16.rknn \
    --image-list images.txt -n 4 -p 2 -P 2 -N 500 --cores split

# 视频
./build/rtdetr_pipeline -m model.rknn -v cars.mp4 -n 3 -p 2 -P 2 --cores split
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `-m, --model` | 必填 | RKNN 模型路径（拆分模式下为 backbone） |
| `--decoder` | — | 指定后进入拆分模式：`-m` 是 backbone(INT8)，该参数是 decoder(FP16) |
| `-i, --image` | — | 单图 |
| `--image-list` | — | 图片列表（每行一个路径） |
| `-v, --video` | — | 视频文件（软解） |
| `-n, --npu` | 3 | NPU worker 数（每个一个 context） |
| `-p, --pre` | 2 | 预处理线程 |
| `-P, --post` | 1 | 后处理线程 |
| `-q, --queue` | 16 | 每级队列容量 |
| `-N, --frames` | 200 | 测量帧数（0 = 全部） |
| `--warmup` | 20 | 预热帧数 |
| `--cores` | split | `split`（每 worker 一核）/ `all` / `auto` |
| `-c, --conf` | 0.45 | 置信度阈值 |
| `-F, --fps` | 0 | 视频节流帧率（0 = 全速） |
| `--latency` | — | 单帧串行时延模式 |

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `RTDETR_NO_RGA=1` | 未设置 | 关闭 RGA，走 CPU 预处理 |
| `RTDETR_IO_MEM=0` | 未设置 | 关闭 io_mem，回退 `rknn_inputs_set` |
| `RTDETR_DST=dma` | `heap` | 预处理输出到 DRM 缓冲（仅调试） |
| `UAVDETR_SRC` | 内置路径 | 导出/参考脚本用的 ultralytics 源码目录 |

## 模型生成

见 [`docs/model.md`](docs/model.md)。关键步骤：

```bash
# 1) 导出 ONNX（含 4 维网格重写）
python python/export/export_transpose_opt.py --weights best.pt --value-nchw \
    --fold-offset-scale --msda-4d-grid \
    --output model/rtdetr_tpopt2_4d.onnx

# 2a) FP16 单模型
python python/quantize/convert_native_gpu.py \
    model/rtdetr_tpopt2_4d.onnx model/rtdetr_tpopt2_4d_mulnpu.rknn \
    --opt-level 0 --disable-rules squeeze_nd_mul,fuse_mul_into_gemm \
    --op-target-map 'GridSample=gpu,Mul=npu'

# 2b) v4.1 拆分：先切图，再分别编译
python python/export/split_decoder.py model/rtdetr_q150_l2_4d.onnx \
    --feat /model.19/cv3/act/Mul_output_0 /model.22/cv3/act/Mul_output_0 \
           /model.25/cv3/act/Mul_output_0 \
    --out-a model/q150_backbone.onnx --out-b model/q150_decoder.onnx

python python/quantize/convert_int8.py model/q150_backbone.onnx \
    model/rtdetr_v41_backbone_int8.rknn \
    --dataset calib.txt --custom-string v41-q150l2

python python/quantize/convert_native_gpu.py model/q150_decoder.onnx \
    model/rtdetr_v41_decoder_fp16.rknn --opt-level 0 \
    --disable-rules squeeze_nd_mul,fuse_mul_into_gemm \
    --op-target-map 'GridSample=gpu,Mul=npu' \
    --inputs /model.19/cv3/act/Mul_output_0,/model.22/cv3/act/Mul_output_0,/model.25/cv3/act/Mul_output_0 \
    --input-sizes 1x256x80x80\;1x256x40x40\;1x256x20x20 --no-normalize \
    --custom-string v41-q150l2
```

> 不要漏掉 `--msda-4d-grid`：没有它时 6 维 Mul 在 NPU 上会输出全 0，
> 模型能跑、速度也快，但 decoder 结果是错的。

> 拆分模式下两个模型的 `--custom-string` 必须一致，运行时会校验；
> 不一致时 C++ 直接拒绝启动，避免拿错版本静默输出垃圾。

## 精度验证

```bash
python python/evaluation/dump_ref.py --images /path/to/images/*.jpg --out ref.npz
python python/evaluation/compare_eval.py --ref ref.npz --board board_eval.npz
```

Python 推理入口保留在 `python/runtime/infer.py`（300 query，用于回退或与 C++ 对照）。

## 已知限制

- **GridSample 只能 FP16**：decoder 必须保持 FP16（GPU 实现不接受 INT8）。
  FP16 单模型 46.6 MB；v4.1 拆分为 backbone INT8 17.5 MB + decoder FP16 8.1 MB。
- **仅 RK3588**：RK356X 官方不支持 GridSample
- **依赖 OpenCL**：需要导出 `clImportMemoryARM` 的 Mali 库
- **视频为软解**：解码 3~6 ms/帧，未接 MPP（当前不是瓶颈）
- **长稳未验证**：目前只有 200~500 帧级压测

## 文档

- [架构与数据通路](docs/architecture.md)
- [基准数据](docs/benchmarks.md)
- [模型与编译](docs/model.md)
- [常见问题](docs/troubleshooting.md)
- [6 维 Mul 缺陷与修复](docs/mul-grid-bug.md)
- [路线图](docs/roadmap.md)

## 致谢

流水线与硬件加速思路参考 [rt-detr-rk3588-deploy](https://github.com/zepher-kk/rt-detr-rk3588-deploy)（v3）。
本版本在 RKNN 2.5 原生 GridSample 的基础上重写了模型与推理路径，两者模型与性能口径不同，不可直接混比。
