# RT-DETR RK3588 部署 v4.0.0

RK3588 上的 RT-DETR 部署工程：FP16 + 原生 GridSample（GPU）+ C++ 多实例流水线。

**单帧 128 ms，3 实例绑核 18.1 FPS，4 实例 18.66 FPS，精度与 PyTorch 一致。**

## 核心结果

| 指标 | 数值 |
|---|---|
| 模型 | `rtdetr_native_tpopt2_mulnpu_fp16.rknn`（FP16，44.4 MB，md5 `33832CF0…`） |
| 单帧延迟 | **128.4 ms**（`--latency` best，mean 134.3） |
| 3 实例各绑一核 | **18.06 FPS** |
| 4 实例（3 核） | **18.66 FPS** |
| 视频（cars.mp4，3 实例） | **17.65 FPS**，解码仅 4.3 ms/帧 |
| 精度（conf≥0.45，11 图） | recall 100% / precision 100% / meanIoU 0.9912 |
| 检出 | 220/220，与 PyTorch FP32 一致 |

相比 v2 的 Python 实现（7.62 FPS / 131 ms），吞吐提升 **2.4×**，单帧延迟不变。

## v4 的改进

### 1. 算子层（沿用 v2 的成果）

| 改动 | 收益 |
|---|---|
| GridSample 走 GPU（`op_target`） | 替代 Gather 展开，335 ms → 186 ms |
| decoder Mul 从 CPU 移到 NPU | -24.9 ms（去掉每次 4 ms 的跨设备同步） |
| 重写 deformable attention，消掉 12 宽 transpose | -13.1 ms |
| value 保持 channel-first，消掉布局往返 | -16.6 ms |

输出逐位不变（PyTorch 侧 max diff = 0）。

### 2. 工程层（v4 新增）

| 改动 | 收益 |
|---|---|
| C++ 三级流水线（pre → NPU → post，有界队列 + 背压） | 消除流水线气泡 |
| 每 worker 独立 context，**一实例绑一核** | 7.96 → 18.06 FPS |
| RGA 预处理（letterbox + BGR→RGB） | 解码/预处理不再占关键路径 |
| `rknn_set_io_mem` + NEON 自转换 | 去掉 6.8 ms/帧的通用输入转换 |
| 视频输入（OpenCV/FFmpeg 软解 + 读取线程） | 支持 mp4，仅 -1.3% 吞吐 |

## 目录

```text
.
├── cpp/
│   ├── include/          types / pipeline / rknn_detector / rga_utils / drm_alloc
│   ├── src/              对应实现 + main
│   ├── tests/            DRM 探测、RGA 自测、偏移写入验证
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
│   └── roadmap.md        v4.1 计划
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

# 视频
./build/rtdetr_pipeline -m model.rknn -v cars.mp4 -n 3 -p 2 -P 2 --cores split
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `-m, --model` | 必填 | RKNN 模型路径 |
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

见 [`docs/model.md`](docs/model.md)。关键三步：

```bash
python python/export/export_transpose_opt.py --weights best.pt --value-nchw \
    --output model/rtdetr_native_tpopt2.onnx

python python/quantize/convert_native_gpu.py \
    model/rtdetr_native_tpopt2.onnx model/rtdetr_native_tpopt2_mulnpu_fp16.rknn \
    --opt-level 0 --disable-rules squeeze_nd_mul,fuse_mul_into_gemm \
    --op-target-map 'GridSample=gpu,Mul=npu'
```

## 精度验证

```bash
python python/evaluation/dump_ref.py --images /path/to/images/*.jpg --out ref.npz
python python/evaluation/compare_eval.py --ref ref.npz --board board_eval.npz
```

Python 推理入口保留在 `python/runtime/infer.py`（300 query，用于回退或与 C++ 对照）。

## 已知限制

- **仅 FP16**：GridSample 的 GPU 实现不接受 INT8；模型 44.4 MB
- **仅 RK3588**：RK356X 官方不支持 GridSample
- **依赖 OpenCL**：需要导出 `clImportMemoryARM` 的 Mali 库
- **视频为软解**：解码 4~6 ms/帧，未接 MPP（当前不是瓶颈）
- **长稳未验证**：目前只有 200~300 帧级压测

## 文档

- [架构与数据通路](docs/architecture.md)
- [基准数据](docs/benchmarks.md)
- [模型与编译](docs/model.md)
- [常见问题](docs/troubleshooting.md)
- [路线图（v4.1 混合量化）](docs/roadmap.md)

## 致谢

流水线与硬件加速思路参考 [rt-detr-rk3588-deploy](https://github.com/zepher-kk/rt-detr-rk3588-deploy)（v3）。
本版本在 RKNN 2.5 原生 GridSample 的基础上重写了模型与推理路径，两者模型与性能口径不同，不可直接混比。
