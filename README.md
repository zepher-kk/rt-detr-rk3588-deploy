![Framework](https://img.shields.io/badge/Framework-RT--DETR-orange.svg)
![Language](https://img.shields.io/badge/Language-C++%20%7C%20Python-green.svg)
![Platform](https://img.shields.io/badge/Platform-RK3588-blue.svg)

# RT-DETR RK3588 部署

简体中文 | [English](README.en.md)

## 核心成果

**FPS 上限突破 25 FPS，基本解决 INT8 量化的精度损失问题，实现模型体积、精度与性能的多重同步优化。**

| 当前模型 | 指标 |
|---|---:|
| 模型体积 | 28.43 MB |
| VisDrone val AP | 0.2049 |
| AP50 | 0.3556 |
| AP75 | 0.2024 |
| 1080p 推理性能 | 25.04 ± 0.09 FPS（`-n 10`） |

以上为当前模型及参考测试口径下的结果，实际性能会随运行环境变化。

### 模型格式对比

| 格式 | 模型体积 | AP | AP50 | AP75 | 1080p FPS |
|---|---:|---:|---:|---:|---:|
| 原始 PT | 241.95 MB | 0.3060* | 0.5010* | — | — |
| FP32 ONNX | 80.85 MB | 0.2180 | 0.3725 | 0.2208 | — |
| FP16 RKNN | 55.68 MB | 0.2096 | 0.3659 | 0.2081 | 10.02 |
| **部署 INT8（150 query）** | **28.43 MB** | **0.2049** | **0.3556** | **0.2024** | **25.04 ± 0.09** |

> `*` PT 的 AP 数据为原始论文参考口径，不与下列端到端结果直接混比。
> FP32 ONNX、FP16 RKNN 与部署 INT8 使用同一套后处理与精度评测口径。

相对 FP16，部署 INT8 的体积减少 **48.9%**，AP 保留 **97.8%**，FPS 提升
**149.9%**；相对 FP32 ONNX，体积减少 **64.8%**，AP 保留 **94.0%**。相对
原始 PT，模型体积减少 **88.3%**。

本项目提供 RK3588 上的 RT-DETR 目标检测源码，支持图片、视频文件和 V4L2
摄像头输入。C++ 主流程使用 RGA、RKNN/NPU、GStreamer 和 RK MPP，兼顾硬件
加速与零拷贝数据通路。

仓库只提供源码，不附带编译产物、模型、数据集或宿主机配置。构建命令写在
本文和专项文档中，由使用者根据目标环境完成构建。

初次使用或希望查看原有分步说明，可阅读[小白向补充文档](README-beginner.md)。

## 当前支持

| 项目 | 状态 |
|---|---|
| 目标平台 | RK3588 |
| C++ 推理 | 支持 150 和 300 query 的 RT-DETR 输出 |
| Python 推理 | 当前只支持 300 query，暂不兼容 150 query |
| 当前部署结构 | 640 输入、150 queries、2 层 decoder |
| 模型格式 | RKNN |

Python 推理脚本保留原实现。使用 150 query 模型时，请使用 C++ 主程序；Python
路径将在完成对应适配和验证后再更新。

## 功能

- V4L2 摄像头、视频和图片输入
- GStreamer 与 RK MPP 硬件编解码
- RGA 预处理与 RKNN 零拷贝推理
- 多 NPU worker 与核心掩码
- 全局置信度阈值和按类阈值
- 批量图片推理和检测结果导出
- 实时显示、帧率覆盖和自适应画框
- 异常退出、模型加载失败保护和队列限界

## 数据通路

```text
camera/video/image
  -> GStreamer/V4L2
  -> RGA resize + RGB conversion
  -> RKNN/NPU
  -> NEON postprocess
  -> display or hardware encode
```

摄像头格式在运行时协商。设备节点、分辨率、pixel format、stride 和吞吐由
实际硬件与连接方式决定。模块说明见 [系统架构](docs/architecture.md)。

## 目录

```text
.
├── cpp/                    C++ 源码和测试源码
├── python/                 模型导出、量化、评估与训练工具
├── docs/                   架构、模型、指标和排障文档
├── img/                    示例输入
└── img_results/            示例输出
```

## 环境要求

构建 C++ 程序需要：

- aarch64 C++17 工具链
- CMake 3.10+
- OpenCV
- GStreamer 1.0 与 rockchipmpp
- `librga`、libdrm 和 V4L2
- RKNN Runtime 头文件与库

Python 模型工具需要：

- PyTorch 与 Ultralytics
- ONNX 与 ONNX Runtime
- RKNN-Toolkit2
- Pillow（数据集转换）

具体版本应按照目标板端 runtime 和模型要求选择。

## 构建

```bash
cd cpp
cmake -S . -B build -DRTDETR_INPUT_SIZE=640
cmake --build build -j
```

默认输入尺寸为 640。模型输入尺寸、编译尺寸和预处理尺寸必须一致。

板端或交叉编译所需的库路径通过 `CMakeLists.txt` 中的 CMake 变量配置。
仓库不提供预编译二进制。

## 运行

### 单张图片

```bash
./build/rtdetr_pipeline \
  -m /path/to/model.rknn \
  -i ../img/uav.jpg \
  -o result.jpg \
  -c -0.53f
```

### 视频文件

```bash
./build/rtdetr_pipeline \
  -m /path/to/model.rknn \
  -v /path/to/video.mp4 \
  -o result.mp4 \
  -p 2 -n 10 -P 3 -c -0.53f
```

### 摄像头

```bash
./build/rtdetr_pipeline \
  -m /path/to/model.rknn \
  -d /dev/video0 \
  -W 1280 -H 720 \
  -c -0.53f
```

设备节点和分辨率按实际硬件填写。程序会查询摄像头支持的格式并完成协商。

### 实时显示

```bash
./build/rtdetr_pipeline \
  -m /path/to/model.rknn \
  -v /path/to/video.mp4 \
  --display
```

按 `q` 或 `Esc` 退出。无图形环境时，显示模块应降级而不中断检测。

### 批量图片

```bash
./build/rtdetr_pipeline \
  -m /path/to/model.rknn \
  --image-list image_list.txt \
  --dump-det detections.txt \
  --profile recall \
  -c -0.13f
```

检测导出格式：

```text
<stem> <class_id> <score> <x1> <y1> <x2> <y2>
```

完整参数见 `./build/rtdetr_pipeline -h`。

## 置信度

模型输出未归一化，阈值可使用负数。

- `-c -0.13`：保留较多高分检测。
- `-c -0.53`：提高召回，适合当前实时模型。
- `--profile recall`：使用预设的按类阈值。
- `--conf-class`：只覆盖指定类别。

示例：

```bash
--conf-class "0:-0.50,3:-0.15,9:-0.50"
```

不指定按类配置时，全部类别使用全局 `-c`。

## 模型工具

最新模型下载：[123 云盘 - rtdetr_i8.rknn](https://1856090596.share.123pan.cn/123pan/AyyRvd-SsyZv?pwd=XFT1#)

模型工具按流程拆分在 `python/` 下：

- [工具总览](python/README.md)
- [ONNX 导出](python/export/README.md)
- [RKNN 量化](python/quantization/README.md)
- [精度评估](python/evaluation/README.md)
- [数据集转换](python/dataset/README.md)
- [模型工作流](docs/model-pipeline.md)

流程如下：

```text
checkpoint
  -> ONNX export
  -> RKNN conversion
  -> accuracy evaluation
  -> target validation
```

模型文件、校准数据、训练权重和宿主机备份不进入仓库。

## 参考指标

目标模型的参考结果：

| 指标 | 参考值 |
|---|---:|
| VisDrone val AP | 0.2049 |
| VisDrone val AP50 | 0.3556 |
| 1080p 推理吞吐 | 约 25 FPS |

指标会随运行环境、构建参数和模型文件变化。复测时应记录同口径命令与配置，
但不依赖固定的宿主机环境。

## 已知限制

- Python 推理入口暂无 150 query 适配。
- 模型输入尺寸必须与编译和预处理尺寸一致。
- 模型输出语义需要通过图片、视频和批量评估共同验证。
- 摄像头格式与吞吐取决于实际硬件，不能沿用其他设备的固定结论。
- 模型文件和 RKNN runtime 必须保持版本兼容。

更多说明见 [常见问题](docs/troubleshooting.md)。

## 资料

- [CSDN 项目发起者的博客主页](https://blog.csdn.net/weixin_65585850?type=blog)
- [RT-DETR 部署文章](https://blog.csdn.net/weixin_65585850/article/details/158509230?spm=1001.2014.3001.5502)
- [RK3588 模型转换文章](https://blog.csdn.net/weixin_65585850/article/details/158575468?spm=1001.2014.3001.5502)
- [模型与演示文件](https://drive.google.com/drive/folders/1zS1hq9Hf5GdOKOKwz2djPmIrUm8cQLn_?usp=sharing)
- [Rockchip RKNN Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)
- [Rockchip 官网](https://www.rock-chips.com/)
