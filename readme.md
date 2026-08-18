# 🚀 RT-DETR on RK3588: 高性能边缘部署实战

[![Platform: RK3588](https://img.shields.io/badge/Platform-RK3588-blue.svg)](https://www.rock-chips.com/)
[![Framework: RT-DETR](https://img.shields.io/badge/Framework-RT--DETR-orange.svg)]()
[![Language: C++/Python](https://img.shields.io/badge/Language-C++%20%7C%20Python-green.svg)]()

本项目致力于将先进的 **RT-DETR (Real-Time DEtection TRansformer)** 算法部署到 **RK3588** 边缘计算平台上。

凭借 RT-DETR **“去 NMS (NMS-free)”** 的架构优势，结合我们专门为 RK3588 设计的 **C++ 异步多线程流水线**，完美解决了传统目标检测算法在边缘端后处理耗时过长的痛点，最大化压榨 NPU 算力！

**算子优化请参考博客：[Matmul#2](https://blog.csdn.net/weixin_65585850/article/details/158509230?spm=1001.2014.3001.5502), [grid_sample](https://blog.csdn.net/weixin_65585850/article/details/158575468?spm=1001.2014.3001.5502)**
---

## 🛠️ 1. 环境配置 (Environment Setup)

本项目包含模型转换与板端推理两部分，需要分别在 PC 端和 RK3588 板端配置环境：

* **PC 端 (Ubuntu x86) - 用于模型转换**:
    * Python 3.8+
    * [RKNN-Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) (推荐 v2.0 以上)
    * ONNX (用于加载和验证导出模型)
* **RK3588 板端 (Ubuntu/Debian) - 用于模型推理**:
    * **Python 推理**: `rknn-toolkit-lite2` (与 PC 端版本对齐), `OpenCV-Python`
    * **C++ 推理**: 
        * GCC / G++
        * CMake (>= 3.4)
        * OpenCV (C++ 版，用于图像预处理/画框)
        * RKNPU2 驱动 (`librknnrt.so`)
        * librga / libdrm / libv4l2（V3 起 RGA_DMA 零拷贝通路依赖）
        * GStreamer 1.x + rockchipmpp 插件（V3 起 MPP 硬件编解码依赖，`gst_io`）

---

## 🔄 2. 模型转换 (Model Conversion)
**[模型下载](https://drive.google.com/drive/folders/1zS1hq9Hf5GdOKOKwz2djPmIrUm8cQLn_?usp=sharing),包含了pt，onnx，rknn模型**

RT-DETR 采用一对一匈牙利匹配，**直接输出 300 个预测框**。
> ⚠️ **注意**：导出 ONNX 时请务必使用 **静态 Shape (Static Shape)**，并去掉所有后处理 NMS 节点，保持最纯粹的张量输出（例如 `[1, 300, 14]`）。

**转换步骤**：
在部署了 `RKNN-Toolkit2` 的 PC 上，编写并运行转换脚本（支持 INT8 量化或 FP16）：
```bash
# 示例：通过 Toolkit2 将 onnx 转为 rknn
python convert.py <path_to_onnx> rk3588 fp16 <output_name.rknn>
python
测试单张图片（安静出图，不测速）：
Bash
python infer.py --model_path rtdetr.rknn   test.jpg --conf_thres 0.5
测试本地视频（狂飙出图，动态打印 FPS）：
Bash
python infer.py --model_path rtdetr.rknn --source demo.mp4 --img_size 640
接入 USB 摄像头实时测试：
Bash
python infer.py --model_path rtdetr.rknn --source 0
```
🐍 3. Python 推理 (Python Inference)
我们提供了高度工程化的 Python 推理脚本，支持 单图、本地视频流、USB 摄像头，并内置动态 FPS 测速。

运行方式：

```bash
cd python/

# 1. 单张图片测试 (安静出图)
python infer.py --model_path ../model/best.rknn --source ../img/uav.jpg --conf_thres 0.45

# 2. 本地视频动态测速
python infer.py --model_path ../model/best.rknn --source ../test.mp4

# 3. USB 摄像头实时推理 (注意替换实际的 /dev/video 节点)
python infer.py --model_path ../model/best.rknn --source 21 
检测结果（图片或视频）将自动保存在当前目录下。
```
⚡ 4. C++ 推理 (C++ Inference)
为了彻底解决 Python 单线程下 NPU 等待 CPU 前后处理的性能瓶颈，我们用 C++ 编写了三段式异步流水线 (PipelineManager)。
系统将任务解耦为：洗菜工(前处理) -> 主厨(NPU多核并行) -> 洗碗工(后处理与渲染)，大幅提升吞吐量。

编译流程：
```bash
cd cpp/
mkdir build && cd build
cmake ..
make -j4
# 可执行文件生成于 build/ 下：./rtdetr_pipeline（make install 会安装到 bin/）
```

> ⚠️ **V3 起使用方式已重构**：可执行文件更名为 `rtdetr_pipeline`，命令行参数全面变更（`-i` 图片 / `-v` 视频 / `-d` 摄像头 / `-G` 日志级别等）。V2 的 `rknn_rtdetr_demo` 与 `-s` / `-l` / `--pre` 等参数已废弃，**请以文末 V3 章节的用法为准**。

运行方式（完整示例与参数说明见文末 V3 章节）：
```bash
# 1. 单张图片检测（JPEG 走 MPP 硬解，失败自动回退 OpenCV）
./rtdetr_pipeline -m ../model/best.rknn -i ../img/uav.jpg -o result.jpg -c -0.13f

# 2. 视频推理（GStreamer + MPP 硬解，输出 H.264 硬编）
./rtdetr_pipeline -m ../model/best.rknn -v ../test.mp4 -o result.mp4 -p 2 -n 8 -P 3 -c -0.13f

# 3. 查看帮助菜单
./rtdetr_pipeline -h
```
📊 5. 效果展示 (Results)
🎯 检测精度与效果


RT-DETR 在不依赖 NMS 的情况下，依然能精准区分紧密贴合的物体，彻底告别 YOLO 常见的“重叠漏检”问题。

🚀 性能飞跃 (Python vs C++ 多线程)
(在这里插入你的终端 FPS 截图)

Python 串行推理：约 2.67X FPS (主要受限于 OpenCV CPU 前处理与 GIL 锁)。

C++ 异步流水线：约 7X FPS (3个 NPU 核心满载，性能提升数倍！)。

💡 6. 小结 (Summary)
为什么选择 RT-DETR？ 在 RK3588 这种边缘芯片上，NPU 的张量运算极快，但 CPU 相对较弱。RT-DETR 免去了极其消耗 CPU 算力的 NMS 后处理，完美契合了“全流程 NPU 加速”的理念。

多线程的魅力：通过 C++ SafeQueue 解耦的流水线，使得读图、预处理、推理、解码各司其职，有效消灭了硬件闲置期。

下一步优化 (TODO)：
- 🔴 **NPU 推理耗时瓶颈（最高优先级，任务 6）**：INT8 模型单次推理约 500~560ms/帧，是当前整体 FPS 的主要瓶颈，待专项优化；
- 预处理/后处理进一步提速（任务 5，暂缓，待 NPU 瓶颈处理后评估恢复）；
- RGA 源缓冲 stride 对齐根治（建议改用 4 字节像素源缓冲，消除 3 字节非整除对齐的 CPU 回退路径）；
- 相机 DMA→DMA 通路 stride 隐患复测与加固；
- 板端 GStreamer 无 RGA 插件，NV12→BGR 可探索接入 librga `imcvtcolor`，进一步卸载 CPU。

👉 更详细的技术内幕与原理解析，请移步博客：[你的CSDN或个人博客链接](https://blog.csdn.net/weixin_65585850?type=blog)

------

### 🎉 V2 版本重大更新 (v2.0 Release Notes)

我们持续在榨干 RK3588 的性能，V2 版本带来了以下关键修复与升级：

- 🛠️ **模型导出与后处理重构** 

- 全面修复并优化了 `export.py` 的 ONNX 导出逻辑，同时同步重写了配套的张量后处理代码，数据流转更加严谨（具体实现请查阅最新源码）。

- 🎯 **INT8 量化精度修复 & 权重开源** 

- 成功攻克了 RT-DETR 在 INT8 (`i8`) 量化转换时精度断崖式下降的痛点(FPS=12)！为了方便大家复现，我们现已开源经过验证的 `best.pt` 原始权重，强烈建议大家下载体验、自行转换或作为您自己项目的 Baseline 参考。

- 🚀 **视频流推理指令优化 (日志净化)** 

- **关于警告的说明**：在处理视频流时，由于 NPU 处理极边缘目标坐标时的半精度误差，底层驱动会抛出 `GatherElements` 越界警告。但这已被底层驱动自动修正，**绝对不影响最终的画框精度与程序稳定性**。 

- 为了保持终端测速日志的绝对纯净（眼不见为净），请在运行视频压测时加上 `grep -v` 魔法后缀来过滤底层警告：

  ```python
  ./rknn_rtdetr_demo -m /home/cat/project/rknn/rt-detr-rknn/model/rtdetr_i8.rknn -s /home/cat/project/rknn/rt-detr-rknn/img/cars.mp4 2>&1 | grep -v "GatherElements"
  ```

------

### 🎉 V3 版本重大更新 (v3.0 Release Notes: RGA_DMA 零拷贝 + GStreamer/MPP 全链路硬件加速)

> ⚠️ **重要**：V3 相对 V2 是**结构性重构**，不是简单增量——可执行文件、命令行参数与整条数据通路均已变更（V2 的 `rknn_rtdetr_demo`、`-s`、`-l`、`--pre` 等用法已废弃）。**cpp/ 的使用方式请以本文档 V3 章节为准。**

#### V3 vs V2 主要差异

| 维度 | V2 (`rknn_rtdetr_demo`) | V3 (`rtdetr_pipeline`) |
|------|------------------------|------------------------|
| 可执行文件 | `rknn_rtdetr_demo` | `rtdetr_pipeline` |
| 输入方式 | `-s` 图片/视频 | `-i` 图片 / `-v` 视频 / `-d` V4L2 摄像头 |
| 视频编解码 | OpenCV/FFmpeg 软解 + `mp4v` 软编 | GStreamer + RK MPP 硬解（H.264/H.265）/ H.264 硬编 |
| 预处理 | OpenCV CPU `resize+cvtColor` | RGA 硬件加速（DMA→DMA / virt→DMA）+ stride 安全回退 |
| 后处理 | 纯 C++ 解码 | NEON SIMD 加速 |
| 日志 | 散落打印 | 模块化级联 logger（`-G` 0~8） |
| 新增能力 | - | 图片模式、NPU 多核绑定、实时显示、帧率覆盖、VFR 两遍法、健壮性熔断 |

#### 核心更新内容

- 🧩 **端到端零拷贝数据通路 (Camera → RGA → NPU)**：`DmaBufferPool`（DRM dumb buffer + PRIME fd + mmap，预分配循环复用）打通 V4L2 摄像头、RGA 预处理与 NPU 推理：
  - `V4l2ZeroCopyCapture`：V4L2 MMAP + EXPBUF 零拷贝采集；
  - `RgaPreprocessor`：RGA 硬件缩放 + BGR→RGB（DMA→DMA / virt→DMA 双路径，非对齐 stride 自动安全回退 CPU，杜绝逐行偏移污染）；
  - `RKNNDetector::infer_zero_copy`：`rknn_create_mem_from_fd` + `rknn_set_io_mem` 零拷贝推理。
- 🎬 **GStreamer + RK MPP 硬解/硬编（新增 `gst_io` 模块）**：
  - 视频读取 `GstVideoReader`：qtdemux/avidemux/matroskademux/tsdemux → `mppvideodec` H.264/H.265 硬解，NV12 直拉 + OpenCV NEON `cvtColor` 转 BGR（规避 `videoconvert` ~60ms/帧瓶颈）；图片 JPEG 走 `mppjpegdec` 硬解，PNG/BMP/WebP 软解；
  - 视频输出 `GstVideoWriter`：appsrc(BGR) → `mpph264enc` H.264 硬编 → mp4mux，替换 OpenCV `mp4v` 软编；
  - 全部保留 OpenCV 回退（GStreamer 不可用时自动降级软解/软编）。
- 🎞️ **编码质量与 VFR 修复**：appsrc 显式 `colorimetry=bt709`（对比度 std 41.3→47.5 完全恢复）+ 默认 **fixqp(qp22) + High profile** 恒质量编码；VFR 视频两遍法实测帧率（特殊 480×332@60 输出 26fps/21.31s，时长与源一致），CFR 视频零额外开销。
- 🔧 **正确性与健壮性**：1080p 顶部绿条根治（`GstVideoMeta` 真实 offset/stride 组装 + `arm-afbc=false`）；RGA 并发调用互斥串行化（rga_fail=0）；NPU 初始化失败熔断 + `wait_idle()` 30s 上限 + 队列安全 shutdown，杜绝挂起。
- 🛠️ **图片检测模式与日志系统**：新增 `-i/--image` 单图检测（JPEG 硬解优先）；`-G/--debug` 模块化级联日志（0=仅错误+报告，8=全部）。
- 🖥️ **实时检测画面显示（`--display`）**：专用显示线程 + 丢旧保新队列播放检测帧，按 `q`/`ESC` 优雅退出；无显示环境（headless）自动降级，不中断检测。
- ⏱️ **输入/输出帧率覆盖（`-F/--fps`）**：视频文件输入侧按指定速率限速喂帧、输出侧按指定 fps 写容器；摄像头采集与输出同步生效；未指定时自动使用源帧率。
- 📷 **USB 摄像头多格式支持**：V4L2 自动协商 BGR3/RGB3/YUYV，YUYV 帧走 RGA/CPU 转换回退；`read_frame` 带超时等待与 buffer 生命周期加固，支持优雅退出。
- 🎨 **自适应画框与固定调色板**：线宽/字号随输入分辨率线性缩放并限幅，小目标只画细框不画文字；颜色按类别固定（确定性调色板），便于观察与调试。
- ⚙️ **推理输出预分配**：RKNN 输出张量在 `init` 时按元素数预分配，推理 `is_prealloc` 复用，避免每帧动态分配。
- ⚡ **NPU 多核绑定**：`rknn_set_core_mask` + `--npu-cores auto|0|1|2|0,1|0,1,2`，配合多线程流水线按核心分配推理线程。
- 📦 **构建依赖**：CMake 新增 `gstreamer-1.0 / gstreamer-app-1.0 / gstreamer-video-1.0` 依赖（含 rockchipmpp 插件），其余同 V3 基础（C++17、-O3/OpenMP/NEON、`librga/libdrm/libv4l2`、`RK3588_TOOLCHAIN` 交叉编译）。

#### 性能概述（板端 performance 模式实测）

| 场景 | FPS | 平均 CPU | 峰值 RSS | 说明 |
|------|-----|----------|----------|------|
| 默认参数 `-p 2 -n 3 -P 1` | 8.55 | - | - | 基线 1（cars.mp4 720p@30） |
| 最佳性能 `-p 2 -n 14 -P 3` | **~16.0** | ~394% | ~1650 MB | NPU 3 核满载（基线 2） |
| 甜点位 `-p 2 -n 8 -P 3` | 15.61 | 365% | 953 MB | CPU -15% / 内存 -36% / FPS 仅 -3% |

修复后全链路实测（`-p 2 -n 8 -P 3 -c -0.13f`）：特殊 480×332@60 → **16.23 FPS / 321% / 827MB**；cars.mp4 720p → **15.81 FPS / 332% / 954MB**；1080p → **15.37 FPS / 370% / 1152MB**。图片模式 uav.jpg 单图（含模型加载）约 **293ms**，检出 38 目标，与 V2 原版一致。单元测试 **29/29**（含 VFR 帧率估算、1080p 无绿条、模型加载失败不挂起、YUYV 转换、自适应画框、丢旧保新队列等回归用例）。

#### V3 快速上手（以 V3 为准）

```bash
cd cpp/
mkdir -p build && cd build
cmake .. && make -j4

# 1. 单张图片检测（JPEG 走 MPP 硬解，失败自动回退 OpenCV）
./rtdetr_pipeline -m rtdetr_r18.rknn -i uav.jpg -o result.jpg -c -0.13f

# 2. 视频文件：MPP 硬解 + H.264 硬编 [最佳性能]
./rtdetr_pipeline -m rtdetr_r18.rknn -v test.mp4 -o result.mp4 -p 2 -n 14 -P 3 -c -0.13f

# 3. 视频文件 [CPU/内存甜点位]
./rtdetr_pipeline -m rtdetr_r18.rknn -v test.mp4 -o result.mp4 -p 2 -n 8 -P 3 -c -0.13f

# 4. V4L2 摄像头零拷贝实时推理
./rtdetr_pipeline -m rtdetr_r18.rknn -d /dev/video0 -W 1920 -H 1080

# 5. 实时显示检测画面（按 q/ESC 退出；headless 自动降级）
./rtdetr_pipeline -m rtdetr_r18.rknn -v test.mp4 --display

# 6. 覆盖输入/输出帧率（视频文件限速喂帧 + 输出 fps；摄像头同样生效）
./rtdetr_pipeline -m rtdetr_r18.rknn -v test.mp4 -F 25 -o result.mp4 -p 2 -n 8 -P 3 -c -0.13f

# 7. 仅性能测试 + 精简日志（-G 0 只输出错误与运行报告）
./rtdetr_pipeline -m rtdetr_r18.rknn -v test.mp4 -G 0

# 8. 查看完整帮助
./rtdetr_pipeline -h
```

> 模型输出未归一化，`-c` 阈值可取 > -1（如 `-0.13f`）；`-i` 与 `-v` 互斥；更多参数见 `./rtdetr_pipeline -h`。
