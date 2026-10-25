# RT-DETR on RK3588: High-Performance Edge Deployment

> This is the beginner-oriented supplementary guide. See [README.en.md](README.en.md)
> for the primary project documentation and current support status.

English | [简体中文](README-beginner.md)

[![Platform: RK3588](https://img.shields.io/badge/Platform-RK3588-blue.svg)](https://www.rock-chips.com/)
[![Framework: RT-DETR](https://img.shields.io/badge/Framework-RT--DETR-orange.svg)]()
[![Language: C++/Python](https://img.shields.io/badge/Language-C++%20%7C%20Python-green.svg)]()

## Key Results

**Throughput exceeds 25 FPS. The accuracy loss caused by INT8 quantization is
largely resolved, with simultaneous improvements in model size, accuracy, and
performance.**

| Format | Model size | AP | AP50 | AP75 | 1080p FPS |
|---|---:|---:|---:|---:|---:|
| Original PT | 241.95 MB | 0.3060* | 0.5010* | — | — |
| FP32 ONNX | 80.85 MB | 0.2180 | 0.3725 | 0.2208 | — |
| FP16 RKNN | 55.68 MB | 0.2096 | 0.3659 | 0.2081 | 10.02 |
| **Deployed INT8 (150 query)** | **28.43 MB** | **0.2049** | **0.3556** | **0.2024** | **25.04 ± 0.09** |

> `*` The PT AP values use the original paper's reference protocol and are not
> directly comparable with the end-to-end results below. FP32 ONNX, FP16 RKNN,
> and deployed INT8 use the same postprocessing and accuracy-evaluation protocol.

Compared with FP16, deployed INT8 reduces model size by **48.9%**, retains
**97.8%** of AP, and improves FPS by **149.9%**. Compared with FP32 ONNX, it
reduces model size by **64.8%** and retains **94.0%** of AP. Compared with the
original PT artifact, model size is reduced by **88.3%**.

This project deploys the **RT-DETR (Real-Time DEtection TRansformer)** model on
the **RK3588** edge platform. Its NMS-free architecture avoids the CPU-heavy NMS
stage, while the C++ asynchronous pipeline decouples preprocessing, NPU
inference, and postprocessing.

For operator-level details, see the source blog articles:
[MatMul optimization](https://blog.csdn.net/weixin_65585850/article/details/158509230?spm=1001.2014.3001.5502)
and
[grid_sample optimization](https://blog.csdn.net/weixin_65585850/article/details/158575468?spm=1001.2014.3001.5502).

## 1. Environment Setup

The project includes model conversion and board-side inference. Configuration
depends on whether the step runs on a PC or on RK3588.

### PC: Model Conversion

- Python 3.8 or later
- [RKNN-Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)
- ONNX for model loading and validation

### RK3588 Board: Inference

Python inference:

- `rknn-toolkit-lite2` compatible with the conversion environment
- OpenCV Python

C++ inference:

- GCC/G++
- CMake 3.10 or later
- C++ OpenCV
- RKNPU2 driver and `librknnrt.so`
- `librga`, libdrm, and libv4l2 for the RGA DMA zero-copy path
- GStreamer 1.x and the rockchipmpp plugin for MPP hardware codec support

## 2. Model Conversion

Original model package:
[Google Drive](https://drive.google.com/drive/folders/1zS1hq9Hf5GdOKOKwz2djPmIrUm8cQLn_?usp=sharing).
It contains PT, ONNX, and RKNN artifacts.

Latest model:
[123 Cloud Drive - rtdetr_i8.rknn](https://1856090596.share.123pan.cn/123pan/AyyRvd-SsyZv?pwd=XFT1#).

RT-DETR uses one-to-one Hungarian matching and directly produces prediction
boxes. The current deployment model uses 150 queries. The original Python
inference script expects 300-query output.

> When exporting ONNX, use static shapes and exclude NMS postprocessing from the
> graph. Keep the tensor outputs explicit.

Convert ONNX to RKNN with a Toolkit2 environment:

```bash
python python/legacy/convert.py <path_to_onnx> rk3588 fp16 <output_name.rknn>
```

For the current workflow, prefer the parameterized tools documented in
[python/README.en.md](python/README.en.md).

## 3. Python Inference

The Python entry point supports a single image, a local video, or a camera
source, and reports runtime FPS.

> The current Python inference script only supports 300-query output and is not
> yet compatible with 150-query models. Use the C++ entry point for the deployed
> 150-query model.

Single image:

```bash
python python/runtime/infer.py \
  --model_path model/best.rknn \
  --source img/uav.jpg \
  --conf_thres 0.45
```

Video:

```bash
python python/runtime/infer.py \
  --model_path model/best.rknn \
  --source test.mp4
```

Camera:

```bash
python python/runtime/infer.py \
  --model_path model/best.rknn \
  --source /dev/video0
```

Detection images and videos are saved in the current working directory.

## 4. C++ Inference

The C++ implementation uses a staged asynchronous pipeline:

```text
preprocessing -> multi-core NPU inference -> postprocessing and rendering
```

Build from source:

```bash
cd cpp
mkdir build
cd build
cmake ..
make -j4
```

The executable is generated as `build/rtdetr_pipeline`.

> V3 changed the executable name and command-line interface. Use
> `-i` for images, `-v` for videos, `-d` for V4L2 cameras, and `-G` for log
> levels. V2 arguments such as `-s`, `-l`, and `--pre` are deprecated.

Examples:

```bash
# Image
./rtdetr_pipeline -m ../model/best.rknn -i ../img/uav.jpg -o result.jpg -c -0.13f

# Video
./rtdetr_pipeline -m ../model/best.rknn -v ../test.mp4 -o result.mp4 -p 2 -n 8 -P 3 -c -0.13f

# Help
./rtdetr_pipeline -h
```

## 5. Results

RT-DETR can separate closely adjacent objects without an NMS stage. The C++
pipeline distributes decoding, preprocessing, NPU inference, and
postprocessing across bounded queues and worker threads.

The current deployment model reaches 25.04 ± 0.09 FPS at 1080p under the
reference test configuration. See the format-comparison table above for model
size and accuracy results.

## 6. Summary

RK3588 provides high-throughput NPU tensor computation while its CPU resources
are more limited. RT-DETR's NMS-free design avoids a CPU-heavy postprocessing
stage and maps well to hardware-accelerated inference.

The C++ pipeline separates frame reading, preprocessing, inference, and decoding
with bounded queues, reducing idle time across hardware components.

Current engineering work:

- Continue optimizing the NPU inference path.
- Evaluate additional preprocessing and postprocessing acceleration.
- Validate RGA stride handling for non-aligned source buffers.
- Recheck camera DMA-to-DMA paths.
- Investigate additional color-conversion acceleration when RGA is unavailable.

More implementation notes are available in the
[CSDN project blog](https://blog.csdn.net/weixin_65585850?type=blog).

## V2 Release Notes

V2 introduced the following model and runtime improvements:

- Refactored ONNX export and tensor postprocessing.
- Reduced the INT8 quantization accuracy drop.
- Published validated baseline weights for reproduction.
- Added guidance for video-stream testing and driver log filtering.

## V3 Release Notes

V3 is a structural update rather than a small incremental change. The
executable, command-line interface, and data path changed from V2.

| Area | V2 (`rknn_rtdetr_demo`) | V3 (`rtdetr_pipeline`) |
|---|---|---|
| Executable | `rknn_rtdetr_demo` | `rtdetr_pipeline` |
| Input options | `-s` for images/video | `-i` image, `-v` video, `-d` V4L2 camera |
| Video codec | OpenCV/FFmpeg software decode and `mp4v` encode | GStreamer + RK MPP hardware decode/encode |
| Preprocessing | OpenCV CPU resize and color conversion | RGA acceleration with safe stride fallback |
| Postprocessing | C++ decode | NEON SIMD acceleration |
| Logging | Ad hoc output | Modular logger (`-G 0..8`) |
| Added capabilities | — | Batch images, class-wise thresholds, NPU core masks, live display, frame-rate override, VFR handling, fail-safe shutdown |

### Main V3 Changes

- **Zero-copy path:** DMA buffers, RGA preprocessing, and RKNN input/output
  memory are shared through DRM, PRIME file descriptors, and mapped buffers.
- **GStreamer and RK MPP:** hardware H.264/H.265 decoding and H.264 encoding
  replace software codec paths where supported, with OpenCV fallback.
- **Video timing:** CFR and VFR inputs are distinguished, with additional
  handling for variable-frame-rate output.
- **Image and log modes:** `-i/--image` handles single images; `-G/--debug`
  controls module-level logging.
- **Live display:** `--display` uses a dedicated display thread and a
  drop-oldest queue.
- **Frame-rate override:** `-F/--fps` controls input pacing and output frame
  rate for videos and cameras.
- **Camera formats:** V4L2 negotiates BGR3, RGB3, YUYV, and MJPG where available.
- **Adaptive drawing:** line width and font size scale with image dimensions.
- **Output preallocation:** RKNN output buffers are allocated during
  initialization and reused.
- **Batch image evaluation:** `--image-list` and `--dump-det` export detections
  for mAP evaluation without reloading the model for every image.
- **Class-wise thresholds:** `--profile` and `--conf-class` apply per-class
  confidence thresholds across image, video, and batch modes.
- **NPU core control:** `--npu-cores` selects NPU core masks for worker threads.

### Quick Start

```bash
cd cpp
mkdir -p build
cd build
cmake ..
make -j4

# Image
./rtdetr_pipeline -m rtdetr_r18.rknn -i uav.jpg -o result.jpg -c -0.13f

# Video: hardware decode and H.264 hardware encode
./rtdetr_pipeline -m rtdetr_r18.rknn -v test.mp4 -o result.mp4 -p 2 -n 14 -P 3 -c -0.13f

# Lower CPU and memory configuration
./rtdetr_pipeline -m rtdetr_r18.rknn -v test.mp4 -o result.mp4 -p 2 -n 8 -P 3 -c -0.13f

# V4L2 camera
./rtdetr_pipeline -m rtdetr_r18.rknn -d /dev/video0 -W 1920 -H 1080

# Live display
./rtdetr_pipeline -m rtdetr_r18.rknn -v test.mp4 --display

# Override input/output frame rate
./rtdetr_pipeline -m rtdetr_r18.rknn -v test.mp4 -F 25 -o result.mp4 -p 2 -n 8 -P 3 -c -0.13f

# Performance test with reduced logs
./rtdetr_pipeline -m rtdetr_r18.rknn -v test.mp4 -G 0

# Batch image evaluation
./rtdetr_pipeline -m rtdetr_r18.rknn --image-list val_list.txt --dump-det det.txt --profile recall -c -0.13f

# Class-wise threshold override
./rtdetr_pipeline -m rtdetr_r18.rknn -i uav.jpg -o result.jpg --conf-class "0:-0.40,3:0.00,9:-0.55"
```

Model outputs are not normalized, so thresholds may be negative. `-i` and `-v`
are mutually exclusive. The detection export format is:

```text
<stem> <class_id> <score> <x1> <y1> <x2> <y2>
```
