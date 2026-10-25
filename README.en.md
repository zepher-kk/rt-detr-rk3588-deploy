![Framework](https://img.shields.io/badge/Framework-RT--DETR-orange.svg)
![Language](https://img.shields.io/badge/Language-C++%20%7C%20Python-green.svg)
![Platform](https://img.shields.io/badge/Platform-RK3588-blue.svg)

# RT-DETR Deployment on RK3588

English | [简体中文](README.md)

## Key Results

**Throughput exceeds 25 FPS. The accuracy loss caused by INT8 quantization is
largely resolved, enabling simultaneous improvements in model size, accuracy,
and performance.**

| Current model | Metric |
|---|---:|
| Model size | 28.43 MB |
| VisDrone val AP | 0.2049 |
| AP50 | 0.3556 |
| AP75 | 0.2024 |
| 1080p inference throughput | 25.04 ± 0.09 FPS (`-n 10`) |

These values correspond to the current model and the stated reference test
configuration. Actual performance depends on the runtime environment.

### Model Format Comparison

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

This project provides RT-DETR object-detection source code for RK3588, with
support for images, video files, and V4L2 camera input. The primary C++ pipeline
uses RGA, RKNN/NPU, GStreamer, and RK MPP for hardware acceleration and
zero-copy data transfer.

This repository contains source code only. It does not include build artifacts,
models, datasets, or host-machine configurations. Build instructions are
provided in this document and the related guides.

For the beginner-oriented walkthrough, see the
[supplementary guide](README-beginner.en.md).

## Current Support

| Item | Status |
|---|---|
| Target platform | RK3588 |
| C++ inference | Supports 150- and 300-query RT-DETR outputs |
| Python inference | Supports 300-query output only; currently incompatible with 150-query models |
| Current deployment architecture | 640 input, 150 queries, 2 decoder layers |
| Model format | RKNN |

The Python inference script retains its original implementation. Use the C++
entry point for 150-query models. The Python path will be updated after the
required compatibility work has been completed and verified.

## Features

- V4L2 camera, video, and image input
- GStreamer and RK MPP hardware decoding and encoding
- RGA preprocessing and zero-copy RKNN inference
- Multi-worker NPU execution and NPU core masks
- Global confidence thresholds and class-wise thresholds
- Batch image inference and detection export
- Live display, frame-rate override, and adaptive drawing
- Graceful shutdown, model-load protection, and bounded queues

## Data Path

```text
camera/video/image
  -> GStreamer/V4L2
  -> RGA resize + RGB conversion
  -> RKNN/NPU
  -> NEON postprocess
  -> display or hardware encode
```

Camera formats are negotiated at runtime. The device node, resolution, pixel
format, stride, and throughput depend on the connected hardware. See
[System Architecture](docs/architecture.md) for module details.

## Directory Layout

```text
.
├── cpp/                    C++ source and tests
├── python/                 Model export, quantization, evaluation, and training tools
├── docs/                   Architecture, model, benchmark, and troubleshooting guides
├── img/                    Sample inputs
└── img_results/            Sample outputs
```

## Requirements

Building the C++ application requires:

- An aarch64 C++17 toolchain
- CMake 3.10 or later
- OpenCV
- GStreamer 1.0 and rockchipmpp
- `librga`, libdrm, and V4L2
- RKNN Runtime headers and libraries

The Python model tools require:

- PyTorch and Ultralytics
- ONNX and ONNX Runtime
- RKNN-Toolkit2
- Pillow for dataset conversion

Select versions compatible with the target board runtime and model.

## Build

```bash
cd cpp
cmake -S . -B build -DRTDETR_INPUT_SIZE=640
cmake --build build -j
```

The default input size is 640. The model input size, compile-time input size,
and preprocessing size must match.

Library paths for on-board or cross-compilation builds are configured through
the CMake variables in `CMakeLists.txt`. The repository does not provide
prebuilt binaries.

## Usage

### Single Image

```bash
./build/rtdetr_pipeline \
  -m /path/to/model.rknn \
  -i ../img/uav.jpg \
  -o result.jpg \
  -c -0.53f
```

### Video File

```bash
./build/rtdetr_pipeline \
  -m /path/to/model.rknn \
  -v /path/to/video.mp4 \
  -o result.mp4 \
  -p 2 -n 10 -P 3 -c -0.53f
```

### Camera

```bash
./build/rtdetr_pipeline \
  -m /path/to/model.rknn \
  -d /dev/video0 \
  -W 1280 -H 720 \
  -c -0.53f
```

Use the device node and resolution appropriate for the connected hardware. The
application queries and negotiates the formats supported by the camera.

### Live Display

```bash
./build/rtdetr_pipeline \
  -m /path/to/model.rknn \
  -v /path/to/video.mp4 \
  --display
```

Press `q` or `Esc` to exit. When no graphical environment is available, the
display module should degrade without interrupting inference.

### Batch Images

```bash
./build/rtdetr_pipeline \
  -m /path/to/model.rknn \
  --image-list image_list.txt \
  --dump-det detections.txt \
  --profile recall \
  -c -0.13f
```

Detection export format:

```text
<stem> <class_id> <score> <x1> <y1> <x2> <y2>
```

Run `./build/rtdetr_pipeline -h` for the full option list.

## Confidence Thresholds

Model outputs are not normalized, so negative thresholds are valid.

- `-c -0.13`: Retains more high-confidence detections.
- `-c -0.53`: Increases recall and is suitable for the current real-time model.
- `--profile recall`: Uses the preset class-wise thresholds.
- `--conf-class`: Overrides thresholds for selected classes only.

Example:

```bash
--conf-class "0:-0.50,3:-0.15,9:-0.50"
```

When no class-wise configuration is provided, every class uses the global `-c`
threshold.

## Model Tools

Latest model download: [123 Cloud Drive](https://1856090596.share.123pan.cn/123pan/AyyRvd-SsyZv?pwd=XFT1#)

Model tools are organized by workflow stage under `python/`:

- [Tool overview](python/README.en.md)
- [ONNX export](python/export/README.en.md)
- [RKNN quantization](python/quantization/README.en.md)
- [Accuracy evaluation](python/evaluation/README.en.md)
- [Dataset conversion](python/dataset/README.en.md)
- [Model workflow](docs/model-pipeline.md)

Workflow:

```text
checkpoint
  -> ONNX export
  -> RKNN conversion
  -> accuracy evaluation
  -> target validation
```

Model files, calibration data, training weights, and host backups are not
included in the repository.

## Reference Metrics

| Metric | Reference value |
|---|---:|
| VisDrone val AP | 0.2049 |
| VisDrone val AP50 | 0.3556 |
| 1080p inference throughput | Approximately 25 FPS |

Metrics vary with the runtime environment, build configuration, and model
artifact. Revalidation should use a consistent test procedure and configuration
without depending on a fixed host environment.

## Known Limitations

- The Python inference entry point does not yet support 150-query models.
- The model input size must match the compile-time and preprocessing sizes.
- Model output semantics must be validated with images, video, and batch
  evaluation.
- Camera formats and throughput depend on the actual hardware and must not be
  inferred from another device.
- The model file and RKNN runtime must remain version-compatible.

See [Troubleshooting](docs/troubleshooting.md) for additional guidance.

## Resources

- [CSDN project blog](https://blog.csdn.net/weixin_65585850?type=blog)
- [RT-DETR deployment article](https://blog.csdn.net/weixin_65585850/article/details/158509230?spm=1001.2014.3001.5502)
- [RK3588 model-conversion article](https://blog.csdn.net/weixin_65585850/article/details/158575468?spm=1001.2014.3001.5502)
- [Models and demonstration files](https://drive.google.com/drive/folders/1zS1hq9Hf5GdOKOKwz2djPmIrUm8cQLn_?usp=sharing)
- [Rockchip RKNN Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)
- [Rockchip](https://www.rock-chips.com/)
