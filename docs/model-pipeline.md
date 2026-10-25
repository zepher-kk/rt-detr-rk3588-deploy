# 模型工作流

## 流程

```text
训练权重
  -> ONNX 导出
  -> RKNN 转换
  -> 精度评估
  -> 目标平台验证
```

## 当前模型结构

当前 C++ 部署路径使用：

- 640×640 输入；
- 150 queries；
- 2 层 decoder；
- RKNN 格式。

C++ 运行时同时支持 150 和 300 query 输出。Python 验证脚本当前只适配
300 query，因此不能直接运行当前 150 query 模型。

## ONNX 导出

```bash
python python/export/export_uavdetr_onnx.py \
  --repo /path/to/model-repository \
  --checkpoint /path/to/model.pt \
  --output /path/to/model.onnx \
  --imgsz 640 \
  --num-queries 150 \
  --decoder-layers 2
```

导出脚本会替换不利于端侧转换的动态算子。具体参数见
[导出说明](../python/export/README.md)。

## RKNN 转换

```bash
python python/quantization/convert_rknn_variant.py \
  --onnx /path/to/model.onnx \
  --dataset /path/to/calibration.txt \
  --output /path/to/model.rknn
```

当前生产路径采用 auto-hybrid 和 per-channel 量化。具体参数应根据目标
runtime 与模型验证结果确定。

## 评估

转换后至少完成：

1. 单图输出与参考模型对比；
2. 批量图片 mAP、AP50 和 AP75；
3. 不同置信度下的 Precision、Recall 和 F1；
4. 目标平台视频吞吐与资源占用；
5. 摄像头格式协商和实际画面检查。

仅检查转换命令返回成功不足以确认模型可用。
