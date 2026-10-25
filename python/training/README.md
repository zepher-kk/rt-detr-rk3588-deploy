# 微调工具

简体中文 | [English](README.en.md)

## `finetune_uavdetr.py`

从现有权重启动限量微调。

```bash
python python/training/finetune_uavdetr.py \
  --repo /path/to/model-repository \
  --checkpoint /path/to/model.pt \
  --data /path/to/data.yaml \
  --project /path/to/runs \
  --name experiment
```

脚本关闭 mosaic、mixup、copy-paste 和在线验证，以缩短单次实验时间。训练需要
PyTorch、Ultralytics 和 UAV-DETR 源码。

微调后的模型必须重新导出、量化并完成完整评估，不能直接替换部署模型。
