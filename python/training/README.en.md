# Fine-Tuning

English | [简体中文](README.md)

## `finetune_uavdetr.py`

Starts a bounded fine-tuning run from an existing checkpoint.

```bash
python python/training/finetune_uavdetr.py \
  --repo /path/to/model-repository \
  --checkpoint /path/to/model.pt \
  --data /path/to/data.yaml \
  --project /path/to/runs \
  --name experiment
```

The script disables mosaic, mixup, copy-paste, and online validation to keep
each experiment short. Training requires PyTorch, Ultralytics, and the
UAV-DETR source tree.

Fine-tuned models must be exported, quantized, and fully re-evaluated before
they can replace the deployment model.
