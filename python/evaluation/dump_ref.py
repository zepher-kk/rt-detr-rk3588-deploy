"""Dump PyTorch (FP32) reference outputs for RT-DETR under the same letterbox preprocessing.

Run on PC (WSL rk25 env). Output .npz is later compared with board RKNN outputs.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

DETR_SRC = os.environ.get('UAVDETR_SRC', '/mnt/d/learn/detr/UAV-DETR-rt')


def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    canvas = np.zeros((size, size, 3), np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, (scale, left, top)


def _flatten(obj):
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(_flatten(v))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            out.extend(_flatten(v))
        return out
    return []


def to_2d(out):
    """Normalize outputs to (N,4) boxes and (N,10) scores.

    Prefer the concatenated decoder output y = (N, 4+classes), where the score
    columns are already sigmoid probabilities. Fall back to the first matching
    (N,4)/(N,classes) tensors.
    """
    items = [np.squeeze(t.detach().cpu().numpy()) for t in _flatten(out)]
    for arr in items:
        if arr.ndim == 2 and arr.shape[-1] == 4 + 10:
            return arr[:, 4:], arr[:, :4]
    boxes = logits = None
    for arr in items:
        if boxes is None and arr.ndim == 2 and arr.shape[-1] == 4:
            boxes = arr
        elif logits is None and arr.ndim == 2 and arr.shape[-1] == 10:
            logits = arr
    return logits, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='/mnt/d/learn/rknn2.5/rt-detr-rknn/model/best.pt')
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--num-queries', type=int, default=300)
    ap.add_argument('--decoder-layers', type=int, default=None)
    ap.add_argument('--out', default='/mnt/d/learn/rknn2.5/logs/ref_outputs.npz')
    args = ap.parse_args()

    sys.path.insert(0, DETR_SRC)
    from ultralytics import RTDETR

    print('loading', args.weights, flush=True)
    wrapper = RTDETR(args.weights)
    model = wrapper.model.cpu().eval()

    # same pruning as the export script
    decoder = model.model[-1]
    decoder.num_queries = args.num_queries
    if args.decoder_layers is not None:
        decoder.decoder.layers = nn.ModuleList(
            list(decoder.decoder.layers)[:args.decoder_layers])
        decoder.dec_score_head = nn.ModuleList(
            list(decoder.dec_score_head)[:args.decoder_layers])
        decoder.dec_bbox_head = nn.ModuleList(
            list(decoder.dec_bbox_head)[:args.decoder_layers])
        decoder.decoder.num_layers = args.decoder_layers
        decoder.decoder.eval_idx = args.decoder_layers - 1
    print(f'pruned: num_queries={args.num_queries} layers={args.decoder_layers}', flush=True)

    if hasattr(model, 'head'):
        model.head.export = True
        model.head.format = 'onnx'

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {'paths': np.array(args.images)}
    for i, path in enumerate(args.images):
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f'cannot read {path}')
        canvas, meta = letterbox(img, args.imgsz)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            out = model(x)
        logits, boxes = to_2d(out)
        if logits is None or boxes is None:
            raise SystemExit(f'unexpected output shapes for {path}: {type(out)}')
        payload[f'logits_{i}'] = logits.astype(np.float32)
        payload[f'boxes_{i}'] = boxes.astype(np.float32)
        payload[f'meta_{i}'] = np.array(meta, dtype=np.float64)
        print(f'[{i}] {os.path.basename(path)} score_max={logits.max():.4f}', flush=True)

    np.savez_compressed(args.out, **payload)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
