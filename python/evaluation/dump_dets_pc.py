"""Dump VisDrone detections from the PyTorch or ONNX model (PC side).

Same preprocessing / decoding as python/evaluation/dump_dets_board.py, so the
AP numbers are directly comparable across PT / ONNX / RKNN.

  python dump_dets_pc.py --backend torch --weights best.pt --images "dir/*.jpg"
  python dump_dets_pc.py --backend onnx  --onnx model.onnx --images "dir/*.jpg"
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

IMGSZ = 640


def letterbox(img, size=IMGSZ):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    canvas = np.zeros((size, size, 3), np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale, left, top


def flatten(obj):
    if isinstance(obj, (list, tuple)):
        for o in obj:
            yield from flatten(o)
    else:
        yield obj


def to_y(out):
    """Return the concatenated (N, 4+classes) decoder output."""
    items = [np.squeeze(t.detach().cpu().numpy()) if hasattr(t, 'detach')
             else np.squeeze(np.asarray(t)) for t in flatten(out)]
    for a in items:
        if a.ndim == 2 and a.shape[-1] == 14:
            return a.astype(np.float32)
    raise RuntimeError(f'no y output, got {[a.shape for a in items]}')


def write_dets(f, stem, y, w, h, scale, left, top, conf):
    scores = y[:, 4:]
    best = scores.max(axis=1)
    cls = scores.argmax(axis=1)
    n = 0
    for q in np.nonzero(best >= conf)[0]:
        cx, cy, bw, bh = y[q, :4]
        x1, x2 = (cx - bw / 2) * IMGSZ, (cx + bw / 2) * IMGSZ
        y1, y2 = (cy - bh / 2) * IMGSZ, (cy + bh / 2) * IMGSZ
        x1 = min(max((x1 - left) / scale, 0), w)
        x2 = min(max((x2 - left) / scale, 0), w)
        y1 = min(max((y1 - top) / scale, 0), h)
        y2 = min(max((y2 - top) / scale, 0), h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        f.write(f'{stem} {int(cls[q])} {best[q]:.5f} '
                f'{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n')
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['torch', 'onnx'], required=True)
    ap.add_argument('--weights', default='/mnt/d/learn/rknn2.5/rt-detr-rknn/model/best.pt')
    ap.add_argument('--onnx')
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    files = []
    for pattern in args.images:
        files.extend(sorted(glob.glob(pattern)))
    files = sorted(set(files))
    print(f'{len(files)} images', flush=True)

    if args.backend == 'torch':
        src = os.environ.get('UAVDETR_SRC', '/mnt/d/learn/detr/UAV-DETR-rt')
        sys.path.insert(0, src)
        import torch
        from ultralytics import RTDETR
        model = RTDETR(args.weights).model.cpu().eval()
        if hasattr(model, 'head'):
            model.head.export = True
            model.head.format = 'onnx'

        def forward(x):
            with torch.no_grad():
                return to_y(model(torch.from_numpy(x)[None]))
    else:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
        name = sess.get_inputs()[0].name

        def forward(x):
            return to_y(sess.run(None, {name: x[None]}))

    total = 0
    with open(args.out, 'w') as f:
        for i, path in enumerate(files):
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            canvas, scale, left, top = letterbox(img)
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
            y = forward(x)
            stem = os.path.splitext(os.path.basename(path))[0]
            total += write_dets(f, stem, y, w, h, scale, left, top, args.conf)
            if (i + 1) % 50 == 0:
                print(f'  {i + 1}/{len(files)}  dets={total}', flush=True)
    print(f'wrote {args.out} ({total} detections)', flush=True)


if __name__ == '__main__':
    main()
