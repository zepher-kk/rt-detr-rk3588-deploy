"""Dump VisDrone detections from RKNN models on the board.

Writes `<stem> <cls> <score> <x1> <y1> <x2> <y2>` in original image pixels,
which is what python/evaluation/visdrone_map.py consumes.

Single model:
  python dump_dets_board.py --model m.rknn --images dir/*.jpg --out det.txt
Split model (backbone INT8 + decoder FP16):
  python dump_dets_board.py --backbone a.rknn --decoder b.rknn ...
"""
import argparse
import glob
import os

import cv2
import numpy as np

from rknnlite.api import RKNNLite


def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    canvas = np.zeros((size, size, 3), np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale, left, top


def parse_y(outs):
    for o in outs:
        sq = np.squeeze(o)
        if sq.ndim == 2 and sq.shape[-1] == 4 + 10:
            return sq.astype(np.float32)
    raise RuntimeError('concatenated y output not found')


def load(path):
    rk = RKNNLite()
    assert rk.load_rknn(path) == 0, path
    assert rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0, path
    return rk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--backbone')
    ap.add_argument('--decoder')
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.model:
        rk = load(args.model)
        run = lambda x: parse_y(rk.inference(inputs=[x]))
    else:
        a = load(args.backbone)
        b = load(args.decoder)

        def run(x):
            feats = a.inference(inputs=[x])
            packed = [np.ascontiguousarray(np.asarray(f).transpose(0, 2, 3, 1))
                      for f in feats]
            return parse_y(b.inference(inputs=packed))

    files = []
    for pattern in args.images:
        files.extend(sorted(glob.glob(pattern)))
    files = sorted(set(files))
    print(f'{len(files)} images', flush=True)

    n_det = 0
    with open(args.out, 'w') as f:
        for i, path in enumerate(files):
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            canvas, scale, left, top = letterbox(img)
            x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]
            y = run(x)
            scores = y[:, 4:]
            best = scores.max(axis=1)
            cls = scores.argmax(axis=1)
            boxes = y[:, :4]
            stem = os.path.splitext(os.path.basename(path))[0]
            for q in np.nonzero(best >= args.conf)[0]:
                cx, cy, bw, bh = boxes[q]
                x1 = (cx - bw / 2) * 640
                y1 = (cy - bh / 2) * 640
                x2 = (cx + bw / 2) * 640
                y2 = (cy + bh / 2) * 640
                x1 = min(max((x1 - left) / scale, 0), w)
                x2 = min(max((x2 - left) / scale, 0), w)
                y1 = min(max((y1 - top) / scale, 0), h)
                y2 = min(max((y2 - top) / scale, 0), h)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                f.write(f'{stem} {int(cls[q])} {best[q]:.5f} '
                        f'{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n')
                n_det += 1
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{len(files)}', flush=True)
    print(f'wrote {args.out} ({n_det} detections)', flush=True)


if __name__ == '__main__':
    main()
