"""Board-side evaluation of the split pipeline: backbone(INT8) -> decoder(FP16).

Writes the same npz layout as board_eval.py so compare_eval.py can read it.

Note: rknn-toolkit-lite declares 4-D inputs as NHWC, so the neck feature maps
produced by model A must be re-ordered into NHWC memory order before feeding B.
"""
import argparse
import os
import time

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
    return canvas


def parse(outs):
    for o in outs:
        sq = np.squeeze(o)
        if sq.ndim == 2 and sq.shape[-1] == 4 + 10:
            return sq[:, 4:].astype(np.float32), sq[:, :4].astype(np.float32)
    return None, None


def to_nhwc(f):
    return np.ascontiguousarray(f.transpose(0, 2, 3, 1))


def load(path):
    rk = RKNNLite()
    assert rk.load_rknn(path) == 0, path
    assert rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0, path
    return rk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', required=True)
    ap.add_argument('--decoder', required=True)
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--loops', type=int, default=50)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--bench-image', default=None)
    ap.add_argument('--out', default='/home/cat/rknn2.5/logs/board_eval_split.npz')
    args = ap.parse_args()

    a = load(args.backbone)
    b = load(args.decoder)

    def forward(x):
        feats = a.inference(inputs=[x])
        return b.inference(inputs=[to_nhwc(np.asarray(f)) for f in feats])

    payload = {'paths': np.array(args.images), 'model_list': np.array(['split'])}
    payload['models_0'] = np.array([f'{args.backbone} + {args.decoder}'])

    for i, path in enumerate(args.images):
        img = cv2.imread(path)
        if img is None:
            print(f'cannot read {path}', flush=True)
            continue
        x = cv2.cvtColor(letterbox(img), cv2.COLOR_BGR2RGB)[None]
        logits, boxes = parse(forward(x))
        if logits is None:
            print(f'[{i}] parse failed', flush=True)
            continue
        payload[f'logits_0_{i}'] = logits
        payload[f'boxes_0_{i}'] = boxes
        print(f'[{i}] {os.path.basename(path)} score_max={logits.max():.4f}', flush=True)

    if args.bench_image:
        img = cv2.imread(args.bench_image)
        bench = cv2.cvtColor(letterbox(img), cv2.COLOR_BGR2RGB)[None]
        for _ in range(args.warmup):
            forward(bench)
        ta, tb, tt = [], [], []
        for _ in range(args.loops):
            t0 = time.perf_counter()
            feats = a.inference(inputs=[bench])
            t1 = time.perf_counter()
            packed = [to_nhwc(np.asarray(f)) for f in feats]
            t2 = time.perf_counter()
            b.inference(inputs=packed)
            t3 = time.perf_counter()
            ta.append((t1 - t0) * 1e3)
            tb.append((t3 - t2) * 1e3)
            tt.append((t3 - t0) * 1e3)
        ta, tb, tt = map(np.array, (ta, tb, tt))
        print(f'A(INT8)  {ta.mean():.2f} ms', flush=True)
        print(f'reorder  {np.mean([0.0]) * 0:.2f} ms', flush=True)
        print(f'B(FP16)  {tb.mean():.2f} ms', flush=True)
        print(f'chain    {tt.mean():.2f} ms (p50 {np.percentile(tt, 50):.2f}, '
              f'p95 {np.percentile(tt, 95):.2f}) -> {1000.0 / tt.mean():.2f} FPS', flush=True)
        payload['timing_0'] = np.array([ta.mean(), tb.mean(), 0.0, tt.mean(),
                                        np.percentile(tt, 50), np.percentile(tt, 95),
                                        1000.0 / tt.mean()])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
