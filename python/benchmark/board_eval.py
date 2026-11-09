"""Board-side evaluation: run one or more RKNN models, dump outputs + timing.

Designed to run on RK3588 with rknn-toolkit-lite2. Usage example:
  LD_LIBRARY_PATH=/home/cat/opencl_shim:$LD_LIBRARY_PATH \
  python board_eval.py --models m1.rknn m2.rknn --images a.jpg b.jpg --loops 50 \
      --bench-image a.jpg --out /home/cat/rknn2.5/logs/board_eval.npz
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
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas


def parse(outs):
    logits = boxes = None
    for o in outs:
        sq = np.squeeze(o)
        if sq.shape == (300, 10):
            logits = sq.astype(np.float32)
        elif sq.shape == (300, 4):
            boxes = sq.astype(np.float32)
    return logits, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--loops', type=int, default=50)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--bench-image', default=None)
    ap.add_argument('--out', default='/home/cat/rknn2.5/logs/board_eval.npz')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {'paths': np.array(args.images)}

    for mi, model_path in enumerate(args.models):
        tag = os.path.splitext(os.path.basename(model_path))[0]
        print(f'\n===== {tag} =====', flush=True)
        rk = RKNNLite()
        if rk.load_rknn(model_path) != 0:
            print(f'[{tag}] load failed', flush=True)
            continue
        if rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            print(f'[{tag}] init failed', flush=True)
            rk.release()
            continue

        for i, path in enumerate(args.images):
            img = cv2.imread(path)
            if img is None:
                print(f'[{tag}] cannot read {path}', flush=True)
                continue
            canvas = letterbox(img)
            x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]
            outs = rk.inference(inputs=[x])
            logits, boxes = parse(outs)
            if logits is None or boxes is None:
                print(f'[{tag}] output parse failed: {[o.shape for o in outs]}', flush=True)
                continue
            payload[f'logits_{mi}_{i}'] = logits
            payload[f'boxes_{mi}_{i}'] = boxes
            print(f'[{tag}] [{i}] {os.path.basename(path)} score_max={logits.max():.4f}', flush=True)

        if args.bench_image:
            img = cv2.imread(args.bench_image)
            h, w = img.shape[:2]
            bench = cv2.cvtColor(letterbox(img), cv2.COLOR_BGR2RGB)[None]
            for _ in range(args.warmup):
                rk.inference(inputs=[bench])
            pre, inf, post = [], [], []
            for _ in range(args.loops):
                t0 = time.perf_counter()
                canvas = letterbox(img)
                xx = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]
                t1 = time.perf_counter()
                oo = rk.inference(inputs=[xx])
                t2 = time.perf_counter()
                lg, bx = parse(oo)
                if lg is not None:
                    m = lg.max(-1) > 0.45
                    _ = bx[m]
                t3 = time.perf_counter()
                pre.append((t1 - t0) * 1e3)
                inf.append((t2 - t1) * 1e3)
                post.append((t3 - t2) * 1e3)
            pre = np.array(pre); inf = np.array(inf); post = np.array(post)
            total = pre + inf + post
            timing = np.array([
                pre.mean(), inf.mean(), post.mean(), total.mean(),
                np.percentile(inf, 50), np.percentile(inf, 95),
                1000.0 / total.mean(),
            ])
            payload[f'timing_{mi}'] = timing
            print(
                f'[{tag}] pre {pre.mean():.2f} | inference {inf.mean():.2f} '
                f'(p50 {np.percentile(inf, 50):.2f}, p95 {np.percentile(inf, 95):.2f}) | '
                f'post {post.mean():.2f} | total {total.mean():.2f} ms | '
                f'{1000.0 / total.mean():.2f} FPS',
                flush=True,
            )

        payload[f'models_{mi}'] = np.array([model_path])
        rk.release()

    payload['model_list'] = np.array(args.models)
    np.savez_compressed(args.out, **payload)
    print('\nsaved', args.out, flush=True)


if __name__ == '__main__':
    main()
