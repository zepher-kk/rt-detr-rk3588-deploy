"""Interleaved multi-model benchmark on RK3588 (rknn-toolkit-lite2).

Loads every model once, then runs ROUNDS rounds; inside each round every model
runs WARMUP + LOOPS inferences. Model order is rotated each round so slow
thermal drift / frequency changes do not favour one model.

Usage:
  export LD_LIBRARY_PATH=/home/cat/opencl_shim:$LD_LIBRARY_PATH
  python bench_native.py --models a.rknn b.rknn --image uav.jpg \
      --rounds 3 --loops 50 --out bench.json
"""
import argparse
import json
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


def read_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return int(f.read().strip()) / 1000.0
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--loops', type=int, default=50)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--out', default='bench.json')
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit('cannot read ' + args.image)
    bench = cv2.cvtColor(letterbox(img), cv2.COLOR_BGR2RGB)[None]

    runtimes = []
    for path in args.models:
        rk = RKNNLite()
        if rk.load_rknn(path) != 0:
            raise SystemExit('load failed: ' + path)
        if rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            raise SystemExit('init failed: ' + path)
        runtimes.append((path, rk))
        print('loaded', path, flush=True)

    results = {p: [] for p, _ in runtimes}
    temps = {}
    for rnd in range(args.rounds):
        order = runtimes[rnd % len(runtimes):] + runtimes[:rnd % len(runtimes)]
        for path, rk in order:
            for _ in range(args.warmup):
                rk.inference(inputs=[bench])
            samples = []
            for _ in range(args.loops):
                t0 = time.perf_counter()
                rk.inference(inputs=[bench])
                samples.append((time.perf_counter() - t0) * 1e3)
            samples = np.array(samples)
            results[path].append({
                'mean': float(samples.mean()),
                'p50': float(np.percentile(samples, 50)),
                'p95': float(np.percentile(samples, 95)),
                'min': float(samples.min()),
                'max': float(samples.max()),
            })
            temps[f'{path}@r{rnd}'] = read_temp()
            print(f'round {rnd} {path}: mean {samples.mean():.2f} p50 {np.percentile(samples, 50):.2f} '
                  f'p95 {np.percentile(samples, 95):.2f} ms  temp {read_temp()}', flush=True)

    summary = {}
    for path, rows in results.items():
        summary[path] = {
            'mean': float(np.median([r['mean'] for r in rows])),
            'p50': float(np.median([r['p50'] for r in rows])),
            'p95': float(np.median([r['p95'] for r in rows])),
            'fps': float(1000.0 / np.median([r['mean'] for r in rows])),
            'rounds': rows,
        }
    payload = {'image': args.image, 'rounds': args.rounds, 'loops': args.loops,
               'temps': temps, 'summary': summary}
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)

    print()
    print(f'{"model":<46}{"mean":>9}{"p50":>9}{"p95":>9}{"FPS":>8}')
    for path, s in summary.items():
        print(f'{path.split("/")[-1]:<46}{s["mean"]:9.2f}{s["p50"]:9.2f}{s["p95"]:9.2f}{s["fps"]:8.2f}')
    print('saved', args.out)


if __name__ == '__main__':
    main()
