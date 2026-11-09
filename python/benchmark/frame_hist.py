"""Histogram of per-frame inference times: find whether slow frames are periodic."""
import sys
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite

model = sys.argv[1]
loops = int(sys.argv[2]) if len(sys.argv) > 2 else 300
image = sys.argv[3] if len(sys.argv) > 3 else '/home/cat/rknn2.5/test_images/uav.jpg'


def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    canvas = np.zeros((size, size, 3), np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas


img = cv2.imread(image)
bench = cv2.cvtColor(letterbox(img), cv2.COLOR_BGR2RGB)[None]

rk = RKNNLite()
assert rk.load_rknn(model) == 0
assert rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
for _ in range(10):
    rk.inference(inputs=[bench])

ts = []
for _ in range(loops):
    t0 = time.perf_counter()
    rk.inference(inputs=[bench])
    ts.append((time.perf_counter() - t0) * 1e3)
ts = np.array(ts)

print(f'model={model.split("/")[-1]} loops={loops}')
print(f'mean {ts.mean():.2f} p50 {np.percentile(ts,50):.2f} p90 {np.percentile(ts,90):.2f} '
      f'p95 {np.percentile(ts,95):.2f} p99 {np.percentile(ts,99):.2f} max {ts.max():.2f}')
for th in (170, 180, 190):
    print(f'  frames > {th}ms: {(ts > th).sum()} ({(ts > th).mean()*100:.1f}%)')
slow = np.where(ts > 175)[0]
print('slow frame indices:', slow[:40].tolist())
if len(slow) > 1:
    print('gaps between slow frames:', np.diff(slow)[:30].tolist())
print('first 40 frames:', [round(x, 1) for x in ts[:40]])
