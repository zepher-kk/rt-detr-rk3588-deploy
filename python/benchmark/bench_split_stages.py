"""Time backbone (A) and decoder (B) separately on 1/2/3 NPU cores."""
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite

A = '/home/cat/rknn2.5/model/rtdetr_v41_backbone_int8.rknn'
B = '/home/cat/rknn2.5/model/rtdetr_v41_decoder_fp16.rknn'
IMG = '/home/cat/rknn2.5/test_images/uav.jpg'

MASKS = {
    '1core': RKNNLite.NPU_CORE_0,
    '2core': RKNNLite.NPU_CORE_0_1,
    '3core': RKNNLite.NPU_CORE_0_1_2,
}

img = cv2.imread(IMG)
h, w = img.shape[:2]
s = min(640 / h, 640 / w)
nh, nw = int(h * s), int(w * s)
canvas = np.zeros((640, 640, 3), np.uint8)
t, l = (640 - nh) // 2, (640 - nw) // 2
canvas[t:t + nh, l:l + nw] = cv2.resize(img, (nw, nh))
x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]


def load(path, mask):
    rk = RKNNLite()
    assert rk.load_rknn(path) == 0
    assert rk.init_runtime(core_mask=mask) == 0
    return rk


def bench(fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


feats_ref = None
for tag, mask in MASKS.items():
    a = load(A, mask)
    b = load(B, mask)
    feats = [np.ascontiguousarray(np.asarray(f).transpose(0, 2, 3, 1))
             for f in a.inference(inputs=[x])]
    ta = bench(lambda: a.inference(inputs=[x]))
    tb = bench(lambda: b.inference(inputs=feats))
    print(f'{tag}: A(backbone INT8) {ta:6.1f} ms | B(decoder FP16) {tb:6.1f} ms '
          f'| A+B {ta + tb:6.1f} ms | 3x chain {3000 / (ta + tb):5.1f} FPS', flush=True)
    a.release()
    b.release()
