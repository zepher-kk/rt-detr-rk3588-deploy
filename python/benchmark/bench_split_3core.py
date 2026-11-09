"""Measure aggregate throughput of the split chain in data-parallel mode.

Each process pins one NPU core and runs A(INT8) -> B(FP16) on the same input.
This emulates the C++ pipeline (3 workers, one core each).
"""
import multiprocessing as mp
import os
import time

import cv2
import numpy as np

A = '/home/cat/rknn2.5/model/q150_backbone_int8_45.rknn'
B = '/home/cat/rknn2.5/model/q150_decoder_fp16.rknn'
BA = '/home/cat/rknn2.5/model/rtdetr_q150_l2_4d_mulnpu.rknn'
IMG = '/home/cat/rknn2.5/test_images/uav.jpg'


def bench_chain(core, frames, q):
    from rknnlite.api import RKNNLite
    masks = {0: RKNNLite.NPU_CORE_0, 1: RKNNLite.NPU_CORE_1,
             2: RKNNLite.NPU_CORE_2}

    def load(path):
        rk = RKNNLite()
        assert rk.load_rknn(path) == 0
        assert rk.init_runtime(core_mask=masks[core]) == 0
        return rk

    a, b = load(A), load(B)
    img = cv2.imread(IMG)
    h, w = img.shape[:2]
    s = min(640 / h, 640 / w)
    nh, nw = int(h * s), int(w * s)
    canvas = np.zeros((640, 640, 3), np.uint8)
    t, l = (640 - nh) // 2, (640 - nw) // 2
    canvas[t:t + nh, l:l + nw] = cv2.resize(img, (nw, nh))
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]

    for _ in range(5):
        b.inference(inputs=[np.ascontiguousarray(
            np.asarray(f).transpose(0, 2, 3, 1)) for f in a.inference([x])])
    t0 = time.perf_counter()
    for _ in range(frames):
        feats = a.inference(inputs=[x])
        packed = [np.ascontiguousarray(np.asarray(f).transpose(0, 2, 3, 1))
                  for f in feats]
        b.inference(inputs=packed)
    dt = time.perf_counter() - t0
    q.put((core, frames / dt, dt / frames * 1000))


def bench_single(core, frames, q, model_path=None):
    from rknnlite.api import RKNNLite
    masks = {0: RKNNLite.NPU_CORE_0, 1: RKNNLite.NPU_CORE_1,
             2: RKNNLite.NPU_CORE_2}
    rk = RKNNLite()
    assert rk.load_rknn(model_path) == 0
    assert rk.init_runtime(core_mask=masks[core]) == 0
    img = cv2.imread(IMG)
    h, w = img.shape[:2]
    s = min(640 / h, 640 / w)
    nh, nw = int(h * s), int(w * s)
    canvas = np.zeros((640, 640, 3), np.uint8)
    t, l = (640 - nh) // 2, (640 - nw) // 2
    canvas[t:t + nh, l:l + nw] = cv2.resize(img, (nw, nh))
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]
    for _ in range(5):
        rk.inference(inputs=[x])
    t0 = time.perf_counter()
    for _ in range(frames):
        rk.inference(inputs=[x])
    dt = time.perf_counter() - t0
    q.put((core, frames / dt, dt / frames * 1000))


def run(fn, frames=40, **kw):
    q = mp.Queue()
    ps = [mp.Process(target=fn, args=(c, frames, q), kwargs=kw) for c in (0, 1, 2)]
    for p in ps:
        p.start()
    res = sorted(q.get() for _ in ps)
    for p in ps:
        p.join()
    total = sum(r[1] for r in res)
    print('  per-core:', [f'core{c} {fps:.2f} FPS ({ms:.1f} ms)' for c, fps, ms in res])
    print(f'  aggregate: {total:.2f} FPS')
    return total


if __name__ == '__main__':
    print('=== 150q/2L FP16 single model, 3 workers x 1 core ===')
    run(bench_single, model_path=BA)
    print('=== 150q/2L split INT8->FP16, 3 workers x 1 core ===')
    run(bench_chain)
