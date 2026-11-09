"""Measure aggregate throughput with N concurrent RKNNLite instances.

Each worker loads its own context and runs LOOP inferences; the parent measures
wall time and reports aggregate FPS. Core masks are configurable per worker.
"""
import sys
import time
import multiprocessing as mp

import cv2
import numpy as np

MODEL = '/home/cat/rknn2.5/model/rtdetr_native_tpopt2_mulnpu_fp16.rknn'
IMG = '/home/cat/rknn2.5/test_images/uav.jpg'
LOOPS = 100
WARMUP = 10

CORES = {
    'auto': None,
    'all': '0,1,2',
    'c0': '0',
    'c1': '1',
    'c2': '2',
}


def worker(idx, mask_key, loops, q):
    from rknnlite.api import RKNNLite

    img = cv2.imread(IMG)
    h, w = img.shape[:2]
    scale = min(640 / h, 640 / w)
    nh, nw = int(h * scale), int(w * scale)
    canvas = np.zeros((640, 640, 3), np.uint8)
    top, left = (640 - nh) // 2, (640 - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    bench = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]

    rk = RKNNLite()
    assert rk.load_rknn(MODEL) == 0
    mask = CORES[mask_key]
    if mask is None:
        assert rk.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) == 0
    elif mask == '0,1,2':
        assert rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
    elif mask == '0':
        assert rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0) == 0
    elif mask == '1':
        assert rk.init_runtime(core_mask=RKNNLite.NPU_CORE_1) == 0
    elif mask == '2':
        assert rk.init_runtime(core_mask=RKNNLite.NPU_CORE_2) == 0

    for _ in range(WARMUP):
        rk.inference(inputs=[bench])
    q.put(('ready', idx))

    ts = []
    for _ in range(loops):
        t0 = time.perf_counter()
        rk.inference(inputs=[bench])
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    q.put(('done', idx, float(ts.mean() * 1e3), float(np.percentile(ts, 50) * 1e3)))
    rk.release()


def run_case(n, masks):
    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(i, masks[i], LOOPS, q)) for i in range(n)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for _ in range(n):
        q.get(timeout=600)  # ready
    t1 = time.perf_counter()          # all warm, start measuring
    results = {}
    for _ in range(n):
        msg = q.get(timeout=600)
        while msg[0] != 'done':
            msg = q.get(timeout=600)
        results[msg[1]] = (msg[2], msg[3])
    t2 = time.perf_counter()
    for p in procs:
        p.join()

    wall = t2 - t1
    agg = n * LOOPS / wall
    print(f'n={n} masks={masks} aggregate={agg:.2f} FPS  wall={wall:.2f}s '
          f'per-worker mean={[round(v[0],1) for v in results.values()]}')
    return agg


if __name__ == '__main__':
    print(f'model={MODEL.split("/")[-1]} loops={LOOPS} warmup={WARMUP}')
    run_case(1, ['all'])
    run_case(2, ['all', 'all'])
    run_case(3, ['all', 'all', 'all'])
    run_case(3, ['c0', 'c1', 'c2'])
    run_case(2, ['c0', 'c1'])
