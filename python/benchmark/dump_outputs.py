"""Dump all RKNN outputs for one image and compare with PyTorch tensors."""
import sys
import numpy as np

IMG = '/home/cat/rknn2.5/test_images/uav.jpg'
MODEL = sys.argv[1] if len(sys.argv) > 1 else '/home/cat/rknn2.5/model/rtdetr_native_tpopt2_mulnpu_fp16.rknn'

import cv2
from rknnlite.api import RKNNLite

img = cv2.imread(IMG)
h, w = img.shape[:2]
scale = min(640 / h, 640 / w)
nh, nw = int(h * scale), int(w * scale)
canvas = np.zeros((640, 640, 3), np.uint8)
top, left = (640 - nh) // 2, (640 - nw) // 2
canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]

rk = RKNNLite()
assert rk.load_rknn(MODEL) == 0
assert rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
outs = rk.inference(inputs=[x])
print(f'n_outputs={len(outs)}')
for i, o in enumerate(outs):
    a = np.squeeze(o).astype(np.float32)
    print(f'out[{i}] shape={a.shape} min={a.min():.4f} max={a.max():.4f} '
          f'mean={a.mean():.4f}  (raw {o.shape})')
