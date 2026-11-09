"""Save all RKNN outputs for one image to npz for cross-checking."""
import sys
import cv2
import numpy as np
from rknnlite.api import RKNNLite

MODEL = sys.argv[1]
OUT = sys.argv[2]
IMG = '/home/cat/rknn2.5/test_images/uav.jpg'

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
np.savez(OUT, **{f'out{i}': np.squeeze(o).astype(np.float32) for i, o in enumerate(outs)})
print('saved', OUT, [np.squeeze(o).shape for o in outs])
