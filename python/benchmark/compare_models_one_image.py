import numpy as np
import cv2
from rknnlite.api import RKNNLite

ROOT = '/home/cat/rknn2.5'
IMG = f'{ROOT}/visdrone_val/0000001_02999_d_0000005.jpg'

img = cv2.imread(IMG)
h, w = img.shape[:2]
scale = min(640 / h, 640 / w)
nh, nw = int(h * scale), int(w * scale)
canvas = np.zeros((640, 640, 3), np.uint8)
top, left = (640 - nh) // 2, (640 - nw) // 2
canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]
print('image', img.shape, 'scale', round(scale, 4), 'pad', top, left)


def run(path):
    rk = RKNNLite()
    assert rk.load_rknn(path) == 0
    assert rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
    outs = rk.inference(inputs=[x])
    y = np.squeeze(np.asarray(outs[0])).astype(np.float32)
    return y


for tag, path in (('v4.0.1', f'{ROOT}/model/rtdetr_tpopt2_4d_mulnpu.rknn'),
                  ('v3.0.5', f'{ROOT}/model/rtdetr_fp16.rknn')):
    y = run(path)
    s = y[:, 4:]
    best = s.max(-1)
    order = np.argsort(-best)[:5]
    print(f'--- {tag}: y max={s.max():.4f}  n>0.5={int((best > 0.5).sum())} '
          f'n>0.3={int((best > 0.3).sum())}')
    for i in order:
        print(f'    cls={int(s[i].argmax())} score={best[i]:.4f} '
              f'box={np.round(y[i, :4], 4)}')
