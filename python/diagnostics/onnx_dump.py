#!/usr/bin/env python3
"""FP32 ONNX (same model as the board's RKNN) detection dump for the VisDrone val set.

Replicates the deployed C++ pipeline exactly:
  * preprocessing: cv2.resize(img,(640,640)) horizontal/vertical stretch, BGR->RGB, /255
  * decode: argmax class over 10 logits + max score, boxes normalised to the input canvas,
    scaled by the ORIGINAL image size, clipped to the image, invalid boxes dropped
  * rows whose max logit <= -1.0 are written with class -1 / score -1.0 (same as the C++ code)
Output format: <stem> <class_id> <score> <x1> <y1> <x2> <y2>
"""
import os, sys, time, argparse
import numpy as np
import onnxruntime as ort
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    so = ort.SessionOptions()
    if a.threads:
        so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    names = sorted(n for n in os.listdir(a.images) if n.lower().endswith((".jpg", ".jpeg", ".png")))
    if a.limit:
        names = names[:a.limit]

    n_rows = 0
    t0 = time.time()
    with open(a.out, "w") as f:
        for i, n in enumerate(names):
            stem = os.path.splitext(n)[0]
            im = Image.open(os.path.join(a.images, n)).convert("RGB")
            W, H = im.size
            arr = np.asarray(im.resize((640, 640), Image.BILINEAR)).astype(np.float32) / 255.0
            outs = sess.run(None, {iname: arr.transpose(2, 0, 1)[None].copy()})
            boxes, logits = outs[3][0], outs[4][0]          # [300,4] , [300,10]
            cls = logits.argmax(1)
            sc = logits.max(1)
            cx, cy, w, h = boxes[:, 0] * W, boxes[:, 1] * H, boxes[:, 2] * W, boxes[:, 3] * H
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            np.clip(x1, 0, W, out=x1); np.clip(x2, 0, W, out=x2)
            np.clip(y1, 0, H, out=y1); np.clip(y2, 0, H, out=y2)
            for k in range(300):
                if sc[k] <= -1.0:
                    f.write("%s -1 -1.0 0 0 0 0\n"); n_rows += 1
                    continue
                X1, Y1, X2, Y2 = int(round(x1[k])), int(round(y1[k])), int(round(x2[k])), int(round(y2[k]))
                if X2 <= X1 or Y2 <= Y1:
                    continue
                f.write("%s %d %.5f %d %d %d %d\n" % (stem, cls[k], sc[k], X1, Y1, X2, Y2))
                n_rows += 1
            if (i + 1) % 100 == 0:
                print("  %d/%d  %.1fs" % (i + 1, len(names), time.time() - t0), flush=True)
    print("done: images=%d rows=%d elapsed=%.1fs (%.2f img/s)"
          % (len(names), n_rows, time.time() - t0, len(names) / (time.time() - t0)))


if __name__ == "__main__":
    main()
