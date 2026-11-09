"""Compare one model from board npz A against one model from board npz B."""
import argparse
import os

import numpy as np


def iou_matrix(a, b):
    ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
    ax2, ay2 = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    bx2, by2 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :]), 0, None)
    inter = iw * ih
    area_a = np.clip(a[:, 2], 0, None) * np.clip(a[:, 3], 0, None)
    area_b = np.clip(b[:, 2], 0, None) * np.clip(b[:, 3], 0, None)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def agreement(la, ba, lb, bb, thres=0.45, iou_thres=0.5):
    sa, sb = la.max(-1), lb.max(-1)
    ma, mb = sa > thres, sb > thres
    ca, cb = la.argmax(-1)[ma], lb.argmax(-1)[mb]
    A, B = ba[ma], bb[mb]
    if len(A) == 0 and len(B) == 0:
        return 1.0, 1.0, 0.0
    if len(A) == 0 or len(B) == 0:
        return 0.0, 0.0, 0.0
    ious = iou_matrix(A, B)
    used, matched, iou_sum = set(), 0, 0.0
    for i in range(len(A)):
        best_j, best_iou = -1, 0.0
        for j in range(len(B)):
            if j in used or cb[j] != ca[i]:
                continue
            if ious[i, j] > best_iou:
                best_j, best_iou = j, ious[i, j]
        if best_j >= 0 and best_iou >= iou_thres:
            used.add(best_j)
            matched += 1
            iou_sum += best_iou
    return matched / len(A), matched / len(B), (iou_sum / matched if matched else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--a-index', type=int, default=0)
    ap.add_argument('--b', required=True)
    ap.add_argument('--b-index', type=int, default=0)
    ap.add_argument('--thres', type=float, default=0.45)
    args = ap.parse_args()

    fa = np.load(args.a, allow_pickle=True)
    fb = np.load(args.b, allow_pickle=True)
    na = [os.path.basename(str(p)) for p in fa['paths']]
    nb = [os.path.basename(str(p)) for p in fb['paths']]
    amap = {n: i for i, n in enumerate(na)}

    cos, rec, pre, iou = [], [], [], []
    for bi, name in enumerate(nb):
        if name not in amap:
            continue
        ai = amap[name]
        la = fa[f'logits_{args.a_index}_{ai}']
        ba = fa[f'boxes_{args.a_index}_{ai}']
        lb = fb[f'logits_{args.b_index}_{bi}']
        bb = fb[f'boxes_{args.b_index}_{bi}']
        cos.append(float(np.dot(la.ravel(), lb.ravel()) / (np.linalg.norm(la) * np.linalg.norm(lb) + 1e-9)))
        r, p, i = agreement(la, ba, lb, bb, args.thres)
        rec.append(r); pre.append(p); iou.append(i)

    print(f'A = {os.path.basename(args.a)}[{args.a_index}]')
    print(f'B = {os.path.basename(args.b)}[{args.b_index}]')
    print(f'paired {len(cos)} images, conf>={args.thres}')
    print(f'logits cosine: mean {np.mean(cos):.6f} min {np.min(cos):.6f}')
    print(f'detection    : recall {np.mean(rec) * 100:.2f}% precision {np.mean(pre) * 100:.2f}% meanIoU {np.mean(iou):.4f}')
    for tag, arr, idx in (('A', fa, args.a_index), ('B', fb, args.b_index)):
        key = f'timing_{idx}'
        if key in arr:
            t = arr[key]
            print(f'{tag} timing    : inference {t[1]:.2f} ms (p50 {t[4]:.2f}, p95 {t[5]:.2f}) | {t[6]:.2f} FPS')


if __name__ == '__main__':
    main()
