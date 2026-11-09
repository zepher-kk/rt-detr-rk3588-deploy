"""Compare PyTorch reference outputs with board RKNN outputs."""
import argparse
import os

import numpy as np


def iou_matrix(a, b):
    # a: (N,4) cxcywh, b: (M,4)
    ax1 = a[:, 0] - a[:, 2] / 2
    ay1 = a[:, 1] - a[:, 3] / 2
    ax2 = a[:, 0] + a[:, 2] / 2
    ay2 = a[:, 1] + a[:, 3] / 2
    bx1 = b[:, 0] - b[:, 2] / 2
    by1 = b[:, 1] - b[:, 3] / 2
    bx2 = b[:, 0] + b[:, 2] / 2
    by2 = b[:, 1] + b[:, 3] / 2
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = np.clip(a[:, 2], 0, None) * np.clip(a[:, 3], 0, None)
    area_b = np.clip(b[:, 2], 0, None) * np.clip(b[:, 3], 0, None)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def detection_agreement(ref_logits, ref_boxes, logits, boxes, thres=0.45, iou_thres=0.5):
    rs = ref_logits.max(-1)
    bs = logits.max(-1)
    rmask = rs > thres
    bmask = bs > thres
    rcls = ref_logits.argmax(-1)[rmask]
    bcls = logits.argmax(-1)[bmask]
    rb = ref_boxes[rmask]
    bb = boxes[bmask]
    if len(rb) == 0 and len(bb) == 0:
        return 1.0, 1.0, 0.0
    if len(rb) == 0 or len(bb) == 0:
        return 0.0, 0.0, 0.0
    ious = iou_matrix(rb, bb)
    used = set()
    matched = 0
    iou_sum = 0.0
    for i in range(len(rb)):
        best_j, best_iou = -1, 0.0
        for j in range(len(bb)):
            if j in used or bcls[j] != rcls[i]:
                continue
            if ious[i, j] > best_iou:
                best_j, best_iou = j, ious[i, j]
        if best_j >= 0 and best_iou >= iou_thres:
            used.add(best_j)
            matched += 1
            iou_sum += best_iou
    recall = matched / len(rb)
    precision = matched / len(bb)
    mean_iou = iou_sum / matched if matched else 0.0
    return recall, precision, mean_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--board', required=True)
    ap.add_argument('--thres', type=float, default=0.45)
    args = ap.parse_args()

    ref = np.load(args.ref, allow_pickle=True)
    board = np.load(args.board, allow_pickle=True)
    models = [str(x) for x in board['model_list']]
    board_paths = [str(x) for x in board['paths']]
    ref_paths = [str(x) for x in ref['paths']]

    # Pair images by basename: ref and board may list them in different order.
    ref_index = {os.path.basename(p): i for i, p in enumerate(ref_paths)}
    pairs = []
    for bi, bp in enumerate(board_paths):
        name = os.path.basename(bp)
        if name not in ref_index:
            print(f'[skip] no reference for {name}')
            continue
        pairs.append((ref_index[name], bi, name))

    print(f'ref   : {args.ref}')
    print(f'board : {args.board}')
    print(f'images: {len(pairs)} paired (ref {len(ref_paths)}, board {len(board_paths)})')
    print()

    for mi, model in enumerate(models):
        cos, ldiff, bdiff = [], [], []
        recalls, precisions, ious = [], [], []
        for ri, bi, name in pairs:
            rl = ref[f'logits_{ri}']
            rb = ref[f'boxes_{ri}']
            bl = board[f'logits_{mi}_{bi}']
            bb = board[f'boxes_{mi}_{bi}']
            cos.append(float(np.dot(rl.ravel(), bl.ravel()) / (np.linalg.norm(rl) * np.linalg.norm(bl) + 1e-9)))
            ldiff.append(float(np.abs(rl - bl).max()))
            bdiff.append(float(np.abs(rb - bb).max()))
            r, p, iou = detection_agreement(rl, rb, bl, bb, thres=args.thres)
            recalls.append(r); precisions.append(p); ious.append(iou)
        print(f'===== {model} =====')
        print(f'  logits cosine : mean {np.mean(cos):.6f}  min {np.min(cos):.6f}')
        print(f'  logits maxdiff: mean {np.mean(ldiff):.5f}  max {np.max(ldiff):.5f}')
        print(f'  boxes  maxdiff: mean {np.mean(bdiff):.6f}  max {np.max(bdiff):.6f}')
        print(f'  det>={args.thres}: recall {np.mean(recalls)*100:.2f}%  precision {np.mean(precisions)*100:.2f}%  meanIoU {np.mean(ious):.4f}')
        if f'timing_{mi}' in board:
            t = board[f'timing_{mi}']
            print(f'  timing: pre {t[0]:.2f} | inference {t[1]:.2f} (p50 {t[4]:.2f}, p95 {t[5]:.2f}) | post {t[2]:.2f} | total {t[3]:.2f} ms | {t[6]:.2f} FPS')
        print()


if __name__ == '__main__':
    main()
