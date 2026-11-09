#!/usr/bin/env python3
"""COCO-style mAP evaluator for VisDrone-DET detections (pure Python, no deps).

Detections file lines:  <image_stem> <class_id 0..9> <score> <x1> <y1> <x2> <y2>
GT: VisDrone annotation txt (left,top,width,height,score,category,truncation,occlusion)
Class ids follow the model order: 0 Pedestrian .. 9 Motor  (VisDrone categories 1..10)
"""
import os, sys, glob, math, argparse
from collections import defaultdict

CLASSES = ["Pedestrian", "People", "Bicycle", "Car", "Van",
           "Truck", "Tricycle", "Awning-tricycle", "Bus", "Motor"]
AREA_RNG = {"all": (0, 1e18), "small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e18)}


def load_gt(ann_dir, stems):
    gt = defaultdict(lambda: defaultdict(list))  # img -> cls -> [box]
    ign = defaultdict(list)                      # img -> [ignored region box]
    for stem in stems:
        p = os.path.join(ann_dir, stem + ".txt")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                v = line.split(",")
                if len(v) < 6:
                    continue
                x, y, w, h = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
                score = int(float(v[4]))
                cat = int(float(v[5]))
                if cat == 0 or score == 0:
                    ign[stem].append((x, y, x + w, y + h))
                    continue
                if cat < 1 or cat > 10:
                    continue
                gt[stem][cat - 1].append((x, y, x + w, y + h))
    return gt, ign


def in_ignored(box, regions):
    """VisDrone 规则：检测框与忽略区域交集 / 检测框面积 > 0.5 时该检测被忽略。"""
    a = area_of(box)
    if a <= 0:
        return False
    for r in regions:
        ix1, iy1 = max(box[0], r[0]), max(box[1], r[1])
        ix2, iy2 = min(box[2], r[2]), min(box[3], r[3])
        iw, ih = ix2 - ix1, iy2 - iy1
        if iw > 0 and ih > 0 and (iw * ih) / a > 0.5:
            return True
    return False


def load_dets(path):
    dets = defaultdict(list)  # (img, cls) -> [(score, box)]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = line.split()
            stem, cls, score, x1, y1, x2, y2 = v[0], int(v[1]), float(v[2]), float(v[3]), float(v[4]), float(v[5]), float(v[6])
            dets[stem].append((cls, score, (x1, y1, x2, y2)))
    return dets


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def area_of(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def voc_ap(tp, fp, npos):
    """101-point interpolated AP (COCO style)."""
    if npos == 0:
        return float("nan")
    tp = list(tp); fp = list(fp)
    for i in range(1, len(tp)):
        tp[i] += tp[i - 1]; fp[i] += fp[i - 1]
    rec = [t / npos for t in tp]
    prec = [tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0 for i in range(len(tp))]
    # monotonic precision envelope
    for i in range(len(prec) - 2, -1, -1):
        if prec[i] < prec[i + 1]:
            prec[i] = prec[i + 1]
    ap, prev_r = 0.0, 0.0
    for k in range(101):
        r = k / 100.0
        p = 0.0
        for i in range(len(rec)):
            if rec[i] >= r:
                p = prec[i]
                break
        ap += p
        prev_r = r
    return ap / 101.0


def eval_class(dets, gt, cls, thr, area_key):
    lo, hi = AREA_RNG[area_key]
    # per image: list of (score, box, img)
    rows = []
    for stem, dl in dets.items():
        for c, s, b in dl:
            if c == cls:
                rows.append((s, b, stem))
    rows.sort(key=lambda r: -r[0])

    npos = 0
    for stem in dets.keys():
        for b in gt.get(stem, {}).get(cls, []):
            if lo < area_of(b) <= hi or (area_key == "all" and area_of(b) > 0):
                npos += 1
    if area_key == "all":
        npos = sum(len(gt.get(s, {}).get(cls, [])) for s in dets.keys())

    used = defaultdict(set)
    tp, fp = [], []
    for s, b, stem in rows:
        gts = gt.get(stem, {}).get(cls, [])
        best, best_j = -1.0, -1
        for j, g in enumerate(gts):
            if j in used[stem]:
                continue
            v = iou(b, g)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best >= thr:
            used[stem].add(best_j)
            ga = area_of(gts[best_j])
            if area_key != "all" and not (lo < ga <= hi):
                continue  # matched GT outside area range -> ignore this detection
            tp.append(1); fp.append(0)
        else:
            fp.append(1); tp.append(0)
    return voc_ap(tp, fp, npos), npos


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--det", required=True)
    ap_.add_argument("--ann", required=True)
    ap_.add_argument("--img-dir", default=None)
    ap_.add_argument("--conf", default="-0.13", help="comma list of operating thresholds for P/R/F1")
    ap_.add_argument("--ignore-regions", action="store_true",
                     help="apply VisDrone ignored-region rule (IoA>0.5 -> drop detection)")
    a = ap_.parse_args()

    dets = load_dets(a.det)
    stems = sorted(dets.keys())
    if a.img_dir:
        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(a.img_dir, "*"))
                       if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    gt, ign = load_gt(a.ann, stems)
    if a.ignore_regions:
        dropped = 0
        for stem in list(dets.keys()):
            regs = ign.get(stem, [])
            if regs:
                keep = [d for d in dets[stem] if not in_ignored(d[2], regs)]
                dropped += len(dets[stem]) - len(keep)
                dets[stem] = keep
        print("ignored-region rule ON: dropped %d detections (%.2f%%)"
              % (dropped, 100.0 * dropped / max(1, 164400)))

    ious = [0.5 + 0.05 * i for i in range(10)]
    summary = {}
    per_class50 = {}
    for c in range(10):
        aps = [eval_class(dets, gt, c, t, "all")[0] for t in ious]
        vals = [v for v in aps if not math.isnan(v)]
        summary[CLASSES[c]] = (sum(vals) / len(vals) if vals else float("nan"), aps[0])
        per_class50[CLASSES[c]] = aps[0]

    def mean(vals):
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    print("images_with_detections = %d, gt_images = %d" % (len(dets), len(gt)))
    print("classes = %d" % 10)
    npos_cls = [sum(len(gt.get(s, {}).get(c, [])) for s in gt.keys()) for c in range(10)]
    print("\nper-class: GT  AP(0.5:0.95)  AP50")
    for c in range(10):
        m, a50 = summary[CLASSES[c]]
        print("  %-16s %7d  %.4f  %.4f" % (CLASSES[c], npos_cls[c], m, a50))
    print("\n== COCO-style summary ==")
    mAP = mean([summary[CLASSES[c]][0] for c in range(10)])
    mAP50 = mean([summary[CLASSES[c]][1] for c in range(10)])
    print("AP      = %.4f" % mAP)
    print("AP50    = %.4f" % mAP50)
    ap75 = mean([eval_class(dets, gt, c, 0.75, "all")[0] for c in range(10)])
    print("AP75    = %.4f" % ap75)
    for k in ("small", "medium", "large"):
        v = mean([eval_class(dets, gt, c, 0.5, k)[0] for c in range(10)])
        print("AP50_%-7s = %.4f" % (k, v))
    print("AP_small(0.5:0.95) = %.4f" % mean([eval_class(dets, gt, c, t, "small")[0] for c in range(10) for t in ious]))
    print("AP_medium(0.5:0.95) = %.4f" % mean([eval_class(dets, gt, c, t, "medium")[0] for c in range(10) for t in ious]))
    print("AP_large(0.5:0.95) = %.4f" % mean([eval_class(dets, gt, c, t, "large")[0] for c in range(10) for t in ious]))

    ndet = sum(len(v) for v in dets.values())
    npos_all = sum(len(gt[s][c]) for s in gt for c in gt[s])
    print("\ndetections_total = %d (%.2f/img), gt_objects = %d (%.2f/img)"
          % (ndet, ndet / max(1, len(dets)), npos_all, npos_all / max(1, len(gt))))

    # operating points: P/R/F1 with argmax-class detections at score >= thr
    for thr in [float(t) for t in str(a.conf).split(",")]:
        tp = fp = 0
        for stem in dets:
            matched = defaultdict(set)
            cand = sorted([(s, c, b) for (c, s, b) in dets[stem] if s >= thr], key=lambda r: -r[0])
            for s, c, b in cand:
                gts = gt.get(stem, {}).get(c, [])
                best, bj = 0.0, -1
                for j, g in enumerate(gts):
                    if j in matched[c]:
                        continue
                    v = iou(b, g)
                    if v > best:
                        best, bj = v, j
                if bj >= 0 and best >= 0.5:
                    matched[c].add(bj); tp += 1
                else:
                    fp += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / npos_all if npos_all else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        sig = 1.0 / (1.0 + math.exp(-thr))
        print("op thr=%-7.3f (sigmoid=%.3f): TP=%6d FP=%6d FN=%6d  P=%.4f R=%.4f F1=%.4f"
              % (thr, sig, tp, fp, npos_all - tp, prec, rec, f1))


if __name__ == "__main__":
    main()
# end
