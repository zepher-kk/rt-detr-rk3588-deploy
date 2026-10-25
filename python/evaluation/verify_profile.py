#!/usr/bin/env python3
"""Evaluate a threshold profile's operating point from its (already thresholded) dump.

All rows in the dump are counted (they already passed the class-wise thresholds),
so the pooled P/R/F1 and the macro (class-balanced) F1 are the deployed metrics.
"""
import os, sys, argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visdrone_map as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--global-conf", type=float)
    ap.add_argument(
        "--conf-class",
        help="Optional per-class thresholds, e.g. 0:-0.50,3:-0.10",
    )
    ap.add_argument("--ignore-regions", action="store_true")
    a = ap.parse_args()

    dets = M.load_dets(a.det)
    stems = sorted(dets.keys())
    gt, ign = M.load_gt(a.ann, stems)
    class_thresholds = {}
    if a.conf_class:
        for item in a.conf_class.split(","):
            class_id, threshold = item.split(":", 1)
            class_thresholds[int(class_id)] = float(threshold)
    thresholds = {
        class_id: class_thresholds.get(class_id, a.global_conf)
        for class_id in range(10)
    }

    if a.ignore_regions or any(value is not None for value in thresholds.values()):
        filtered = defaultdict(list)
        for stem, detections in dets.items():
            ignored_regions = ign.get(stem, [])
            for class_id, score, box in detections:
                if class_id not in thresholds:
                    continue
                threshold = thresholds[class_id]
                if threshold is not None and score < threshold:
                    continue
                if a.ignore_regions and ignored_regions:
                    if M.in_ignored(box, ignored_regions):
                        continue
                filtered[stem].append((class_id, score, box))
        dets = filtered

    print(
        "thresholds="
        + ",".join(
            f"{class_id}:{threshold:+.3f}"
            for class_id, threshold in thresholds.items()
            if threshold is not None
        )
    )
    npos = sum(len(gt[s][c]) for s in gt for c in gt[s])

    tp = fp = 0
    ndet = 0
    per = []
    for c in range(10):
        t = f = 0
        for stem, dl in dets.items():
            cand = sorted([(s, b) for (cc, s, b) in dl if cc == c], key=lambda r: -r[0])
            gts = gt.get(stem, {}).get(c, [])
            matched = set()
            for s, b in cand:
                best, bj = 0.0, -1
                for j, g in enumerate(gts):
                    if j in matched:
                        continue
                    v = M.iou(b, g)
                    if v > best:
                        best, bj = v, j
                if bj >= 0 and best >= a.iou:
                    matched.add(bj); t += 1
                else:
                    f += 1
        n = sum(len(gt[s].get(c, [])) for s in gt)
        p = t / (t + f) if t + f else 0.0
        r = t / n if n else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per.append((n, t, f, p, r, f1)); tp += t; fp += f

    ndet = tp + fp
    P = tp / ndet if ndet else 0.0
    R = tp / npos if npos else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    macro = sum(x[5] for x in per) / 10.0
    print("dump=%s rows=%d images=%d" % (os.path.basename(a.det), ndet, len(dets)))
    print("%-16s %5s %6s %6s %7s %7s %7s" % ("class", "GT", "TP", "FP", "P", "R", "F1"))
    for c in range(10):
        n, t, f, p, r, f1 = per[c]
        print("%-16s %5d %6d %6d %7.4f %7.4f %7.4f" % (M.CLASSES[c], n, t, f, p, r, f1))
    print("POOLED  TP=%d FP=%d FN=%d  P=%.4f R=%.4f F1=%.4f  | MACRO F1=%.4f"
          % (tp, fp, npos - tp, P, R, F1, macro))


if __name__ == "__main__":
    main()
