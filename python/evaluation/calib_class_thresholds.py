#!/usr/bin/env python3
"""Class-wise confidence-threshold calibration on the full VisDrone-val detection dump.

For every class: sweep logit thresholds, compute P/R/F1 @IoU 0.5 (argmax-class assignment),
pick the F1-optimal threshold; then evaluate the pooled metrics when every class uses its
own calibrated threshold.  Pure local recomputation - no board access required.
"""
import os, sys, math, argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visdrone_map as M


def prf(dets, gt, cls, thr):
    """TP/FP/P/R/F1 for one class at one threshold (IoU>=0.5)."""
    npos = sum(len(gt[s].get(cls, [])) for s in gt)
    tp = fp = 0
    for stem, dl in dets.items():
        cand = sorted([(s, b) for (c, s, b) in dl if c == cls and s >= thr], key=lambda r: -r[0])
        if not cand:
            continue
        gts = gt.get(stem, {}).get(cls, [])
        matched = set()
        for s, b in cand:
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j in matched:
                    continue
                v = M.iou(b, g)
                if v > best:
                    best, bj = v, j
            if bj >= 0 and best >= 0.5:
                matched.add(bj); tp += 1
            else:
                fp += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / npos if npos else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return tp, fp, npos - tp, p, r, f1


def pooled(dets, gt, thr_map):
    """Pooled P/R/F1 when每个类别使用 thr_map[c]（None 表示该类不输出）。"""
    tp = fp = fn = 0
    for stem, dl in dets.items():
        sel = [(c, s, b) for (c, s, b) in dl
               if thr_map.get(c) is not None and s >= thr_map[c]]
        sel.sort(key=lambda r: -r[1])
        matched = defaultdict(set)
        for c, s, b in sel:
            gts = gt.get(stem, {}).get(c, [])
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j in matched[c]:
                    continue
                v = M.iou(b, g)
                if v > best:
                    best, bj = v, j
            if bj >= 0 and best >= 0.5:
                matched[c].add(bj); tp += 1
            else:
                fp += 1
    npos = sum(len(gt[s].get(c, [])) for s in gt for c in range(10))
    fn = npos - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / npos if npos else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return tp, fp, fn, p, r, f1


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--det", required=True)
    ap_.add_argument("--ann", required=True)
    ap_.add_argument("--lo", type=float, default=-0.75)
    ap_.add_argument("--hi", type=float, default=0.25)
    ap_.add_argument("--step", type=float, default=0.05)
    ap_.add_argument("--ignore-regions", action="store_true")
    a = ap_.parse_args()

    dets = M.load_dets(a.det)
    stems = sorted(dets.keys())
    gt, ign = M.load_gt(a.ann, stems)
    if a.ignore_regions:
        filtered = defaultdict(list)
        for stem, detections in dets.items():
            regions = ign.get(stem, [])
            for class_id, score, box in detections:
                if regions and M.in_ignored(box, regions):
                    continue
                filtered[stem].append((class_id, score, box))
        dets = filtered
    n = int(round((a.hi - a.lo) / a.step))
    thrs = [round(a.lo + i * a.step, 4) for i in range(n + 1)]

    print("images=%d  thresholds=%s" % (len(dets), thrs))
    best = {}
    ignore_state = "ON" if a.ignore_regions else "OFF"
    print(
        "\n== per-class F1 sweep "
        f"(IoU>=0.5, argmax class, ignore-region rule {ignore_state}) =="
    )
    for c in range(10):
        row = []
        for t in thrs:
            tp, fp, fn, p, r, f1 = prf(dets, gt, c, t)
            row.append((t, tp, fp, p, r, f1))
        bt = max(row, key=lambda x: x[5])
        b13 = min(row, key=lambda x: abs(x[0] - (-0.13)))
        b53 = min(row, key=lambda x: abs(x[0] - (-0.53)))
        best[c] = bt
        print("%-16s  best thr=%+.2f  F1=%.4f (P=%.4f R=%.4f TP=%d FP=%d) | "
              "@-0.13 F1=%.4f | @-0.53 F1=%.4f"
              % (M.CLASSES[c], bt[0], bt[5], bt[3], bt[4], bt[1], bt[2], b13[5], b53[5]))

    print("\n== calibrated threshold map ==")
    print(" ".join("%d:%.2f" % (c, best[c][0]) for c in range(10)))

    thr_map = {c: best[c][0] for c in range(10)}
    for name, tm in [("-0.13 everywhere", {c: -0.13 for c in range(10)}),
                     ("-0.53 everywhere", {c: -0.53 for c in range(10)}),
                     ("class-wise calibrated", thr_map)]:
        tp, fp, fn, p, r, f1 = pooled(dets, gt, tm)
        macro = sum(best[c][5] if tm.get(c) == best[c][0] else prf(dets, gt, c, tm[c])[5]
                    for c in range(10)) / 10.0
        print("%-22s TP=%6d FP=%6d FN=%6d  P=%.4f R=%.4f F1=%.4f  macroF1=%.4f"
              % (name, tp, fp, fn, p, r, f1, macro))

    # threshold curve, all classes (appendix table)
    print("\n== appendix: F1 vs thr, all classes ==")
    hdr = "thr    " + "".join("%-12s" % M.CLASSES[c] for c in range(10))
    print(hdr)
    for t in thrs:
        line = "%+.2f  " % t
        for c in range(10):
            line += "%-12.4f" % prf(dets, gt, c, t)[5]
        print(line)


if __name__ == "__main__":
    main()
# end
