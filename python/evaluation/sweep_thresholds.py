#!/usr/bin/env python3
"""Threshold sweep + operating-point analysis on top of the full (-c -100) detection dump."""
import os, sys, math, argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visdrone_map as M


def op_point(dets, gt, thr):
    npos = sum(len(gt[s][c]) for s in gt for c in gt[s])
    tp = fp = ndet = 0
    for stem, dl in dets.items():
        cand = sorted([(s, c, b) for (c, s, b) in dl if s >= thr], key=lambda r: -r[0])
        ndet += len(cand)
        matched = defaultdict(set)
        for s, c, b in cand:
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
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / npos if npos else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return ndet, tp, fp, npos - tp, prec, rec, f1


def filter_dets(dets, thr):
    out = {}
    for stem, dl in dets.items():
        out[stem] = [(c, s, b) for (c, s, b) in dl if s >= thr]
    return out


def ap_of(dets, gt, ious):
    vals = []
    for c in range(10):
        for t in ious:
            v = M.eval_class(dets, gt, c, t, "all")[0]
            if not math.isnan(v):
                vals.append(v)
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--det", required=True)
    ap_.add_argument("--ann", required=True)
    ap_.add_argument("--thresholds", default="-0.53")
    ap_.add_argument("--sweep", default="")
    a = ap_.parse_args()

    dets = M.load_dets(a.det)
    stems = sorted(dets.keys())
    gt, ign = M.load_gt(a.ann, stems)
    npos = sum(len(gt[s][c]) for s in gt for c in gt[s])
    print("images=%d gt_objects=%d" % (len(dets), npos))

    thrs = [float(x) for x in str(a.thresholds).split(",")]
    if a.sweep:
        lo, hi, step = [float(x) for x in a.sweep.split(",")]
        n = int(round((hi - lo) / step))
        thrs = [round(lo + i * step, 6) for i in range(n + 1)]

    if a.sweep:
        print("\n== sweep: thr, dets/img, P, R, F1, AP50(filtered) ==")
        for t in thrs:
            ndet, tp, fp, fn, p, r, f1 = op_point(dets, gt, t)
            f = filter_dets(dets, t)
            ap50 = sum(M.eval_class(f, gt, c, 0.5, "all")[0] for c in range(10)) / 10.0
            print("thr=%+.3f sig=%.3f dets/img=%6.1f P=%.4f R=%.4f F1=%.4f AP50f=%.4f"
                  % (t, 1 / (1 + math.exp(-t)), ndet / len(dets), p, r, f1, ap50))
    else:
        for t in thrs:
            ndet, tp, fp, fn, p, r, f1 = op_point(dets, gt, t)
            f = filter_dets(dets, t)
            ap = ap_of(f, gt, [0.5 + 0.05 * i for i in range(10)])
            ap50 = sum(M.eval_class(f, gt, c, 0.5, "all")[0] for c in range(10)) / 10.0
            ap75 = sum(M.eval_class(f, gt, c, 0.75, "all")[0] for c in range(10)) / 10.0
            print("\n== thr=%+.3f (sigmoid %.4f) ==" % (t, 1 / (1 + math.exp(-t))))
            print("dets=%d (%.2f/img) TP=%d FP=%d FN=%d P=%.4f R=%.4f F1=%.4f"
                  % (ndet, ndet / len(dets), tp, fp, fn, p, r, f1))
            print("AP(filtered)=%.4f AP50(filtered)=%.4f AP75(filtered)=%.4f" % (ap, ap50, ap75))
            print("per-class AP@0.5 on filtered set:")
            for c in range(10):
                print("   %-16s %.4f  (GT %d)" % (M.CLASSES[c], M.eval_class(f, gt, c, 0.5, "all")[0],
                                                  len(gt.get(stems[0], {}).get(c, [])) * 0 + sum(len(gt[s].get(c, [])) for s in gt)))


if __name__ == "__main__":
    main()
# end
