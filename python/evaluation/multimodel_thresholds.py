#!/usr/bin/env python3
"""Multi-model threshold/recall comparison on top of full (-c -100) detection dumps.

For every model (INT8 / FP16 / experimental INT8 variants) it computes, at several
confidence-threshold strategies:
  * pooled Precision / Recall / F1          (IoU>=0.5, argmax class)
  * macro (class-balanced) F1
  * per-class recall
Strategies: global -0.13, global -0.53, a reference class-wise vector (calibrated on INT8),
and each model's own F1-optimal class-wise vector.

Matching is incremental over score-sorted candidates, so a full threshold sweep costs O(N)
per class.  Models are evaluated in parallel processes.

Usage:
  python multimodel_thresholds.py --ann <val/annotations> \
      --spec "INT8:det_val548.txt,FP16:det_fp16_548.txt" \
      --vector "0:-0.40,1:-0.70,2:-0.90,3:0.00,4:-0.25,5:-0.30,6:-0.70,7:-0.80,8:-0.05,9:-0.55"
"""
import argparse
import math
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visdrone_map as M


def class_curve(dets, gt, cls, iou_thr=0.5):
    """Incremental greedy matching over score-sorted candidates.

    Returns (scores_desc, tp_prefix, fp_prefix, npos).
    tp_prefix[i]/fp_prefix[i] are the counts after consuming candidates 0..i.
    """
    cands = []
    for stem, dl in dets.items():
        for c, s, b in dl:
            if c == cls:
                cands.append((s, stem, b))
    cands.sort(key=lambda r: -r[0])
    npos = sum(len(gt[s].get(cls, [])) for s in gt)
    used = defaultdict(set)
    tp = fp = 0
    tp_p, fp_p, sc = [], [], []
    for s, stem, b in cands:
        gts = gt.get(stem, {}).get(cls, [])
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if j in used[stem]:
                continue
            v = M.iou(b, g)
            if v > best:
                best, bj = v, j
        if bj >= 0 and best >= iou_thr:
            used[stem].add(bj)
            tp += 1
        else:
            fp += 1
        sc.append(s)
        tp_p.append(tp)
        fp_p.append(fp)
    return sc, tp_p, fp_p, npos


def at_threshold(curve, thr):
    sc, tp_p, fp_p, npos = curve
    # last index with score >= thr  (sc is non-increasing)
    lo, hi = 0, len(sc)
    while lo < hi:
        mid = (lo + hi) // 2
        if sc[mid] >= thr:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    tp = tp_p[idx] if idx >= 0 else 0
    fp = fp_p[idx] if idx >= 0 else 0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / npos if npos else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return tp, fp, p, r, f1, npos


def eval_model(job):
    name, det_path, ann, lo, hi, step, ref_vec, fit = job
    dets = M.load_dets(det_path)
    stems = sorted(dets.keys())
    gt, _ = M.load_gt(ann, stems)
    thrs = [round(lo + i * step, 4) for i in range(int(round((hi - lo) / step)) + 1)]

    curves = {c: class_curve(dets, gt, c) for c in range(10)}
    # own F1-optimal threshold per class (skipped when evaluating an unseen split)
    if fit:
        own = {}
        for c in range(10):
            best_t, best_f1 = thrs[0], -1.0
            for t in thrs:
                f1 = at_threshold(curves[c], t)[4]
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            own[c] = best_t
    else:
        own = dict(ref_vec)

    def eval_vec(vec):
        tp = fp = npos = 0
        macro = 0.0
        rec = {}
        for c in range(10):
            t, f, p, r, f1, n = at_threshold(curves[c], vec[c])
            tp += t; fp += f; npos += n
            macro += f1
            rec[c] = r
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / npos if npos else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) else 0.0
        return dict(P=P, R=R, F1=F1, macro=macro / 10.0, TP=tp, FP=fp, FN=npos - tp, rec=rec)

    g13 = {c: -0.13 for c in range(10)}
    g53 = {c: -0.53 for c in range(10)}
    res = {
        'name': name,
        'n_img': len(dets),
        'strategies': {
            'global_-0.13': eval_vec(g13),
            'global_-0.53': eval_vec(g53),
            'int8_vector': eval_vec(ref_vec),
            'own_vector': eval_vec(own),
        },
        'own_vector': own,
        'fit': fit,
    }
    return res


def parse_vec(spec):
    vec = {}
    for part in spec.split(','):
        k, v = part.split(':')
        vec[int(k)] = float(v)
    return vec


def print_model(r):
    print('\n=== %s (images=%d) ===' % (r['name'], r['n_img']))
    print('%-14s %6s %6s %8s %8s %8s %8s' % ('strategy', 'TP', 'FP', 'P', 'R', 'F1', 'macroF1'))
    for k, v in r['strategies'].items():
        print('%-14s %6d %6d %8.4f %8.4f %8.4f %8.4f'
              % (k, v['TP'], v['FP'], v['P'], v['R'], v['F1'], v['macro']))
    own = r['own_vector']
    print('own thresholds: ' + ' '.join('%d:%.2f' % (c, own[c]) for c in range(10)))
    # per-class recall: base vs the strategy with the highest macro F1
    best_key = max(r['strategies'], key=lambda k: r['strategies'][k]['macro'])
    rb = r['strategies']['global_-0.13']['rec']
    rn = r['strategies'][best_key]['rec']
    print('per-class recall  (-0.13 -> %s):' % best_key)
    print('   ' + '  '.join('%s %.3f->%.3f' % (M.CLASSES[c][:6], rb[c], rn[c]) for c in range(10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True)
    ap.add_argument('--spec', required=True, help='name:path[,name:path...]')
    ap.add_argument('--vector', required=True, help='reference (INT8-calibrated) class-wise vector')
    ap.add_argument('--lo', type=float, default=-2.0)
    ap.add_argument('--hi', type=float, default=0.5)
    ap.add_argument('--step', type=float, default=0.05)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--no-fit', action='store_true',
                    help='do not fit thresholds on this split (use the reference vector; for held-out splits)')
    a = ap.parse_args()

    ref_vec = parse_vec(a.vector)
    jobs = []
    for item in a.spec.split(','):
        name, path = item.split(':', 1)
        jobs.append((name, path, a.ann, a.lo, a.hi, a.step, ref_vec, not a.no_fit))

    with Pool(min(a.jobs, len(jobs))) as pool:
        results = pool.map(eval_model, jobs)

    for r in results:
        print_model(r)

    # compact recall-focused table
    print('\n=== 召回/宏 F1 汇总（相对 global -0.13 的提升）===')
    print('%-22s %-14s %8s %8s %8s' % ('model', 'strategy', 'R', 'dR(pp)', 'macroF1'))
    for r in results:
        base = r['strategies']['global_-0.13']
        for k in ('global_-0.13', 'global_-0.53', 'int8_vector', 'own_vector'):
            v = r['strategies'][k]
            print('%-22s %-14s %8.4f %8.2f %8.4f'
                  % (r['name'], k, v['R'], (v['R'] - base['R']) * 100, v['macro']))


if __name__ == '__main__':
    main()
