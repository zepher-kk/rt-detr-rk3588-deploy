#!/usr/bin/env python3
"""Per-layer quantisation clipping analysis from an RKNN hybrid_quantization_step1 cfg.

clip = 1 - (quantised range) / (observed range)  -> larger means the layer lost more
dynamic range during PTQ.  Used to locate the layers responsible for the INT8 accuracy
loss of the UAV-DETR+-R18 model (see 标准分析表_[2026-10-07-02].md §3).

Usage: python clip_analysis.py <best_xxx.quantization.cfg> [topN]
"""
import sys
import statistics
import collections


def parse_cfg(path):
    layers, cur, key = [], None, None
    for line in open(path):
        line = line.rstrip('\n')
        if line.startswith('    /'):
            cur = {'name': line.split(':')[0].strip()}
            layers.append(cur)
            key = None
            continue
        if cur is None:
            continue
        s = line.strip()
        if s.endswith(':') and not s.startswith('-'):
            key = s[:-1]
            continue
        if s.startswith('- ') and key:
            cur[key] = s[2:].strip()
            key = None
            continue
        if ':' in s and not s.startswith('-'):
            k, v = s.split(':', 1)
            cur[k.strip()] = v.strip()
    return layers


def fv(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    path = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    rows = []
    for L in parse_cfg(path):
        if L.get('dtype') != 'int8':
            continue
        mn, mx = fv(L.get('min')), fv(L.get('max'))
        omn, omx = fv(L.get('ori_min')), fv(L.get('ori_max'))
        if None in (mn, mx, omn, omx) or (omx - omn) <= 0:
            continue
        rows.append((1.0 - (mx - mn) / (omx - omn), L['name'], omx - omn, mx - mn))
    rows.sort(reverse=True)

    print('int8 layers parsed: %d' % len(rows))
    print('median clip = %.4f  mean = %.4f' % (
        statistics.median([r[0] for r in rows]), sum(r[0] for r in rows) / len(rows)))
    for th in (0.6, 0.5, 0.4, 0.3, 0.25, 0.1):
        print('  clip > %.2f : %d layers' % (th, sum(1 for r in rows if r[0] > th)))

    print('\n%-7s %-58s %10s %10s' % ('clip', 'layer', 'obsRange', 'qRange'))
    for c, n, orng, qr in rows[:topn]:
        print('%6.3f  %-58s %10.2f %10.2f' % (c, n[:58], orng, qr))

    def optype(n):
        parts = [p for p in n.strip('/').split('/') if p]
        return parts[-1].split('_')[0] if parts else '?'

    print('\nop types in worst-50:',
          dict(collections.Counter(optype(r[1]) for r in rows[:50]).most_common(12)))


if __name__ == '__main__':
    main()
