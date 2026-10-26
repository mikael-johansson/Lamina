#!/usr/bin/env python3
"""Both devices' kernels together: the time no kernel ran anywhere, its gaps by length, and the kernels on each side
of the long ones (what the GPUs waited between).  python3 ktgaps.py <raw>"""
import sys, collections, re
ks = []
for line in open(sys.argv[1]):
    d, s, a, b, name = line.rstrip("\n").split("\t", 4)
    short = re.sub(r"strata::kernels::(\(anonymous namespace\)::)?", "", name)
    ks.append((int(a), int(b), int(d), re.sub(r"\(.*", "", short)[:60]))
ks.sort()
gaps = []; end = ks[0][1]; prev = ks[0]
for k in ks[1:]:
    if k[0] > end: gaps.append((k[0] - end, prev, k))
    if k[1] > end: end, prev = k[1], k
span = ks[-1][1] - ks[0][0]
tot = sum(g[0] for g in gaps)
print(f"span {span/1e6:.1f} ms, nothing running {tot/1e6:.1f} ms ({100*tot/span:.1f}%) in {len(gaps)} gaps")
for lo, hi in [(0, 2e3), (2e3, 5e3), (5e3, 2e4), (2e4, 1e5), (1e5, 1e9)]:
    sel = [g for g in gaps if lo <= g[0] < hi]
    print(f"  {lo/1e3:6.0f}-{hi/1e3:<6.0f} us: {len(sel):6d} gaps, {sum(g[0] for g in sel)/1e6:7.1f} ms")
pairs = collections.Counter(); pt = collections.Counter()
for g, a, b in gaps:
    if g >= 5e3:
        key = f"d{a[2]} {a[3]}  ->  d{b[2]} {b[3]}"
        pairs[key] += 1; pt[key] += g
print("gaps >= 5 us by the kernels around them:")
for k, n in pt.most_common(15):
    print(f"  {n/1e6:7.1f} ms {pairs[k]:5d}x  {k}")
