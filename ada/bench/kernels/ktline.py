#!/usr/bin/env python3
"""Per device, from a KTRACE_RAW dump: each kernel's total time, and the time it ran alone on its device (nothing
else there in flight: a floor on its share of the critical path), plus the idle time.  python3 ktline.py <raw>"""
import sys, collections, re
rows = collections.defaultdict(list)
for line in open(sys.argv[1]):
    d, s, a, b, name = line.rstrip("\n").split("\t", 4)
    short = re.sub(r"strata::kernels::(\(anonymous namespace\)::)?", "", name)
    short = re.sub(r"\(.*", "", short)[:90]
    rows[int(d)].append((int(a), int(b), short))
for dev, ks in sorted(rows.items()):
    ks.sort()
    ev = []
    for i, (a, b, n) in enumerate(ks):
        ev += [(a, 1, i), (b, -1, i)]
    ev.sort()
    alone = collections.Counter(); tot = collections.Counter(); cnt = collections.Counter()
    for a, b, n in ks: tot[n] += b - a; cnt[n] += 1
    live = set(); last = ev[0][0]; idle = 0
    for t, k, i in ev:
        if len(live) == 1: alone[ks[next(iter(live))][2]] += t - last
        if not live: idle += t - last
        last = t
        if k == 1: live.add(i)
        else: live.discard(i)
    span = ks[-1][1] - ks[0][0]
    print(f"device {dev}: span {span/1e6:.1f} ms, idle {idle/1e6:.1f} ms, single-kernel {sum(alone.values())/1e6:.1f} ms")
    print(f"{'total ms':>9} {'alone ms':>9} {'n':>7} {'us':>7}  kernel")
    for n, t in sorted(tot.items(), key=lambda x: -x[1])[:30]:
        print(f"{t/1e6:9.1f} {alone[n]/1e6:9.1f} {cnt[n]:7d} {t/1e3/cnt[n]:7.2f}  {n}")
