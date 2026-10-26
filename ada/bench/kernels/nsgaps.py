#!/usr/bin/env python3
"""From an nsys sqlite export: the long gaps with no kernel on any GPU, and what the host threads did in them (the
CUDA API calls under way, and the CPU samples' innermost strata frames).  python3 nsgaps.py <sqlite> [min_us]"""
import sqlite3, sys, collections, bisect
db = sqlite3.connect(sys.argv[1]); min_ns = float(sys.argv[2] if len(sys.argv) > 2 else 100) * 1e3
S = dict(db.execute("select id, value from StringIds"))
ks = sorted(db.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL"))
gaps = []; end = ks[0][1]
for a, b in ks[1:]:
    if a - end >= min_ns: gaps.append((end, a))
    end = max(end, b)
span = ks[-1][1] - ks[0][0]
gt = sum(b - a for a, b in gaps)
print(f"span {span/1e6:.1f} ms; {len(gaps)} gaps >= {min_ns/1e3:.0f} us, {gt/1e6:.1f} ms")
starts = [g[0] for g in gaps]
def in_gap(t0, t1):   # overlap of [t0, t1) with the gaps
    i = bisect.bisect_right(starts, t1) - 1; o = 0
    while i >= 0 and gaps[i][1] > t0:
        o += max(0, min(t1, gaps[i][1]) - max(t0, gaps[i][0])); i -= 1
    return o
tn = {r[0]: S.get(r[1], "?") for r in db.execute("select globalTid, nameId from ThreadNames")}
api = collections.Counter()
for a, b, nid, tid in db.execute("select start, end, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
    o = in_gap(a, b)
    if o: api[(tid, S.get(nid, "?"))] += o
print("CUDA API time inside the gaps (thread, call):")
for (tid, n), o in api.most_common(12): print(f"  {o/1e6:7.1f} ms  {tn.get(tid, tid)} {tid & 0xffffff}  {n}")
# CPU samples in gaps: innermost frame and innermost frame from the engine (not libc/libcuda)
samp = collections.Counter(); deep = collections.Counter(); nsamp = 0
ev = {r[0]: (r[1], r[2]) for r in db.execute("select id, start, globalTid from COMPOSITE_EVENTS")}
chains = collections.defaultdict(list)
for i, sym, mod, depth in db.execute("select id, symbol, module, stackDepth from SAMPLING_CALLCHAINS order by id, stackDepth"):
    chains[i].append((S.get(sym, "?"), S.get(mod, "?")))
for i, ch in chains.items():
    t, tid = ev.get(i, (0, 0))
    if not in_gap(t, t + 1): continue
    nsamp += 1
    samp[(tn.get(tid, tid), ch[0][0][:70])] += 1
    eng = [f for f, m in ch if "strata" in m]
    path = " < ".join(f[:45] for f in eng[:4])
    deep[(tn.get(tid, tid), path)] += 1
print(f"CPU samples inside the gaps: {nsamp}; by innermost frame:")
for (t, f), n in samp.most_common(15): print(f"  {n:5d}  {t}  {f}")
print("by the engine's innermost frames:")
for (t, f), n in deep.most_common(20): print(f"  {n:5d}  {t}  {f}")
