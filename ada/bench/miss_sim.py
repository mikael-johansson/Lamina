#!/usr/bin/env python3
"""CPU misses and CPU wait per verify window vs the number of experts held in VRAM, from a --dump-routing trace.

    eddoursul/.venv/bin/python ada/bench/miss_sim.py bench/runs/trace-iq3 LEARNED.bin SHIPPED.bin

Three static rankings: the learned profile, the shipped profile, and leave-one-request-out frequencies from the
trace itself. Cost model (ada/bench/results/cpu-pool/tl.out): ~80 us of pool time per distinct missed expert per
layer (7 threads), against ~67 us of the GPU's own expert work, so a layer waits max(0, 80 d - 67) us.
Used for ada/docs/THIRD-GPU-2026-10-29.md.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route_analysis import load_windows, NL, NE

run, learned, shipped = sys.argv[1:4]
wins = load_windows(run + "/routing.bin")
reqs = json.load(open(run + "/requests.json"))
want = [r["predicted_n"] - r["draft_n_accepted"] for r in reqs]
wins = wins[len(wins) - sum(want):]
per_req, i = [], 0
for n in want:
    per_req.append(wins[i:i + n]); i += n
print(len(wins), "windows", np.mean([w.shape[1] for w in wins]), "rows")

def counts(ws):
    c = np.zeros((NL, NE))
    for w in ws:
        for l in range(NL):
            np.add.at(c[l], w[l].ravel(), 1)
    return c

def prof_rank(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"STRP"
        hdr = np.frombuffer(f.read(20), dtype=np.uint32)
        return np.frombuffer(f.read(4 * int(hdr[4])), dtype=np.uint16).reshape(-1, 2)

def res_from_pairs(p, R):
    res = np.zeros((NL, NE), bool); p = p[:R]; res[p[:, 0], p[:, 1]] = True; return res

def res_from_counts(c, R):
    # tie-break with tiny noise so unseen experts are random
    flat = c.ravel() + np.random.default_rng(1).random(c.size) * 1e-6
    idx = np.argsort(-flat)[:R]; res = np.zeros(NL * NE, bool); res[idx] = True
    return res.reshape(NL, NE)

def per_layer_distinct_miss(ws, res):
    d = np.zeros((len(ws), NL)); look = miss = 0
    for k, w in enumerate(ws):
        for l in range(NL):
            e = w[l].ravel(); u = np.unique(e)
            d[k, l] = (~res[l, u]).sum(); look += e.size; miss += (~res[l, e]).sum()
    return d, miss / look

US_PER_EXPERT, GPU_US = 80.0, 67.0   # calibrated: tl.out 6.7 ms pool / (48 x 1.68 distinct); VRAM experts ~67 us/layer
Rs = [17000, 18000, 19000, 20200, 21000, 22000, 22900, 23500, 24000, 24300, 24576]
total = NL * NE
allc = counts(wins)
schemes = {"learned profile": lambda R, k: res_from_pairs(prof_rank(learned), R),
           "shipped profile": lambda R, k: res_from_pairs(prof_rank(shipped), R)}
loo_counts = [allc - counts(per_req[k]) for k in range(len(per_req))]
print(f"{'R':>6} {'onCPU':>6} | " + " | ".join(f"{s:>38}" for s in list(schemes) + ["leave-one-out"]))
for R in Rs:
    row = []
    for name in list(schemes) + ["loo"]:
        ds, mr = [], []
        for k in range(len(per_req)):
            if not per_req[k]: continue
            res = res_from_counts(loo_counts[k], R) if name == "loo" else schemes[name](R, k)
            d, m = per_layer_distinct_miss(per_req[k], res); ds.append(d); mr.append(m * len(per_req[k]))
        d = np.concatenate(ds); m = sum(mr) / len(d)
        pool = d.sum(1).mean() * US_PER_EXPERT / 1000
        stall = np.maximum(0, d * US_PER_EXPERT - GPU_US).sum(1).mean() / 1000
        row.append(f"miss {m:6.2%} d/win {d.sum(1).mean():5.1f} pool {pool:4.1f} stall {stall:4.2f}ms")
    print(f"{R:6d} {total - R:6d} | " + " | ".join(row))
