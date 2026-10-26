#!/usr/bin/env python3
"""What a decode step touches, from a --dump-routing trace (bench/trace_run.py), and what batching would do to it.

    eddoursul/.venv/bin/python bench/route_analysis.py bench/runs/trace-iq3 --profile P --resident 21403

Per decode step (one verify window: 1 + drafts rows, 48 layers x 10 routed experts per row) it counts the routed
lookups, the distinct experts, and the distinct experts outside the resident set (the top `--resident` pairs of the
expert profile: the fork's two GPU tiers at start; the adaptive tier's swaps are left out, so this is the static
cache). Batching is simulated by stepping C different requests together: step j of the batch is window j of each of
C requests (each keeping its own draft rows), and the union per layer is what one shared verify pass would read.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

NL, NE = 48, 512


def load_windows(path):
    raw = np.fromfile(path, dtype=np.int32)
    k = int(raw[1])
    rec = 2 + 2 * k
    n = len(raw) // rec
    raw = raw[:n * rec].reshape(n, rec)
    lay, ids = raw[:, 0], raw[:, 2:2 + k]
    starts = np.r_[0, np.flatnonzero(np.diff(lay) != 0) + 1]
    sizes = np.diff(np.r_[starts, n])
    wb = np.flatnonzero(np.r_[True, np.diff(lay[starts]) <= 0])
    out = []
    for w0, w1 in zip(wb, np.r_[wb[1:], len(starts)]):
        if w1 - w0 != NL:
            continue
        s, e = starts[w0], starts[w1 - 1] + sizes[w1 - 1]
        rows = sizes[w0]
        out.append(ids[s:e].reshape(NL, rows, k))      # [layer, row, k]
    return out


def load_profile(path, slots):
    with open(path, "rb") as f:
        assert f.read(4) == b"STRP"
        hdr = np.frombuffer(f.read(20), dtype=np.uint32)
        pairs = np.frombuffer(f.read(4 * int(hdr[4])), dtype=np.uint16).reshape(-1, 2)
    res = np.zeros((NL, NE), bool)
    p = pairs[:slots]
    res[p[:, 0], p[:, 1]] = True
    return res


def step_stats(wins, res):
    """wins: the windows stepped together. Lookups, distinct, distinct misses, missed lookups."""
    look = dist = dmiss = lmiss = 0
    for l in range(NL):
        e = np.concatenate([w[l].ravel() for w in wins])
        look += e.size
        u = np.unique(e)
        dist += u.size
        dmiss += int((~res[l, u]).sum())
        lmiss += int((~res[l, e]).sum())
    return look, dist, dmiss, lmiss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--resident", type=int, required=True)
    ap.add_argument("--blob-mb", type=float, default=2.05, help="mean expert size in MB (IQ3_S 46.84 GiB / 24576)")
    a = ap.parse_args()
    run = Path(a.run)
    wins = load_windows(run / "routing.bin")
    reqs = json.loads((run / "requests.json").read_text())
    want = [r["predicted_n"] - r["draft_n_accepted"] for r in reqs]
    extra = len(wins) - sum(want)
    print(f"{len(wins)} windows, requests expect {sum(want)}: the first {extra} (warm-up) are dropped")
    wins = wins[extra:]
    per_req, i = [], 0
    for n in want:
        per_req.append(wins[i:i + n])
        i += n
    res = load_profile(a.profile, a.resident)

    print(f"\nresident set: {a.resident} of {NL * NE} experts ({a.resident / (NL * NE):.1%})")
    print("\n== C=1: per decode step (window) ==")
    tot = np.array([step_stats([w], res) for w in wins], float)
    rows = np.array([w.shape[1] for w in wins])
    look, dist, dmiss, lmiss = tot.mean(0)
    print(f"rows (tokens verified) per step: mean {rows.mean():.2f}")
    print(f"routed lookups {look:.0f}, distinct experts {dist:.0f} ({dist / look:.0%} of lookups)")
    print(f"missed lookups {lmiss:.1f} ({tot[:, 3].sum() / tot[:, 0].sum():.2%}), distinct missed experts {dmiss:.1f} "
          f"= {dmiss * a.blob_mb:.0f} MB per step")
    for r in sorted(set(rows.tolist())):
        m = rows == r
        if m.sum() >= 15:
            t = tot[m].mean(0)
            print(f"  {r} rows: {m.sum():4d} steps, distinct {t[1]:.0f}, distinct misses {t[2]:.1f}")

    print("\n== batching: C requests per step (step j = window j of each) ==")
    rng = np.random.default_rng(0)
    big = [k for k, p in enumerate(per_req) if len(p) >= 60]
    for c in (1, 2, 4, 8):
        combos = list(itertools.combinations(big, c))
        rng.shuffle(combos)
        acc = []
        for combo in combos[:40]:
            n = min(len(per_req[k]) for k in combo)
            for j in range(0, n, 3):
                acc.append(step_stats([per_req[k][j] for k in combo], res))
        acc = np.array(acc, float).mean(0)
        print(f"C={c}: lookups {acc[0]:5.0f}  distinct {acc[1]:5.0f}  distinct misses {acc[2]:5.1f} "
              f"({acc[2] * a.blob_mb:5.0f} MB)  per request: distinct {acc[1] / c:5.0f}, misses {acc[2] / c:5.1f}")


if __name__ == "__main__":
    sys.exit(main())
