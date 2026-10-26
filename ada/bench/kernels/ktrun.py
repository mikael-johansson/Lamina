#!/usr/bin/env python3
"""Kernel trace of one request on a server started with bench/libktrace.so preloaded (KTRACE_OUT=<out>):
resets the trace, runs the request (a corpus name or a SYNTH one), writes the table.
    python3 ktrun.py <config> <out> code [--n 2]   (n: that many copies at once)"""
import json, os, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench
cfg = json.loads(Path(sys.argv[1]).read_text()); out = sys.argv[2]; name = sys.argv[3]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 1
bench.API_KEY = cfg.get("api_key")
reqs = {r[0]: r for r in bench.SYNTH}
req = reqs[name] if name in reqs else bench.corpus([name])[0]
def go():
    open(out + ".go", "w").close()
    while os.path.exists(out + ".go"): time.sleep(0.1)
    time.sleep(0.5)
go()   # reset
t0 = time.time()
res = []
def one():
    res.append(bench.call(cfg["port"], req[1], req[2]))
ts = [threading.Thread(target=one) for _ in range(n)]
[t.start() for t in ts]; [t.join() for t in ts]
wall = time.time() - t0
with open(out, "a") as f:
    toks = [r.get("usage", {}).get("completion_tokens") for r in res]
    f.write(f"\n=== {name} x{n}: wall {wall:.2f} s, completion tokens {toks}\n")
go()
