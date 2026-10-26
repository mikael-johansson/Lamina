#!/usr/bin/env python3
"""Concurrency benchmark: C streaming clients at once against a Strata server (fork or upstream).

    python3 bench/load.py configs/up-iq3_s-p4.json --tag up-iq3-p4 --levels 1,2,4
    python3 bench/load.py configs/fork-iq3_s-tuned.json --tag fork-iq3 --levels 1,2,4

Starts the server from the config (from the config's "cwd" checkout), waits until it is loaded, then per level C
runs `--rounds` rounds of C simultaneous requests for each workload:
  short  a ~60-token coding prompt, 512 tokens out (each client a different task, so no shared prefix)
  long   the opencode-30k request from corpus.json (~49K tokens), 256 tokens out; each client's copy starts with
         its own nonce, so every request reads its whole prompt (no prompt-cache hits between clients)
Reports per round: wall time, total output tok/s (all clients' tokens / wall), and per client the time to first
token and decode tok/s (tokens after the first / time after the first). Results in bench/runs/<tag>/load.json.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench import corpus  # noqa: E402

TASKS = [
    "Write a Python function that parses an ISO-8601 duration string into seconds, with tests.",
    "Write a Rust function that merges overlapping intervals, with unit tests.",
    "Write a TypeScript debounce helper with cancellation and a flush method, with tests.",
    "Write a Go HTTP middleware that rate-limits per client IP with a token bucket.",
    "Write a C function that reverses a singly linked list in place, with a small test harness.",
    "Write a Python LRU cache class without functools, with tests.",
    "Write a SQL query and schema for the top 3 products by revenue per month.",
    "Write a bash script that rotates log files older than 7 days and compresses them.",
]


def stream(port, key, body, out):
    """One streaming /v1/chat/completions request; fills out with ttft, end, tokens, error."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), headers)
    t0 = time.time()
    out.update(start=t0, ttft=None, tokens=0, error=None)
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                        out["chunks"] = out.get("chunks", 0) + 1
                        if out["ttft"] is None:
                            out["ttft"] = time.time() - t0
                if ev.get("usage"):
                    out["tokens"] = ev["usage"].get("completion_tokens") or 0
    except Exception as e:
        out["error"] = str(e)[:200]
    out["end"] = time.time()
    if not out["tokens"]:                     # no usage in the stream: count the text chunks instead
        out["tokens"] = out.get("chunks", 0)


def workload(name, c, i, rnd):
    if name == "short":
        return {"messages": [{"role": "user", "content": TASKS[(i + rnd * c) % len(TASKS)]}], "max_tokens": 512,
                "temperature": 0, "reasoning_effort": "none"}
    _, _, body = corpus(["opencode-30k"])[0]
    msgs = [dict(m) for m in body["messages"]]
    nonce = f"[run {time.time_ns()} client {i}] "
    if isinstance(msgs[0].get("content"), str):
        msgs[0]["content"] = nonce + msgs[0]["content"]
    else:
        msgs.insert(0, {"role": "system", "content": nonce})
    return dict(body, messages=msgs, max_tokens=256, temperature=0)


def run_level(port, key, name, c, rnd):
    outs = [{} for _ in range(c)]
    bodies = [workload(name, c, i, rnd) for i in range(c)]
    ths = [threading.Thread(target=stream, args=(port, key, bodies[i], outs[i])) for i in range(c)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    tot = sum(o["tokens"] for o in outs)
    per = []
    for o in outs:
        dec = (o["tokens"] - 1) / (o["end"] - o["start"] - o["ttft"]) if o["ttft"] and o["tokens"] > 1 else None
        per.append({"ttft_s": round(o["ttft"], 2) if o["ttft"] else None, "tokens": o["tokens"],
                    "decode_tps": round(dec, 1) if dec else None, "error": o["error"]})
    return {"workload": name, "c": c, "round": rnd, "wall_s": round(wall, 1), "total_tokens": tot,
            "total_tps": round(tot / wall, 1), "clients": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--levels", default="1,2,4")
    ap.add_argument("--workloads", default="short,long")
    ap.add_argument("--rounds", type=int, default=2, help="rounds per level of the short workload")
    ap.add_argument("--long-rounds", type=int, default=1, help="rounds per level of the long workload")
    ap.add_argument("--no-start", action="store_true", help="use a server that is already running")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    port, key = cfg["port"], cfg.get("api_key")
    run = HERE / "runs" / a.tag
    run.mkdir(parents=True, exist_ok=True)
    srv = None
    if not a.no_start:
        repo = Path(cfg.get("cwd") or HERE.parent / "eddoursul")
        log = open(run / "server.log", "w")
        srv = subprocess.Popen([str(repo / ".venv/bin/python"), "serve/server.py", "--engine", "strata",
                                "--config", str(Path(a.config).resolve()), "--port", str(port)], stdout=log, stderr=subprocess.STDOUT,
                               cwd=repo)
    results = {"tag": a.tag, "config": a.config, "parallel": cfg.get("parallel", 1), "rounds": []}
    t0 = time.time()
    try:
        while True:
            if srv and srv.poll() is not None:
                sys.exit(f"server exited with {srv.returncode}; see {run / 'server.log'}")
            try:
                h = urllib.request.Request(f"http://127.0.0.1:{port}/health")
                if json.loads(urllib.request.urlopen(h, timeout=5).read()).get("loaded"):
                    break
            except Exception:
                pass
            if time.time() - t0 > 1200:
                sys.exit("not ready after 1200 s")
            time.sleep(2)
        warm = {}
        stream(port, key, {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4, "temperature": 0,
                           "reasoning_effort": "none"}, warm)
        results["ready_s"] = round(time.time() - t0)
        print(f"[{a.tag}] ready after {results['ready_s']} s (parallel {results['parallel']})", flush=True)
        for name in a.workloads.split(","):
            for c in [int(x) for x in a.levels.split(",")]:
                for rnd in range(a.rounds if name == "short" else a.long_rounds):
                    r = run_level(port, key, name, c, rnd)
                    results["rounds"].append(r)
                    ttft = [x["ttft_s"] for x in r["clients"]]
                    dec = [x["decode_tps"] for x in r["clients"]]
                    errs = [x["error"] for x in r["clients"] if x["error"]]
                    print(f"[{a.tag}] {name:5s} C={c} round {rnd}: wall {r['wall_s']} s, total {r['total_tps']} tok/s"
                          f" | ttft {ttft} | per-client decode {dec}{' ERR ' + str(errs) if errs else ''}",
                          flush=True)
    finally:
        (run / "load.json").write_text(json.dumps(results, indent=1))
        if srv:
            srv.terminate()
            try:
                srv.wait(90)
            except subprocess.TimeoutExpired:
                srv.kill()


if __name__ == "__main__":
    main()
