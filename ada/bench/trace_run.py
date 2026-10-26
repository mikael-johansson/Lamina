#!/usr/bin/env python3
"""Routing traces from real requests: the fork's engine with --dump-routing and STRATA_DECODE_TIMING=1.

    python3 bench/trace_run.py configs/fork-iq3_s-tuned.json --tag trace-iq3

Replays bench/trace_requests.json (real logged requests, their own sampling, at most 512 tokens out) one after the
other and saves, per request, the server's last_timings (predicted_n, draft_n, draft_n_accepted: the trace's rows
per request are windows + drafts) into bench/runs/<tag>/requests.json; the trace is bench/runs/<tag>/routing.bin and
the engine log has one `strata decode timing:` line per request.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402
from extract_requests import read_request  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    run = HERE / "runs" / a.tag
    run.mkdir(parents=True, exist_ok=True)
    trace = run / "routing.bin"
    cfg["args"] = cfg["args"] + ["--dump-routing", str(trace)]
    cfg["log"] = str(run / "engine.log")
    cfg.pop("vision", None)
    cfg["args"] = [x for x in cfg["args"] if x != "--vision"]
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    (run / "config.json").chmod(0o600)
    bench.API_KEY = cfg.get("api_key")
    port = cfg["port"]
    repo = Path(cfg.get("cwd") or HERE.parent / "eddoursul")
    env = dict(__import__("os").environ, STRATA_DECODE_TIMING="1")
    log = open(run / "server.log", "w")
    srv = subprocess.Popen([str(repo / ".venv/bin/python"), "serve/server.py", "--engine", "strata", "--config",
                            str(run / "config.json"), "--port", str(port)], stdout=log, stderr=subprocess.STDOUT,
                           cwd=repo, env=env)
    out = []
    t0 = time.time()
    try:
        while True:
            if srv.poll() is not None:
                sys.exit(f"server exited with {srv.returncode}; see {run / 'server.log'}")
            try:
                if bench.call(port, "/health", timeout=5).get("loaded"):
                    break
            except Exception:
                pass
            if time.time() - t0 > 1200:
                sys.exit("not ready after 1200 s")
            time.sleep(2)
        print(f"[{a.tag}] ready after {time.time() - t0:.0f} s", flush=True)
        reqs = json.loads((HERE / "trace_requests.json").read_text())
        for i, r in enumerate(reqs):
            _, body = read_request(bench.LOGS / r["file"])
            for k in ("stream", "stream_options", "max_completion_tokens"):
                body.pop(k, None)
            body["max_tokens"] = min(int(body.get("max_tokens") or a.max_tokens), a.max_tokens)
            body["model"] = "strata"
            t = time.time()
            try:
                bench.call(r["path"] == "/v1/messages" and port or port, r["path"], body)
                err = None
            except Exception as e:
                err = str(e)[:200]
            st = bench.call(port, "/v1/status").get("last_timings") or {}
            row = {"i": i, "group": r["group"], "file": r["file"], "wall_s": round(time.time() - t, 1), "error": err,
                   **{k: st.get(k) for k in ("prompt_n", "cache_n", "predicted_n", "draft_n", "draft_n_accepted",
                                             "prompt_per_second", "predicted_per_second")}}
            out.append(row)
            print(f"[{a.tag}] {i:2d} {r['group']:6s} prompt {row['prompt_n']} gen {row['predicted_n']} at "
                  f"{row['predicted_per_second']} tok/s, drafts {row['draft_n']}/{row['draft_n_accepted']}"
                  f"{' ERR ' + err if err else ''}", flush=True)
    finally:
        (run / "requests.json").write_text(json.dumps(out, indent=1))
        srv.terminate()
        try:
            srv.wait(120)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    main()
