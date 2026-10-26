#!/usr/bin/env python3
"""Smoke test for a Strata server config: start serve/server.py, wait for /health, run a few greedy
requests through the OpenAI API, print speed and the answers' heads, stop the server.

    eddoursul/.venv/bin/python smoke.py configs/fork-iq3_s.json [--repo eddoursul] [--keep]
"""
import argparse, json, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

REQUESTS = [
    ("sanity", "What is 17 * 23? Reply with the number only.", False, 32),
    ("code", "Write a Python function that parses an ISO 8601 duration string like 'P3DT4H5M' "
             "into total seconds, with a few doctests.", False, 600),
    ("thinking", "A train leaves at 09:40 and arrives at 13:15, with two stops of 7 minutes each. "
                 "How long is it actually moving? Explain briefly.", True, 1500),
]


def call(port, path, payload=None, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--repo", default="eddoursul", help="checkout whose serve/server.py runs the config")
    ap.add_argument("--keep", action="store_true", help="leave the server running afterwards")
    ap.add_argument("--start-timeout", type=float, default=900)
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    port = cfg["port"]
    repo = HERE / a.repo
    log = open(HERE / f"smoke-{Path(a.config).stem}.server.log", "w")
    srv = subprocess.Popen([str(repo / ".venv/bin/python"), str(repo / "serve/server.py"), "--engine", "strata",
                            "--config", str(Path(a.config).resolve())], stdout=log, stderr=subprocess.STDOUT, cwd=repo)
    t0 = time.time()
    try:
        while True:
            if srv.poll() is not None:
                sys.exit(f"server exited with {srv.returncode}; see {log.name}")
            try:
                h = call(port, "/health", timeout=5)
                if h.get("loaded"):                 # "status" is "ok" before the engine has loaded
                    break
            except Exception:
                pass
            if time.time() - t0 > a.start_timeout:
                sys.exit(f"not ready after {a.start_timeout:.0f} s; see {log.name}")
            time.sleep(2)
        print(f"ready after {time.time() - t0:.0f} s")

        model = call(port, "/v1/models")["data"][0]["id"]
        for name, prompt, think, max_tokens in REQUESTS:
            body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                    "max_tokens": max_tokens, "reasoning_effort": "high" if think else "none"}
            t = time.time()
            r = call(port, "/v1/chat/completions", body)
            wall = time.time() - t
            msg = r["choices"][0]["message"]
            n = r.get("usage", {}).get("completion_tokens", 0)
            tm = r.get("timings") or {}
            speed = tm.get("predicted_per_second") or (n / wall if wall else 0)
            print(f"\n== {name}: {n} tokens, {speed:.1f} tok/s decode, {wall:.1f} s wall, "
                  f"prompt {tm.get('prompt_per_second', 0):.0f} tok/s")
            print((msg.get("content") or "").strip()[:400])
    finally:
        if not a.keep:
            srv.terminate()
            try:
                srv.wait(60)
            except subprocess.TimeoutExpired:
                srv.kill()


if __name__ == "__main__":
    main()
