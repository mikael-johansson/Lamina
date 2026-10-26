#!/usr/bin/env python3
"""Start a Strata server from a config with extra/overridden engine args, replay a corpus of real logged requests
plus two short synthetic ones, record prefill/decode speed and the expert tiers per request, stop the server.

    ../eddoursul/.venv/bin/python bench.py ../configs/fork-iq3_s.json --tag base [--set --pcie-frac=0.15] [--add --arena-place]
"""
import argparse, json, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_requests import LOGS, read_request

SYNTH = [
    ("code", "/v1/chat/completions", {"messages": [{"role": "user", "content":
        "Write a Python function that parses an ISO 8601 duration string like 'P3DT4H5M' into total seconds, "
        "with a few doctests."}], "max_tokens": 600, "reasoning_effort": "none"}),
    ("thinking", "/v1/chat/completions", {"messages": [{"role": "user", "content":
        "Prove that there are infinitely many primes of the form 4k+3, then give the five smallest."}],
        "max_tokens": 2000, "reasoning_effort": "high"}),
]


API_KEY = None   # the config's api_key, set in main()


def call(port, path, payload=None, timeout=1800):
    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(payload).encode() if payload is not None else None,
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def apply_args(args, sets, adds, drops):
    args = list(args)
    for s in sets:                                    # --flag=value: replace the value (or append)
        k, v = s.split("=", 1)
        if k in args:
            args[args.index(k) + 1] = v
        else:
            args += [k, v]
    for d in drops:                                   # --flag or --flag=1 (takes a value)
        k, _, n = d.partition("=")
        if k in args:
            i = args.index(k)
            del args[i:i + 1 + int(n or 0)]
    return args + [x for a in adds for x in a.split()]


def corpus(names):
    sel = json.loads((HERE / "corpus.json").read_text())
    out = []
    for name in names:
        r = sel[name]
        _, body = read_request(LOGS / r["file"])
        body.pop("stream", None); body.pop("stream_options", None)
        body["temperature"] = 0
        for k in ("top_p", "top_k"):
            body.pop(k, None)
        body["max_tokens"] = min(int(body.get("max_tokens") or 1024), 1024)
        body.pop("max_completion_tokens", None)
        out.append((name, r["path"], body))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--drop", action="append", default=[])
    ap.add_argument("--env", action="append", default=[], help="NAME=value for the engine")
    ap.add_argument("--only", default="code,thinking,py-small,opencode-30k,cc-40k,cc-80k,cc-80k-next,cc-150k")
    ap.add_argument("--repo", default=HERE.parent / "eddoursul")
    a = ap.parse_args()

    global API_KEY
    cfg = json.loads(Path(a.config).read_text())
    API_KEY = cfg.get("api_key")
    cfg["args"] = apply_args(cfg["args"], a.set, a.add, a.drop)
    if a.env:
        cfg.setdefault("env", {}).update(dict(e.split("=", 1) for e in a.env))
    cfg["log"] = str(Path(cfg["log"]).with_name(f"bench-{a.tag}.log"))
    run = HERE / "runs" / a.tag
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    (run / "config.json").chmod(0o600)   # it holds the api_key
    port = cfg["port"]
    repo = Path(a.repo)
    log = open(run / "server.log", "w")
    srv = subprocess.Popen([str(repo / ".venv/bin/python"), str(repo / "serve/server.py"), "--engine", "strata",
                            "--config", str(run / "config.json")], stdout=log, stderr=subprocess.STDOUT, cwd=repo)
    t0 = time.time()
    results = {"tag": a.tag, "args": cfg["args"], "env": cfg.get("env"), "requests": []}
    try:
        while True:
            if srv.poll() is not None:
                sys.exit(f"server exited with {srv.returncode}; see {log.name}")
            try:
                if call(port, "/health", timeout=5).get("loaded"):
                    break
            except Exception:
                pass
            if time.time() - t0 > 1200:
                sys.exit("not ready after 1200 s")
            time.sleep(2)
        # the first request finishes the load; time it with a tiny one
        call(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4,
                                            "temperature": 0, "reasoning_effort": "none"})
        results["ready_s"] = round(time.time() - t0)
        print(f"[{a.tag}] ready after {results['ready_s']} s", flush=True)
        names = a.only.split(",")
        reqs = [s for s in SYNTH if s[0] in names] + corpus([n for n in names if n not in dict((s[0], 1) for s in SYNTH)])
        for name, path, body in reqs:
            body = dict(body, model="strata")
            if path == "/v1/chat/completions" and "temperature" not in body:
                body["temperature"] = 0
            t = time.time()
            try:
                r = call(port, path, body)
                err = None
            except Exception as e:
                r, err = {}, str(e)[:200]
            wall = time.time() - t
            st = call(port, "/v1/status").get("last_timings") or {}
            if path == "/v1/messages":
                text = "".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text")
                tools = [c["name"] for c in r.get("content", []) if c.get("type") == "tool_use"]
            else:
                m = (r.get("choices") or [{}])[0].get("message") or {}
                text = m.get("content") or ""
                tools = [t["function"]["name"] for t in m.get("tool_calls") or []]
            row = {"name": name, "wall_s": round(wall, 1), "error": err, "prompt_n": st.get("prompt_n"),
                   "cache_n": st.get("cache_n"), "prefill_tps": st.get("prompt_per_second"),
                   "prompt_s": round((st.get("prompt_ms") or 0) / 1000, 1), "gen_n": st.get("predicted_n"),
                   "decode_tps": st.get("predicted_per_second"), "draft": [st.get("draft_n"), st.get("draft_n_accepted")],
                   "tiers": st.get("expert_tiers"), "tools": tools, "text": text[:300]}
            results["requests"].append(row)
            print(f"[{a.tag}] {name:12s} prompt {row['prompt_n']} (+{row['cache_n']} cached) "
                  f"{row['prefill_tps']} tok/s in {row['prompt_s']} s | gen {row['gen_n']} at {row['decode_tps']} tok/s"
                  f" | tools {tools} {('ERR ' + err) if err else ''}", flush=True)
    finally:
        (run / "results.json").write_text(json.dumps(results, indent=1))
        srv.terminate()
        try:
            srv.wait(90)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    main()
