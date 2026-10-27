#!/usr/bin/env python3
"""Replay logged Hermes traffic against a Strata server at its recorded pace: a stress / soak test.

    python3 bench/replay.py --config configs/fork-iq3_s-split-batch.json --from 2026-10-12T05:00 --minutes 30 \
        --tag soak1 [--speed 1.0] [--max-tokens 0] [--kinds main,side] [--stall 60] [--start]

The requests are the vLLM request logs' Hermes ones (bench/extract_requests.LOGS; only the === REQUEST === bodies are
read, never the headers): the agent's own turns ("main": the Hermes Agent system prompt) and its side calls ("side":
fact extraction, JSON-only).  Each starts at its logged offset from --from, divided by --speed, so the sessions'
overlap and their growing, mostly cached contexts are the real ones.  Every request streams; per request: time to the
first token, the longest gap between chunks, its rate, its finish, its error.  A watchdog flags a request silent for
--stall seconds and records then what the engine was doing (its log's tail, the GPUs, its busiest thread); the
engine's restarts are counted from the server log.  With --start the server is started from the config (and stopped
at the end); its port must not be one a live client uses.  Results: bench/runs/<tag>/replay.json and a summary.
"""
import argparse, json, os, re, subprocess, sys, threading, time, urllib.request, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_requests import LOGS, read_request  # noqa: E402

KINDS = {"main": ("Hermes_Agent",), "side": ("Extract_SIGNIFICANT_facts", "Return_valid_json")}
STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d)Z_chatcmpl")


def pick(t0, minutes, kinds):
    t1 = t0 + datetime.timedelta(minutes=minutes)
    keys = [k for kind in kinds for k in KINDS[kind]]
    out = []
    for f in os.listdir(LOGS):
        m = STAMP.match(f)
        if not m or not any(k in f for k in keys):
            continue
        t = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H-%M-%S")
        if t0 <= t < t1:
            out.append((t, f))
    out.sort()
    return out


def has_image(body):
    for m in body.get("messages", []):
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") in ("image_url", "image") for p in c):
            return True
    return False


class Run:
    def __init__(self, a, cfg):
        self.a, self.cfg = a, cfg
        self.port = cfg["port"]
        self.key = cfg.get("api_key")
        self.lock = threading.Lock()
        self.live = {}          # id -> record of a request in flight
        self.done = []
        self.stalls = []
        self.t_start = None

    def request(self, rid, name, t_off, body):
        rec = {"id": rid, "file": name, "offset_s": round(t_off, 1), "start": None, "ttft_s": None, "tokens": 0,
               "max_gap_s": 0.0, "wall_s": None, "finish": None, "error": None, "prompt_chars": len(json.dumps(body))}
        body = dict(body)
        body["stream"] = True
        body.pop("stream_options", None)
        cap = self.a.max_tokens
        if cap > 0:  # 0: the budgets as logged
            for k in ("max_tokens", "max_completion_tokens"):
                if k in body and body[k]:
                    body[k] = min(int(body[k]), cap)
            if not body.get("max_tokens") and not body.get("max_completion_tokens"):
                body["max_tokens"] = cap
        body.pop("model", None)
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        t0 = time.time()
        rec["start"] = round(t0 - self.t_start, 1)
        rec["last"] = t0
        with self.lock:
            self.live[rid] = rec
        try:
            with urllib.request.urlopen(req, timeout=self.a.timeout) as r:
                for raw in r:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    now = time.time()
                    try:
                        ev = json.loads(data)
                    except ValueError:
                        continue
                    ch = (ev.get("choices") or [{}])[0]
                    delta = ch.get("delta") or ch.get("message") or {}   # a buffered (structured) answer: one message
                    got = bool(delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                               or delta.get("tool_calls"))
                    if got:
                        if rec["ttft_s"] is None:
                            rec["ttft_s"] = round(now - t0, 2)
                        elif now - rec["last"] > rec["max_gap_s"]:
                            rec["max_gap_s"] = round(now - rec["last"], 2)
                            rec["max_gap_end"] = time.strftime("%H:%M:%S", time.localtime(now))   # (the logs' clock)
                        rec["tokens"] += 1
                        rec["last"] = now
                    if ch.get("finish_reason"):
                        rec["finish"] = ch["finish_reason"]
                    if ev.get("usage") and ev["usage"].get("completion_tokens"):
                        rec["tokens"] = ev["usage"]["completion_tokens"]
        except Exception as e:  # noqa: BLE001 - every failure is a result here
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["wall_s"] = round(time.time() - t0, 2)
        if rec["ttft_s"] is not None and rec["wall_s"] > rec["ttft_s"]:
            rec["tok_s"] = round(rec["tokens"] / (rec["wall_s"] - rec["ttft_s"]), 1)
        rec.pop("last", None)
        with self.lock:
            self.live.pop(rid, None)
            self.done.append(rec)

    def diagnose(self):
        """What the engine is doing right now (a stall): its log's tail, the GPUs, its threads' states."""
        d = {"t": round(time.time() - self.t_start, 1)}
        log = self.cfg.get("log")
        try:
            d["engine_log"] = Path(log).read_text(errors="replace").splitlines()[-12:] if log else None
        except OSError:
            d["engine_log"] = None
        try:
            d["gpus"] = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                                        "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            d["gpus"] = None
        exe = Path(self.cfg["exe"]).name
        pid = subprocess.run(["pgrep", "-f", f"{exe} --serve"], capture_output=True, text=True).stdout.split()
        if pid:
            states = []
            for t in Path(f"/proc/{pid[0]}/task").iterdir():
                try:
                    st = (t / "stat").read_text().split()
                    states.append((int(st[13]) + int(st[14]), st[2], (t / "wchan").read_text().strip(), t.name))
                except OSError:
                    pass
            states.sort(reverse=True)
            d["engine_threads"] = [f"{tid} {s} {w} cpu={c}" for c, s, w, tid in states[:6]]
        return d

    def watchdog(self, stop):
        flagged = set()
        while not stop.is_set():
            time.sleep(5)
            now = time.time()
            # a stall: silent after its first chunk; a request still waiting for its first one (queued behind the
            # batch slots, or its prompt being read) only after --first-wait
            with self.lock:
                silent = [(rid, now - r["last"], r["ttft_s"] is None) for rid, r in self.live.items()
                          if now - r["last"] > (self.a.first_wait if r["ttft_s"] is None else self.a.stall)]
            for rid, s, first in silent:
                if rid in flagged:
                    continue
                flagged.add(rid)
                d = self.diagnose()
                d.update({"request": rid, "silent_s": round(s, 1), "kind": "no first token" if first else "stall"})
                with self.lock:
                    self.stalls.append(d)
                    n_live = len(self.live)
                print(f"[replay] {d['kind'].upper()}: request {rid} silent {s:.0f} s ({n_live} in flight); "
                      f"GPUs {d['gpus']!r}",
                      flush=True)


def server_restarts(path):
    try:
        return Path(path).read_text(errors="replace").count("the engine had stopped")
    except OSError:
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--from", dest="t0", required=True, help="the window's start in the logs, e.g. 2026-10-12T05:00")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--speed", type=float, default=1.0, help="replay pace: 2 = twice as fast as recorded")
    ap.add_argument("--kinds", default="main,side")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--stall", type=float, default=60, help="seconds a request may stay silent before it is flagged")
    ap.add_argument("--first-wait", type=float, default=600, help="seconds a request may wait for its first chunk")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--limit", type=int, default=0, help="at most this many requests (0: all in the window)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--start", action="store_true", help="start the server from the config, stop it at the end")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if cfg.get("port") == 8091:
        sys.exit("replay: the config's port is 8091, where the live clients connect; use another")
    run_dir = HERE / "runs" / a.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = datetime.datetime.strptime(a.t0, "%Y-%m-%dT%H:%M")
    picked = pick(t0, a.minutes, a.kinds.split(","))
    reqs = []
    for t, f in picked:
        try:
            _, body = read_request(LOGS / f)
        except Exception:  # noqa: BLE001
            continue
        if not body or "messages" not in body or has_image(body):
            continue
        reqs.append(((t - t0).total_seconds() / a.speed, f, body))
    if a.limit:
        reqs = reqs[:a.limit]
    if reqs:   # the clock starts at the window's first request, not at --from
        first = reqs[0][0]
        reqs = [(off - first, f, body) for off, f, body in reqs]
    print(f"[replay] {len(reqs)} requests in {a.minutes:g} min from {a.t0} (x{a.speed:g}: "
          f"{reqs[-1][0] / 60 if reqs else 0:.1f} min of arrivals)", flush=True)
    srv = None
    server_log = run_dir / "server.log"
    if a.start:
        repo = HERE.parent / "eddoursul"
        cfg_run = dict(cfg)
        (run_dir / "config.json").write_text(json.dumps(cfg_run, indent=1))
        srv = subprocess.Popen([str(repo / ".venv/bin/python"), str(repo / "serve/server.py"), "--engine", "strata",
                                "--config", str(run_dir / "config.json")], stdout=open(server_log, "w"),
                               stderr=subprocess.STDOUT, cwd=repo)
        for _ in range(600):
            if "ready:" in server_log.read_text(errors="replace"):
                break
            if srv.poll() is not None:
                sys.exit("replay: the server exited while starting")
            time.sleep(1)
        print("[replay] server ready", flush=True)
    r = Run(a, cfg)
    r.t_start = time.time()
    stop = threading.Event()
    wd = threading.Thread(target=r.watchdog, args=(stop,), daemon=True)
    wd.start()
    threads = []
    for i, (off, f, body) in enumerate(reqs):
        delay = r.t_start + off - time.time()
        if delay > 0:
            time.sleep(delay)
        th = threading.Thread(target=r.request, args=(i, f, off, body), daemon=True)
        th.start()
        threads.append(th)
        if i % 20 == 0:
            with r.lock:
                print(f"[replay] {i}/{len(reqs)} sent at {time.time() - r.t_start:.0f} s, {len(r.live)} in flight, "
                      f"{len(r.done)} done, {len(r.stalls)} stalls", flush=True)
    for th in threads:
        th.join()
    stop.set()
    restarts = server_restarts(server_log) if a.start else None
    if srv is not None:
        srv.terminate()
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            srv.kill()
    done = sorted(r.done, key=lambda x: x["id"])
    ok = [x for x in done if not x["error"]]

    def pct(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(p * len(v)))] if v else None
    wall = time.time() - r.t_start
    summary = {"tag": a.tag, "window": a.t0, "minutes": a.minutes, "speed": a.speed, "requests": len(done),
               "errors": len(done) - len(ok), "stalls": len(r.stalls), "engine_restarts": restarts,
               "wall_s": round(wall, 1), "tokens": sum(x["tokens"] for x in ok),
               "ttft_p50": pct([x["ttft_s"] for x in ok if x["ttft_s"] is not None], 0.5),
               "ttft_p90": pct([x["ttft_s"] for x in ok if x["ttft_s"] is not None], 0.9),
               "tok_s_p50": pct([x["tok_s"] for x in ok if x.get("tok_s")], 0.5),
               "max_gap_p99": pct([x["max_gap_s"] for x in ok], 0.99),
               "error_kinds": sorted({(x["error"] or "")[:60] for x in done if x["error"]})}
    summary["throughput_tok_s"] = round(summary["tokens"] / max(1.0, wall), 1)
    (run_dir / "replay.json").write_text(json.dumps({"summary": summary, "requests": done, "stalls": r.stalls},
                                                    indent=1))
    print("[replay] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
