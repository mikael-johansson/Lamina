#!/usr/bin/env python3
"""Index logged vLLM requests (~/git/vllm/qwen38-27b-rtx3090/requests): one JSON line per request with its API path,
size, tool count and client. Only the === REQUEST === body is read (the headers hold keys).

    python3 extract_requests.py [--last N] > index.jsonl
"""
import argparse, json, os, sys
from pathlib import Path

LOGS = Path.home() / "git/vllm/qwen38-27b-rtx3090/requests"


def read_request(path):
    text = path.read_text(errors="replace")
    head, _, rest = text.partition("=== REQUEST ===")
    if not rest:
        return None, None
    meta = dict(l.split(": ", 1) for l in head.splitlines() if ": " in l and not l.startswith(("authorization", "x-api-key")))
    body, _ = json.JSONDecoder().raw_decode(rest.lstrip())
    return meta, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=3000)
    a = ap.parse_args()
    files = sorted((f for f in os.listdir(LOGS) if f.endswith(".log") and not f.startswith("#")))[-a.last:]
    for f in files:
        try:
            meta, body = read_request(LOGS / f)
        except Exception:
            continue
        if not body or "messages" not in body:
            continue
        print(json.dumps({"file": f, "path": meta.get("method_path", "").split()[-1],
                          "agent": meta.get("user-agent", "")[:40], "chars": len(json.dumps(body)),
                          "tools": len(body.get("tools") or []), "msgs": len(body["messages"]),
                          "session": meta.get("x-claude-code-session-id") or "",
                          "cancelled": "=== CANCELLED ===" in (LOGS / f).read_text(errors="replace")[-3000:]}))


if __name__ == "__main__":
    main()
