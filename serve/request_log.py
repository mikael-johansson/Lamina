"""Per-request log files, in the format of the vLLM setup's request logging (~/git/vllm/qwen38-27b-rtx3090/
request_logging.py), so the same tools read both (bench/extract_requests.py, bench/replay.py).

One file per request: <UTC time>_<id>_<the transcript's first words>.log, with the request's headers (the API keys
left out), its body, the response as it is generated (a hung or cancelled request still leaves its part) and a
performance footer from the server's own clock and the engine's timings.  On when the config's "request_log_dir"
(or REQUEST_LOG_DIR) names a directory.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]+")
_SECRET_HEADERS = ("authorization", "x-api-key", "api-key", "cookie", "proxy-authorization")

_open_lock = threading.Lock()
_open = 0   # requests with a log open: the concurrency the footer shows


def _sanitize(text: str, max_len: int) -> str:
    return _SANITIZE_RE.sub("_", text[: max_len * 4]).strip("_-.")[:max_len]


def _stub(text: str, n_words: int = 10, max_len: int = 80) -> str:
    parts = [s for s in (_sanitize(w, max_len) for w in text.split()[:n_words]) if s]
    return "_".join(parts)[:max_len] or "prompt"


def render_transcript(messages) -> str:
    """SYSTEM:/USER:/ASSISTANT:/TOOL: blocks, as the vLLM logs' file names start from."""
    lines = []
    for msg in messages if isinstance(messages, list) else []:
        if not isinstance(msg, dict):
            continue
        lines.append(f"{(msg.get('role') or 'unknown').upper()}:")
        content = msg.get("content")
        if isinstance(content, str):
            lines.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    lines.append(part.get("text", "") if part.get("type") == "text"
                                 else f"[attachment: {part.get('type', '')}]")
        lines.append("")
    return "\n".join(lines)


class RequestLog:
    def __init__(self, directory: str, path: str, req: dict, headers, client: str | None):
        global _open
        kind = {"/v1/chat/completions": "chatcmpl", "/v1/messages": "msg", "/v1/responses": "resp"}.get(path, "req")
        self.request_id = f"{kind}-{uuid.uuid4().hex[:16]}"
        messages = req.get("messages")
        if messages is None and isinstance(req.get("input"), list):   # /v1/responses
            messages = req.get("input")
        prompt = render_transcript(messages)
        if isinstance(req.get("system"), str):                          # /v1/messages
            prompt = f"SYSTEM:\n{req['system']}\n" + prompt
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        base = f"{ts}_{self.request_id}_{_stub(prompt)}"
        self.path = d / f"{base}.log"
        self._t0 = time.monotonic()
        self._t_first = None
        self._mode = None   # "reasoning" / "content": the part being written
        self._done = False
        with _open_lock:
            self.concurrent_at_arrival = _open
            _open += 1
        self.concurrent_at_first = None
        try:
            self._fh = open(self.path, "w", encoding="utf-8")
        except OSError:
            self._fh = None
            return
        w = self._fh
        w.write("\n=== STRATA REQUEST LOG ===\n")
        w.write(f"timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-4]}Z\n")
        w.write(f"request_id: {self.request_id}\n")
        w.write(f"client: {client or 'unknown'}\n")
        w.write(f"method_path: POST {path}\n")
        w.write(f"model: {req.get('model')}\n")
        w.write(f"stream: {bool(req.get('stream'))}\n")
        w.write("\n=== HEADERS ===\n")
        for k, v in (headers.items() if headers is not None else []):
            if k.lower() not in _SECRET_HEADERS:
                w.write(f"{k.lower()}: {v}\n")
        w.write("\n=== REQUEST ===\n")
        w.write(json.dumps(req, ensure_ascii=False, indent=2) + "\n")
        w.flush()

    def write(self, reasoning: str, content: str) -> None:
        if self._fh is None or self._done or not (reasoning or content):
            return
        if self._t_first is None:
            self._t_first = time.monotonic()
            with _open_lock:
                self.concurrent_at_first = _open - 1
            self._fh.write("\n=== RESPONSE (streaming) ===\n")
        for mode, text in (("reasoning", reasoning), ("content", content)):
            if not text:
                continue
            if mode != self._mode:
                if mode == "reasoning":
                    self._fh.write("<think>\n")
                elif self._mode == "reasoning":
                    self._fh.write("\n</think>\n\n")
                self._mode = mode
            self._fh.write(text)
        self._fh.flush()

    def finish(self, record: dict) -> None:
        """`record`: the server's request record (state, queue_s, usage, timings, http_status, error...)."""
        global _open
        if self._done:
            return
        self._done = True
        with _open_lock:
            _open -= 1
        if self._fh is None:
            return
        w = self._fh
        if self._mode == "reasoning":
            w.write("\n</think>\n")
        state = record.get("state") or record.get("outcome")
        err = record.get("error")
        if err:
            w.write("\n=== ERROR ===\n")
            w.write((json.dumps(err, ensure_ascii=False) if not isinstance(err, str) else err) + "\n")
            if record.get("http_status") is not None:
                w.write(f"http_status: {record['http_status']}\n")
        elif state not in (None, "completed", "generating"):
            w.write(f"\n=== {str(state).upper()} ===\n")
        if record.get("tool_calls"):
            w.write("\n=== TOOL CALLS ===\n")
            w.write(json.dumps(record["tool_calls"], ensure_ascii=False, indent=2) + "\n")
        rows = []
        usage = record.get("usage") or {}
        t = record.get("timings") or {}
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        cached = t.get("cache_n", (usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        if pt is not None:
            rows.append(("Prompt tokens", str(pt)))
        if ct is not None:
            rows.append(("Completion tokens", str(ct)))
        if cached is not None:
            rows.append(("Cached tokens", f"{cached}" + (f" ({100.0 * cached / pt:.1f}%)" if pt else "")))
        if t.get("prompt_per_second") is not None:
            rows.append(("Prefill speed (PP)", f"{t['prompt_per_second']:.2f} tok/s"))
        if t.get("predicted_per_second") is not None:
            rows.append(("Generation speed (TG)", f"{t['predicted_per_second']:.2f} tok/s"))
        total = time.monotonic() - self._t0
        queue = record.get("queue_s")
        if self._t_first is not None:
            rows.append(("Time to first token", f"{self._t_first - self._t0:.3f} s"))
        if t.get("prompt_ms") is not None:
            rows.append(("Prefill duration", f"{t['prompt_ms'] / 1000:.3f} s"))
        if t.get("predicted_ms") is not None:
            rows.append(("Generation duration", f"{t['predicted_ms'] / 1000:.3f} s"))
        if self._t_first is not None:
            rows.append(("First to last token", f"{time.monotonic() - self._t_first:.3f} s"))
        rows.append(("Total duration", f"{total:.3f} s"))
        if queue is not None:
            rows.append(("Queue wait", f"{queue:.3f} s"))
        if t.get("draft_n") is not None:
            rows.append(("Drafts accepted", f"{t.get('draft_n_accepted', 0)} of {t['draft_n']}"))
        rows.append(("Concurrent @ arrival", f"{self.concurrent_at_arrival} other request(s)"))
        if self.concurrent_at_first is not None:
            rows.append(("Concurrent @ first token", f"{self.concurrent_at_first} other request(s)"))
        if record.get("finish_reason"):
            rows.append(("Finish reason", str(record["finish_reason"])))
        rows.append(("Outcome", str(state or "completed")))
        w.write("\n=== PERFORMANCE ===\n")
        width = max(len(r[0]) for r in rows)
        for name, value in rows:
            w.write(f"{name.ljust(width)} : {value}\n")
        try:
            w.close()
        except OSError:
            pass
        self._fh = None
