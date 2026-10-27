"""GET /v1/prefill_progress — the vLLM fork's prefill-progress contract, adapted to Strata (API-PARITY-PLAN 3).

Answers "has my request been admitted, and how much of its prompt is left?" while a long prefill runs —
the OpenAI stream is silent until the first token, so a client cannot see this any other way.

Strata has no KV disk tier, so there are no restore fields: the keys the vLLM payload carries for
restore state (restoring_kv, restore_ms) are OMITTED here, not zeroed, so a consumer that distinguishes
restore latency keeps working on vLLM and simply skips them on Strata.

The state comes from Strata's own record, read under status_lock: Service.terminal_requests (the queued ->
admitted/waiting -> first_token rows, serve/server.py) plus the engine's slot snapshot (slots_view, what
/slots and /metrics use).  A request is "prefilling" once admitted while its first_token event has not
fired; the per-request progress inside the prefill (prompt_read of prompt_tokens) comes from the slot row.

payload() never raises: on any unexpected shape it returns the empty one, so the route cannot 500.
"""
from __future__ import annotations

import time

# Read at import, like the vLLM module: the route registration in serve/server.py additionally re-reads
# the env per request, so flipping it needs no restart of the Python side.
ENABLED = __import__("os").environ.get("PREFILL_PROGRESS", "0") == "1"


def _ms(since: float | None, now: float) -> int | None:
    return None if since is None else int((now - since) * 1000)


def _phase(row: dict) -> str:
    """vLLM's phase names from a terminal_requests row: a row that got its first_token event is decoding;
    one admitted (or reading its prompt) but silent is prefilling; queued, or waiting for a batch slot
    (Strata's `waiting`: admitted_at is stamped but the engine has not taken the prompt), is queued."""
    if row.get("first_engine_token_s") is not None:
        return "decode"
    if row.get("state") in ("admitted", "reading"):
        return "prefill"
    return "queued"


def payload(svc) -> dict:
    """The JSON body for GET /v1/prefill_progress, from a live Service (duck-typed: status_lock and
    terminal_requests are all a mock needs). Per-request keys follow handler_payload()'s schema in the
    vLLM fork's prefill_progress.py, minus the disk-restore fields (absent, not zero)."""
    empty = {"ts": time.time(), "running": None, "waiting": None, "kv_cache_pct": None, "requests": []}
    try:
        now = time.time()
        with svc.status_lock:
            rows = sorted(svc.terminal_requests.values(), key=lambda r: r.get("queued_at") or 0.0)
            rows = [dict(r) for r in rows]
            s = dict(svc.status)
            live_reqs = len(svc.live_reqs)
        # the engine's slot snapshot: per-slot prompt_read/generated for the requests inside a slot
        slots = []
        view = getattr(svc.engine, "slots_view", None)
        if callable(view):
            try:
                slots = view() or []
            except Exception:
                slots = []
        by_phase_progress = [x for x in slots if isinstance(x, dict) and x.get("state") == "reading"]

        # position: the queue order the admission path will serve (queued_at order among not-yet-prefilling rows)
        position, seen = {}, 0
        for r in rows:
            if _phase(r) == "queued":
                position[r["id"]] = seen
                seen += 1

        reqs = []
        for r in rows:
            phase = _phase(r)
            prompt = r.get("prompt_tokens")
            req = {
                "request_id": r["id"],
                "phase": phase,
                "state": r.get("state"),
                "admitted": r.get("admitted_at") is not None,
                "arrived_ms": _ms(r.get("queued_at"), now),
                "admitted_ms": _ms(r.get("admitted_at"), now),
                "prompt_tokens": prompt,
                "prefill_started_at": r.get("admitted_at") if phase in ("prefill", "decode") else None,
                "first_token_at": (round(r["admitted_at"] + r["first_engine_token_s"], 3)
                                   if r.get("admitted_at") and r.get("first_engine_token_s") is not None else None),
                "computed_tokens": None,
                "cached_tokens": None,
                "remaining_prompt_tokens": None,
                "generated_tokens": None,
                "preemptions": 0,
            }
            if phase == "queued":
                req["position"] = position.get(r["id"], 0)
            elif phase == "prefill":
                # the slot row for a reading slot, in admission order: how much of the prompt is read
                if by_phase_progress:
                    x = by_phase_progress.pop(0)
                    read = x.get("prompt_read")
                    req["computed_tokens"] = read
                    req["remaining_prompt_tokens"] = (max(0, prompt - read)
                                                      if isinstance(prompt, int) and isinstance(read, int) else None)
            else:                                             # decode
                for x in slots:
                    if isinstance(x, dict) and x.get("state") == "generating":
                        req["generated_tokens"] = x.get("generated")
                        break
            reqs.append(req)

        running = live_reqs or int(bool(s.get("busy")))
        waiting = int(s.get("queued") or 0) + int(getattr(svc.engine, "waiting", 0) or 0)
        ctx = int((getattr(svc.engine, "info", None) or {}).get("max_context") or svc.reported_ctx()) \
            if hasattr(svc, "reported_ctx") else 0
        used = (int(s.get("prompt_tokens") or 0) + int(s.get("generated") or 0)) if s.get("busy") else 0
        return {
            "ts": now,
            "running": running,
            "waiting": waiting,
            "kv_cache_pct": round(min(1.0, used / ctx) * 100, 1) if ctx else None,
            "requests": reqs,
        }
    except Exception:
        return empty
