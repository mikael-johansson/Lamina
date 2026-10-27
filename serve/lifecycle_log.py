"""serve/lifecycle_log.py - per-request console lifecycle lines
(API-PARITY-PLAN.md section 10), ported near-verbatim from the vLLM fork's
v1/engine/request_lifecycle_log.py (patches/request-lifecycle-log-timings.patch
and request-lifecycle-log-request-colors.patch), so one
`journalctl -o cat | grep 'chatcmpl-webui.0580615c'` works across both
backends:

    NEW REQUEST chatcmpl-webui.0580615c1c52-m87-3fa2c1-9e81: 83746 prompt tokens
    ADMITTED    chatcmpl-webui.0580615c1c52-m87-3fa2c1-9e81: 83746 prompt, 97.5% cached (cached 81746), 2000 to compute, waited 0.4s
    FINISHED    chatcmpl-webui.0580615c1c52-m87-3fa2c1-9e81: 83746 prompt, WAIT: 0.4s, PP: 7.3s (2000 tokens 273.9 PPtok/s) TTFT: 7.5s, GEN: 237 tokens in 2.7s (87.5 tok/s). E2E: 10.2s

NEW REQUEST prints as soon as the request enters the engine queue (prompt
tokens known from the tokenized ids); ADMITTED at the terminal_requests
queued -> admitted transition, which is where Strata can estimate the cache
hit (the engine's held prefix -- no RESUME line has arrived yet); FINISHED at
the request's DONE settlement, from the engine's own prompt_ms/decode_ms
figures -- errors, cancels and timeouts skip it, as on vLLM.

Strata differences (plan section 10.3): there are no VRAM/RAM/Disk KV tiers,
so the ADMITTED cache split is the prompt-cache (held-prefix) hit in light
green and the tokens to recompute in red, and the Disk segment is absent, not
zero -- the same convention as section 3's prefill_progress.

Color: VLLM_LOGGING_COLOR=1 (the same env name as the vLLM unit) turns it on
and "0" forces it off; unset follows the console (color only on a TTY).
ADMITTED is green, FINISHED light green (NEW REQUEST stays plain); the
FINISHED time split is dark cyan, its prefill rate dark green, its generation
figures light green, and WAIT turns brown above 0.1s. Each request id gets
one _REQ_COLORS color chosen by crc32 of its session (else "chatcmpl-<head>"),
so one conversation's requests -- main turns, aux calls, subagents -- share a
color, and the color agrees wherever the line is printed (a hash, not a
counter). VLLM_REQUEST_LIFECYCLE_LOG=0 silences the lines.
"""

import os
import sys
import zlib

from serve.session_request_id import session_root


def _use_color() -> bool:
    env = os.environ.get("VLLM_LOGGING_COLOR")
    if env is not None:
        return env == "1"
    try:                                    # unset: auto-off when not a TTY
        return bool(sys.stdout) and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


# Read once at import, like the vLLM module -- a deployment-level switch.
_ENABLED = os.environ.get("VLLM_REQUEST_LIFECYCLE_LOG", "1") != "0"
_COLOR = _use_color()

# Matches the vLLM fork's _REQ_ID_MAX (64, not 34): a session-tagged id fits.
_REQ_ID_MAX = 64

# ANSI colors (only when the log is colored).
_GREEN, _LIGHT_GREEN, _DARK_GREEN = "32", "92", "38;5;28"
_BROWN, _RED, _DARK_CYAN = "33", "91", "36"

# Per-request colors (256-color palette), alternating hue families: blues,
# cyans, purples, magentas -- outside the green/yellow/red status palette.
_REQ_COLORS = (
    "38;5;39",   # 1 sky blue        #00afff
    "38;5;141",  # 2 violet          #af87ff
    "38;5;51",   # 3 cyan            #00ffff
    "38;5;201",  # 4 magenta         #ff00ff
    "38;5;69",   # 5 cornflower blue #5f87ff
    "38;5;177",  # 6 orchid          #d787ff
    "38;5;44",   # 7 dark turquoise  #00d7d7
    "38;5;93",   # 8 purple          #8700ff
    "38;5;117",  # 9 light sky blue  #87d7ff
    "38;5;213",  # 10 pink violet    #ff87ff
    "38;5;33",   # 11 dodger blue    #0087ff
    "38;5;147",  # 12 lavender       #afafff
)


def _c(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _COLOR else text


def _short_req_id(req_id: str) -> str:
    if len(req_id) <= _REQ_ID_MAX:
        return req_id
    return req_id[:_REQ_ID_MAX] + "…"


def _req(req_id: str) -> str:
    """Short request id, in its per-request color."""
    short = _short_req_id(req_id)
    if not _COLOR:
        return short
    # Session, else "chatcmpl-<head>" -- the same in every process.
    base = session_root(req_id) or "-".join(req_id.split("-")[:2])
    return _c(_REQ_COLORS[zlib.crc32(base.encode()) % len(_REQ_COLORS)], short)


def lifecycle_log_enabled() -> bool:
    return _ENABLED


def log_new_request(req_id: str, num_prompt_tokens: int) -> None:
    """Emit one line as soon as the request enters the engine queue."""
    if not _ENABLED:
        return
    print(f"NEW REQUEST {_req(req_id)}: {num_prompt_tokens:d} prompt tokens", flush=True)


def log_admitted_request(req_id: str, num_prompt_tokens: int, num_cached_tokens: int,
                         waited_s: float) -> None:
    """Emit one line when the scheduler admits a request. Strata has no
    VRAM/RAM/Disk tiers: the cache split is the held-prefix hit (light green)
    and the tokens left to compute (red); the Disk segment is absent."""
    if not _ENABLED:
        return
    pct_cached = (num_cached_tokens / num_prompt_tokens * 100) if num_prompt_tokens else 0.0
    print(
        f"{_c(_GREEN, 'ADMITTED')} {_req(req_id)}: {num_prompt_tokens:d} prompt, {pct_cached:.1f}% cached "
        f"({_c(_LIGHT_GREEN, f'cached {num_cached_tokens:d}')}), "
        f"{_c(_RED, f'{max(0, num_prompt_tokens - num_cached_tokens):d} to compute')}, waited {waited_s:.1f}s",
        flush=True)


def log_finished_request(req_id: str, num_prompt_tokens: int, num_uncached_tokens: int,
                         prefill_time: float, ttft: float, num_generation_tokens: int,
                         decode_time: float, queue_time: float | None = None,
                         e2e_time: float | None = None) -> None:
    """Emit one line when a request finishes (all sampled, non-error finish
    reasons -- errors/aborts skip this, nothing to summarize).

    queue_time is queued -> admitted (Strata's slot turn), prefill_time the
    engine's prompt read, so WAIT + PP <= TTFT (TTFT also covers the serve
    layer's share: tokenization, transport)."""
    if not _ENABLED:
        return
    pp_rate = f"{num_uncached_tokens / prefill_time:.1f}" if prefill_time > 0 else "?"
    gen_rate = f"{num_generation_tokens / decode_time:.1f}" if decode_time > 0 else "?"
    wait = f"{queue_time:.1f}s" if queue_time is not None else "?"
    if queue_time is not None and queue_time > 0.1:
        wait = _c(_BROWN, wait)
    e2e = f"{e2e_time:.1f}s" if e2e_time is not None else "?"
    print(
        f"{_c(_LIGHT_GREEN, 'FINISHED')} {_req(req_id)}: {num_prompt_tokens:d} prompt, WAIT: {wait}, "
        f"PP: {_c(_DARK_CYAN, f'{prefill_time:.1f}s')} ({num_uncached_tokens:d} tokens "
        f"{_c(_DARK_GREEN, f'{pp_rate} PPtok/s')}) "
        f"TTFT: {_c(_DARK_CYAN, f'{ttft:.1f}s')}, GEN: {_c(_LIGHT_GREEN, str(num_generation_tokens))} "
        f"tokens in {_c(_DARK_CYAN, f'{decode_time:.1f}s')} ({_c(_LIGHT_GREEN, f'{gen_rate} tok/s')}). "
        f"E2E: {_c(_DARK_CYAN, e2e)}",
        flush=True)
