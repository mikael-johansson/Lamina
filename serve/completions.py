"""serve/completions.py - POST /v1/completions: a raw prompt (token ids or text) straight to the engine, in vLLM's
shape, for clients that build their own prompts token by token (voice-lab speakrail's Qwen brain).

No chat template, no reasoning or tool parsing: the client's ids are the prompt and the generated tokens come back as
text (special tokens included: `skip_special_tokens` is always false here).  What a client gets:

- decisions: `max_tokens` 1 with `allowed_token_ids` and `logprobs` -> the engine's own log-probabilities of those
  ids (the `score` path of /v1/choose, one forward pass); `choices[0].logprobs` has `tokens` and `top_logprobs`
  over the allowed ids, keyed "token_id:<id>" with `return_tokens_as_token_ids` (else the token's text); the choice
  is the most likely allowed id.
- replies: `stream` with `stop_token_ids`; `finish_reason` "stop" with `stop_reason` = the id that ended it (the
  engine's end-of-turn ids count as stops too; its text is not in the output, as vLLM without
  include_stop_str_in_output), or "length".  `usage` has `prompt_tokens_details.cached_tokens`
  (the engine's reused prefix).
- `priority` (lower first, vLLM's field) travels in the sampling dict, as on the chat routes: with "parallel" and
  PRIORITY_SCHED=1 it picks the request's admission tier (T0 <= -20 voice decisions, T1 -19..-1 voice replies);
  otherwise it is accepted and ignored.

Accepted and ignored: `cache_salt` (one resident sequence: nothing is shared between tenants), `stream_options`,
`skip_special_tokens`, `n` 1, `echo` false.  Refused: `ignore_eos` (the engine always ends at its end-of-turn ids),
`allowed_token_ids` without max_tokens 1, `n` > 1, `echo`, `best_of`.
"""
from __future__ import annotations

import json
import math
import time
import uuid

FORWARD = ("temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty", "frequency_penalty",
           "repetition_penalty")


def parse(req: dict, tok, vocab_size: int | None, max_context: int) -> dict:
    """The request -> {"ids", "max_new", "sampling", "stop_ids", "allowed", "logprobs", "as_ids", "stream"}; a bad
    field is a ValueError (400)."""
    prompt = req.get("prompt")
    if isinstance(prompt, list) and prompt and all(isinstance(t, int) and not isinstance(t, bool) for t in prompt):
        ids = list(prompt)
    elif isinstance(prompt, str) and prompt:
        ids = tok.encode(prompt, parse_special=True)
    else:
        raise ValueError("prompt: a non-empty list of token ids or a string (one prompt per request)")
    if vocab_size and any(t < 0 or t >= vocab_size for t in ids):
        raise ValueError("prompt: a token id outside the model's vocabulary")
    if req.get("n", 1) != 1 or req.get("best_of") not in (None, 1):
        raise ValueError("n / best_of: one completion per request")
    if req.get("echo"):
        raise ValueError("echo: not supported")
    if req.get("ignore_eos"):
        raise ValueError("ignore_eos: not supported (the engine ends at its end-of-turn ids)")
    max_new = req.get("max_tokens", 16)
    if isinstance(max_new, bool) or not isinstance(max_new, int) or max_new < 1:
        raise ValueError("max_tokens: a whole number of at least 1")
    if len(ids) + max_new > max_context:
        raise ValueError(f"the prompt ({len(ids)} tokens) and max_tokens ({max_new}) do not fit the context "
                         f"({max_context})")
    def id_list(name):
        v = req.get(name) or []
        if not isinstance(v, list) or not all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in v):
            raise ValueError(f"{name}: a list of token ids")
        if vocab_size and any(t >= vocab_size for t in v):
            raise ValueError(f"{name}: a token id outside the model's vocabulary")
        return list(dict.fromkeys(v))
    stop_ids, allowed = id_list("stop_token_ids"), id_list("allowed_token_ids")
    if allowed and max_new != 1:
        raise ValueError("allowed_token_ids: only with max_tokens 1 (a decision: the allowed ids are scored)")
    logprobs = req.get("logprobs")
    if logprobs is not None and (isinstance(logprobs, bool) or not isinstance(logprobs, int) or logprobs < 0):
        raise ValueError("logprobs: a whole number")
    if logprobs and not allowed:
        raise ValueError("logprobs: only with allowed_token_ids (the engine scores given ids, it has no top-k list)")
    sampling = {k: req[k] for k in FORWARD if req.get(k) is not None}
    if allowed:
        sampling = {"score": allowed}                     # the head's own logits for these ids; nothing is sampled
    elif req.get("logit_bias"):                           # (a decision's bias is added to the scores: decision())
        sampling["logit_bias"] = req["logit_bias"]
    if req.get("priority") is not None:
        sampling["priority"] = req["priority"]           # (the admission tier: StrataEngine.generate_batched)
    # "strata_cache_prefix": N - the first N prompt ids are the client's growing context, the rest a throw-away probe
    # (a voice decision): the engine keeps a rolling checkpoint at N and none after it, so the next request with one
    # more word resumes at N.  "strata_checkpoint": false - no checkpoint at all (a one-shot call)
    cp = req.get("strata_cache_prefix")
    if cp is not None:
        if isinstance(cp, bool) or not isinstance(cp, int) or not 0 < cp < len(ids):
            raise ValueError("strata_cache_prefix: a whole number of prompt ids, more than 0 and fewer than the prompt's")
        sampling["strata_cache_prefix"] = cp
    if req.get("strata_checkpoint") is False:
        sampling["strata_checkpoint"] = False
    return {"ids": ids, "max_new": max_new, "sampling": sampling, "stop_ids": set(stop_ids), "allowed": allowed,
            "bias": {int(k): float(v) for k, v in (req.get("logit_bias") or {}).items()} if allowed else {},
            "logprobs": logprobs or 0, "as_ids": bool(req.get("return_tokens_as_token_ids")),
            "stream": bool(req.get("stream"))}


def run(svc, p: dict, cancel, beat_s: float):
    """Yields ("ping", None) while queued or while the engine is quiet, ("token", id) per generated token, then
    ("done", {"finish", "stop_reason", "completion_tokens", "prompt_tokens", "cached", "scores"}); the stop token
    counts in completion_tokens but is not yielded.  One request at a time: holds the engine's turn (svc.fifo) from
    the wait to the end; with "parallel" (the engine's batch slots) the engine orders the requests itself
    (generate_batched: the control lines, the slots, the priority tiers), as Service._run leaves it to."""
    from serve.server import QueueCancelled               # (import here: server imports this module)
    par = bool(getattr(svc.engine, "batch", 0))
    if not par:
        while not svc.fifo.acquire(timeout=beat_s):
            if cancel.is_set():
                raise QueueCancelled("the client disconnected while waiting for the engine")
            yield "ping", None
    try:
        if cancel.is_set():
            raise QueueCancelled("the client disconnected while waiting for the engine")
        if not par:
            svc.ensure_loaded()
        elif not (svc.loaded() and not svc._vision_down()):
            with svc._inference_fifo(None, cancel):       # (parallel requests do not hold the fifo; one starts it)
                svc.ensure_loaded()
        svc.last_request_at = time.time()
        eng = svc.engine
        stops = p["stop_ids"] | set(getattr(svc, "stop_ids", ()) or ())
        n, finish, stop_reason = 0, "length", None
        gen = eng.generate(p["ids"], p["max_new"], p["sampling"], cancel)
        try:
            for t in gen:
                if t is None:
                    yield "ping", None
                    continue
                n += 1
                if t in stops:                            # as vLLM: the stop token ends the text, it is not in it
                    finish, stop_reason = "stop", t
                    break
                yield "token", t                          # (the engine itself ends at max_new: no STOP for that)
        finally:
            gen.close()                                   # an early end (a stop id the engine does not know) STOPs it
        last = dict(getattr(eng, "last", None) or {})
        if cancel.is_set() and n == 0:
            raise QueueCancelled("the client disconnected")
        yield "done", {"finish": finish, "stop_reason": stop_reason, "completion_tokens": n,
                       "prompt_tokens": len(p["ids"]), "cached": min(int(last.get("reused") or 0), len(p["ids"])),
                       "scores": last.get("scores") or {}}
    finally:
        if not par:
            svc.fifo.release()


def _token_key(tok, t: int, as_ids: bool) -> str:
    return f"token_id:{t}" if as_ids else tok.decode([t])


def decision(p: dict, scores: dict, tok) -> tuple[int, dict]:
    """The allowed id with the highest score (+ the request's logit_bias), and vLLM's logprobs block: the allowed ids'
    log-probabilities renormalized over them, as vLLM reports them under allowed_token_ids (it masks the others)."""
    raw = {t: scores[t] + p["bias"].get(t, 0.0) for t in p["allowed"] if t in scores}
    if len(raw) != len(p["allowed"]):
        raise ValueError("the engine returned no log-probabilities for the allowed ids")
    top = max(raw.values())
    z = top + math.log(sum(math.exp(v - top) for v in raw.values()))
    lp = {t: v - z for t, v in raw.items()}
    choice = max(lp, key=lp.get)
    k = p["logprobs"]
    best = sorted(lp.items(), key=lambda kv: -kv[1])[:k] if k else []
    block = None
    if k:
        block = {"tokens": [_token_key(tok, choice, p["as_ids"])], "token_logprobs": [lp[choice]],
                 "top_logprobs": [{_token_key(tok, t, p["as_ids"]): v for t, v in best}], "text_offset": [0]}
    return choice, block


def usage(done: dict) -> dict:
    return {"prompt_tokens": done["prompt_tokens"], "completion_tokens": done["completion_tokens"],
            "total_tokens": done["prompt_tokens"] + done["completion_tokens"],
            "prompt_tokens_details": {"cached_tokens": done["cached"]}}


def responses(svc, p: dict, cancel, beat_s: float, model: str):
    """The OpenAI/vLLM completion objects: the whole one (not streaming), or the stream's chunks (None = a keep-alive,
    the last chunk carries finish_reason, stop_reason and usage)."""
    from serve.server import Detokenizer, QueueCancelled
    cid, created = "cmpl-" + uuid.uuid4().hex[:24], int(time.time())
    def obj(text, finish=None, stop_reason=None, logprobs=None, use=None):
        o = {"id": cid, "object": "text_completion", "created": created, "model": model,
             "choices": [{"index": 0, "text": text, "logprobs": logprobs, "finish_reason": finish,
                          "stop_reason": stop_reason}]}
        if use is not None:
            o["usage"] = use
        return o
    detok, text, done = Detokenizer(svc.tok), [], None
    try:
        for kind, value in run(svc, p, cancel, beat_s):
            if kind == "ping":
                if p["stream"]:
                    yield None
            elif kind == "token":
                if not p["allowed"]:
                    piece = detok.push(value)
                    if piece:
                        if p["stream"]:
                            yield obj(piece)
                        else:
                            text.append(piece)
            else:
                done = value
    except QueueCancelled:                                # the client left: nobody to answer (as the chat route)
        return
    if done is None:
        return
    logprobs = None
    if p["allowed"]:
        choice, logprobs = decision(p, done["scores"], svc.tok)
        piece = svc.tok.decode([choice])
        done = {**done, "finish": "length", "stop_reason": None, "completion_tokens": 1}
        if p["stream"]:
            yield obj(piece, logprobs=logprobs)
        else:
            text = [piece]
    if p["stream"]:
        yield obj("", done["finish"], done["stop_reason"], use=usage(done))
    else:
        yield obj("".join(text), done["finish"], done["stop_reason"], logprobs, usage(done))


def sse(chunk) -> bytes:
    return b": keep-alive\n\n" if chunk is None else b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n"
