"""Session-tagged request ids (API-PARITY-PLAN.md section 2), ported from the
vLLM fork's entrypoints/openai/session_request_id.py
(patches/request-session-id.patch, HERMES_SESSION_TAGGING_REPORT.md).

When a caller sends X-Hermes-Session (or X-Hermes-Client), the request id
carries who sent it and where it sits in its conversation, so every line that
prints the id -- NEW REQUEST / ADMITTED / FINISHED, the /api/requests record
and the API response -- can be grepped by session:

    chatcmpl-webui.0580615c1c52-m87-3fa2c1-9e81
             client.session     msgs hash6  rand4

  client   X-Hermes-Client (missing -> "unknown")
  session  X-Hermes-Session, plus ".sub<N>" from X-Hermes-Subagent
  m<n>     number of messages in the request: rises through a conversation,
           repeats on a fork/regenerate, drops after compression
  hash6    prompt content (messages, tools, reasoning_effort,
           chat_template_kwargs): same m + same hash is a re-send (should be
           ~fully cached), same m + different hash a fork
  rand4    keeps identical re-sends apart

Header values are untrusted: anything outside [A-Za-z0-9] becomes "." (so "-"
stays the field separator) and each field is capped. Untagged requests keep
Strata's plain ids: the 12-hex internal id and the chatcmpl-<24 hex> response
id.

`headers` is any case-insensitive get()-able (the stdlib handler's
email.message.Message); `request` is Strata's parsed body dict.
"""

import hashlib
import json
import re
import uuid
from typing import Any

_UNSAFE_RE = re.compile(r"[^A-Za-z0-9]+")
_CLIENT_MAX = 16
_SESSION_MAX = 40
_SUB_MAX = 8

# Parses the tag back out of an id (external or internal).
TAGGED_ID_RE = re.compile(
    r"chatcmpl-(?P<client>[A-Za-z0-9]+)(?:\.(?P<session>[A-Za-z0-9.]+?))?"
    r"(?:\.sub(?P<sub>[0-9.]+))?-m(?P<msgs>\d+)-(?P<hash>[0-9a-f]{6})-[0-9a-f]{4}"
)


def _clean(value: str | None, max_len: int) -> str:
    if not value:
        return ""
    return _UNSAFE_RE.sub(".", value[: max_len * 4]).strip(".")[:max_len].strip(".")


def _clean_sub(value: str | None) -> str:
    sub = re.sub(r"[^0-9.]", "", value or "")
    return sub[:_SUB_MAX].strip(".")


def prompt_hash(request: Any) -> str:
    """hash6: the prompt content that decides the cache hit -- the same four
    fields the vLLM fork hashes (its model_dump include-set), read off
    Strata's body dict."""
    body = {k: request.get(k) for k in
            ("messages", "tools", "reasoning_effort", "chat_template_kwargs") if request.get(k) is not None}
    data = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(data.encode("utf-8", "ignore"), digest_size=3).hexdigest()


def session_request_id(headers: Any, request: Any) -> str | None:
    """The tagged base id (without "chatcmpl-"), or None when the request
    carries no session tag headers. `headers` is case-insensitive."""
    session = _clean(headers.get("x-hermes-session"), _SESSION_MAX)
    client = _clean(headers.get("x-hermes-client"), _CLIENT_MAX).replace(".", "")
    if not session and not client:
        return None
    sub = _clean_sub(headers.get("x-hermes-subagent"))
    if sub:
        session = f"{session}.sub{sub}"
    msgs = len(request.get("messages") or [])
    tag = f"{client or 'unknown'}.{session}" if session else client
    return f"{tag}-m{msgs}-{prompt_hash(request)}-{uuid.uuid4().hex[:4]}"


def session_header_fields(headers: Any) -> dict:
    """The sanitized Hermes header values for the /api/requests record
    (plan section 6): the same cleaning session_request_id applies, kept as
    separate fields; only the headers actually present are returned."""
    out: dict[str, str] = {}
    session = _clean(headers.get("x-hermes-session"), _SESSION_MAX)
    client = _clean(headers.get("x-hermes-client"), _CLIENT_MAX).replace(".", "")
    sub = _clean_sub(headers.get("x-hermes-subagent"))
    if session:
        out["hermes_session"] = session
    if client:
        out["hermes_client"] = client
    if sub:
        out["hermes_subagent"] = sub
    return out


def session_root(req_id: str) -> str | None:
    """The session of a tagged id without client and .sub (subagents share
    their root's log color), else None."""
    m = TAGGED_ID_RE.match(req_id)
    return m.group("session") if m else None
