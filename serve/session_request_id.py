"""Session-tagged request ids, scheme v2 (issue stack/strata#26, SPEC.md 2-3).

Every id strata mints carries who sent it and which conversation it belongs
to, so one grep matches requests to their caller across every line that
prints the id -- NEW REQUEST / ADMITTED / FINISHED, the /api/requests record,
the request-log filename and the OpenAI response id (response id == log id;
the v1 "chatcmpl-"/"cmpl-" wrappers are gone):

    chat.hindsight-extract.b8b186bd69fc.sub2.1-m14-e3cf9a-6836
    family client        session        sub    msgs hash6 rand4
    cmpl.nova-decide.4f2a91c0d7e3-a91f          (completions: no m/hash)
    chat.plain.13cdafc05768                     (untagged fallback)

  family   chat = /v1/chat/completions (and aliases), cmpl = /v1/completions
  client   X-Client-Name (legacy X-Hermes-Client), lower-case [a-z0-9-];
           a "-m<digits>" run inside the token has that hyphen collapsed, so
           the first "-m<digits>" of an id is always the per-request tail
  session  X-Session-Id (legacy X-Hermes-Session), [a-z0-9] only, cap 40
           (an opencode "ses_" prefix is stripped); ".sub<path>" from
           X-Subagent-Path (legacy X-Hermes-Subagent)
  m<n>     number of messages in the request (chat family only)
  hash6    prompt content (messages, tools, reasoning_effort,
           chat_template_kwargs): same m + same hash is a re-send (should be
           ~fully cached), same m + different hash a fork
  rand4    keeps identical re-sends apart

Fallback ladder when no tag headers are present (SPEC 3): a user-agent
starting "opencode/" plus an x-session-id header is tagged as opencode with
that session sanitized; anything else gets the plain fallback id
(chat.plain.<12hex> / cmpl.plain.<12hex> -- never a bare 12-hex id).

Header values are untrusted: sanitize per the charsets/caps above.

`headers` is any case-insensitive get()-able (the stdlib handler's
email.message.Message); `request` is Strata's parsed body dict.
"""

import hashlib
import json
import re
import uuid
from typing import Any

# v2 sanitizers: "." is the identity separator and the first "-m<digits>"
# starts the per-request tail, so neither may appear inside a field value.
_CLIENT_UNSAFE_RE = re.compile(r"[^a-z0-9-]+")
_M_RUN_RE = re.compile(r"-m\d")            # a "-m<digits>" run inside a client token
_SESSION_UNSAFE_RE = re.compile(r"[^a-z0-9]+")
_CLIENT_MAX = 40
_SESSION_MAX = 40
_SUB_MAX = 8

# v2 ids: family first, no wrapper -- the minted shape and the response id.
# chat carries -m<msgs>-<hash6>-<nonce4>, cmpl only -<nonce4>, and the
# untagged fallback (chat|cmpl).plain.<12hex> carries no tail at all.
TAGGED_ID_RE = re.compile(
    r"^(?P<family>chat|cmpl)\.(?P<client>[a-z0-9-]+)\.(?P<session>[a-z0-9]+(?:\.sub[0-9.]+)?)"
    r"(?:-(?:m(?P<msgs>\d+)-(?P<hash>[0-9a-f]{6})-[0-9a-f]{4}|(?P<nonce>[0-9a-f]{4})))?$"
)
# v1 ids (API-PARITY-PLAN 2), kept so historical logs stay parseable (log
# coloring crc32, session_root, prefill-progress finished entries).
TAGGED_ID_RE_V1 = re.compile(
    r"chatcmpl-(?P<client>[A-Za-z0-9]+)(?:\.(?P<session>[A-Za-z0-9.]+?))?"
    r"(?:\.sub(?P<sub>[0-9.]+))?-m(?P<msgs>\d+)-(?P<hash>[0-9a-f]{6})-[0-9a-f]{4}"
)


def _clean_sub(value: str | None) -> str:
    sub = re.sub(r"[^0-9.]", "", value or "")
    return sub[:_SUB_MAX].strip(".")


def _clean_client(value: str | None) -> str:
    """v2 client token: lower-case [a-z0-9-]. A hyphen that would create a
    "-m<digits>" run is collapsed, so the first "-m<digits>" of an id stays
    the identity/tail separator (SPEC 2)."""
    v = _CLIENT_UNSAFE_RE.sub("", (value or "").lower()[: _CLIENT_MAX * 4])
    while True:
        m = _M_RUN_RE.search(v)
        if not m:
            break
        v = v[:m.start()] + v[m.start() + 1:]
    return v.strip("-")


def _clean_session(value: str | None) -> str:
    """v2 session token: [a-z0-9] only, cap 40; an opencode-style "ses_"
    prefix is dropped, any other punctuation simply falls out."""
    v = (value or "").strip()
    if v.lower().startswith("ses_"):
        v = v[4:]
    return _SESSION_UNSAFE_RE.sub("", v)[:_SESSION_MAX]


def _get2(headers: Any, new: str, legacy: str) -> str | None:
    """Read the v2 header, falling back to its legacy Hermes-branded name."""
    v = headers.get(new)
    return v if v else headers.get(legacy)


def prompt_hash(request: Any) -> str:
    """hash6: the prompt content that decides the cache hit -- the same four
    fields the vLLM fork hashes (its model_dump include-set), read off
    Strata's body dict."""
    body = {k: request.get(k) for k in
            ("messages", "tools", "reasoning_effort", "chat_template_kwargs") if request.get(k) is not None}
    data = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(data.encode("utf-8", "ignore"), digest_size=3).hexdigest()


def _resolve_tag(headers: Any) -> tuple[str, str, str]:
    """The sanitized (client, session, sub) from the tag headers plus the
    fallback ladder of SPEC 3; all empty = no identity found."""
    client = _clean_client(_get2(headers, "x-client-name", "x-hermes-client"))
    session = _clean_session(_get2(headers, "x-session-id", "x-hermes-session"))
    sub = _clean_sub(_get2(headers, "x-subagent-path", "x-hermes-subagent"))
    if not client and session:
        ua = (headers.get("user-agent") or "").strip().lower()
        if ua.startswith("opencode/"):      # opencode sends x-session-id natively
            client = "opencode"
    return client, session, sub


def _identity(client: str, session: str, sub: str) -> str:
    """client.session[.sub<path>] -- a missing half is "plain" (a session with
    no client header still names its caller slot "unknown", as in v1)."""
    ident = f"{client or ('unknown' if session else 'plain')}.{session or 'plain'}"
    if sub:
        ident = f"{ident}.sub{sub}"
    return ident


def session_request_id(headers: Any, request: Any) -> str | None:
    """The v2 chat id, complete and unwrapped: chat.<client>.<session>[.sub
    <path>]-m<msgs>-<hash6>-<nonce4>; untagged requests get
    chat.plain.<12hex-internal-id>. None only with no headers at all (a
    direct svc.run() without a handler)."""
    if headers is None:
        return None
    client, session, sub = _resolve_tag(headers)
    if not client and not session:
        return f"chat.plain.{uuid.uuid4().hex[:12]}"
    msgs = len(request.get("messages") or [])
    return f"chat.{_identity(client, session, sub)}-m{msgs}-{prompt_hash(request)}-{uuid.uuid4().hex[:4]}"


def completions_request_id(headers: Any) -> str | None:
    """The v2 completions id: cmpl.<client>.<session>[.sub<path>]-<nonce4>
    (no m/hash -- the body is a raw token-id array); untagged ->
    cmpl.plain.<12hex>. None only with no headers at all."""
    if headers is None:
        return None
    client, session, sub = _resolve_tag(headers)
    if not client and not session:
        return f"cmpl.plain.{uuid.uuid4().hex[:12]}"
    return f"cmpl.{_identity(client, session, sub)}-{uuid.uuid4().hex[:4]}"


def session_header_fields(headers: Any) -> dict:
    """The sanitized tag-header values for the /api/requests record: the v2
    keys (client/session/subagent_path) plus the legacy hermes_* keys, kept
    for one release so existing consumers keep working."""
    client, session, sub = _resolve_tag(headers)
    out: dict[str, str] = {}
    if client:
        out["client"] = out["hermes_client"] = client
    if session:
        out["session"] = out["hermes_session"] = session
    if sub:
        out["subagent_path"] = out["hermes_subagent"] = sub
    return out


def session_root(req_id: str) -> str | None:
    """The session of a tagged id without client and .sub (subagents share
    their root's log color), else None. Parses v2 ids and v1 (chatcmpl-) ids
    so historical logs stay parseable."""
    m = TAGGED_ID_RE.match(req_id)
    if m:
        s = m.group("session")
        # "plain" is not a session, and neither is the internal 12-hex of the
        # untagged fallback (chat|cmpl).plain.<12hex>: no per-request tail).
        if s == "plain" or (m.group("client") == "plain" and m.group("msgs") is None
                            and m.group("nonce") is None):
            return None
        return re.split(r"\.sub", s, maxsplit=1)[0]
    m = TAGGED_ID_RE_V1.match(req_id)
    return m.group("session") if m else None
