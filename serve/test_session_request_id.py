"""Unit tests for the ported session_request_id (API-PARITY-PLAN.md section 2)
-- stdlib only, no server, no GPU. Ported from the vLLM fork's
test_request_session_id.py (the request-log-filename test has no Strata
equivalent and is not ported)."""

import unittest
from email.message import Message

from serve.session_request_id import (TAGGED_ID_RE, prompt_hash, session_header_fields,
                                      session_request_id, session_root)


def headers(d):
    """A case-insensitive get()-able like the stdlib handler's self.headers."""
    m = Message()
    for k, v in d.items():
        m[k] = v
    return m


def req(n_user=1, **kw):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_user):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    msgs.append({"role": "user", "content": "last"})
    return {"model": "m", "messages": msgs, **kw}


def tagged(hdrs, request=None):
    base = session_request_id(headers(hdrs), request or req())
    return None if base is None else f"chatcmpl-{base}"


class SessionTaggedId(unittest.TestCase):
    def test_untagged_is_none(self):
        self.assertIsNone(tagged({}))
        self.assertIsNone(tagged({"user-agent": "x"}))

    def test_shape_and_parse(self):
        rid = tagged({"X-Hermes-Client": "webui", "X-Hermes-Session": "0580615c1c52"})
        m = TAGGED_ID_RE.match(rid)
        self.assertTrue(m, rid)
        self.assertEqual(m["client"], "webui")
        self.assertEqual(m["session"], "0580615c1c52")
        self.assertIsNone(m["sub"])
        self.assertEqual(m["msgs"], "4")
        self.assertEqual(session_root(rid), "0580615c1c52")
        # The internal id (Strata keeps the whole tagged id) parses the same.
        self.assertEqual(session_root(rid), "0580615c1c52")

    def test_case_insensitive_headers(self):
        rid = tagged({"x-hermes-client": "cli", "X-HERMES-SESSION": "abc"})
        self.assertTrue(rid.startswith("chatcmpl-cli.abc-m4-"), rid)

    def test_session_only_and_client_only(self):
        self.assertTrue(tagged({"X-Hermes-Session": "s1"}).startswith("chatcmpl-unknown.s1-m4-"))
        rid = tagged({"X-Hermes-Client": "opencode"})
        self.assertTrue(rid.startswith("chatcmpl-opencode-m4-"), rid)
        self.assertIsNone(TAGGED_ID_RE.match(rid)["session"])
        self.assertIsNone(session_root(rid))

    def test_subagent(self):
        rid = tagged({"X-Hermes-Client": "subagent", "X-Hermes-Session": "0580615c1c52",
                      "X-Hermes-Subagent": "2"})
        self.assertTrue(rid.startswith("chatcmpl-subagent.0580615c1c52.sub2-m4-"), rid)
        m = TAGGED_ID_RE.match(rid)
        self.assertEqual(m["session"], "0580615c1c52")
        self.assertEqual(m["sub"], "2")
        nested = tagged({"X-Hermes-Client": "subagent", "X-Hermes-Session": "s",
                         "X-Hermes-Subagent": "2.1"})
        self.assertEqual(TAGGED_ID_RE.match(nested)["sub"], "2.1")

    def test_sanitize_untrusted(self):
        rid = tagged({"X-Hermes-Client": "we-b/ui_x",
                      "X-Hermes-Session": "20261018_075512-a1b2/../c" + "z" * 200,
                      "X-Hermes-Subagent": "3; rm -rf"})
        m = TAGGED_ID_RE.match(rid)
        self.assertTrue(m, rid)
        self.assertEqual(m["client"], "webuix")
        self.assertTrue(m["session"].startswith("20261018.075512.a1b2.c"))
        self.assertLessEqual(len(m["session"]), 40)
        self.assertEqual(m["sub"], "3")
        for ch in ("/", "_", " ", ";"):
            self.assertNotIn(ch, rid)

    def test_msgs_and_hash(self):
        h = {"X-Hermes-Client": "webui", "X-Hermes-Session": "s"}
        a1 = TAGGED_ID_RE.match(tagged(h, req(1)))
        a2 = TAGGED_ID_RE.match(tagged(h, req(1)))
        self.assertEqual(a1["hash"], a2["hash"])       # re-send: same prompt, same hash
        b = TAGGED_ID_RE.match(tagged(h, req(2)))
        self.assertEqual(b["msgs"], "6")
        self.assertNotEqual(b["hash"], a1["hash"])
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        c = TAGGED_ID_RE.match(tagged(h, req(1, tools=tools)))
        self.assertEqual(c["msgs"], "4")
        self.assertNotEqual(c["hash"], a1["hash"])    # same count, other prompt

    def test_hash_stability_and_fields(self):
        # hash6 covers exactly messages/tools/reasoning_effort/chat_template_kwargs.
        h1 = prompt_hash(req(1, reasoning_effort="low"))
        h2 = prompt_hash(req(1, reasoning_effort="low", chat_template_kwargs={"c": 1}))
        self.assertNotEqual(h1, h2)
        self.assertEqual(h1, prompt_hash(req(1, reasoning_effort="low", max_tokens=5)))  # other keys ignored
        self.assertRegex(h1, r"^[0-9a-f]{6}$")

    def test_header_fields_for_records(self):
        f = session_header_fields(headers({"X-Hermes-Client": "we-b", "X-Hermes-Session": "s_1",
                                           "X-Hermes-Subagent": "2; x"}))
        self.assertEqual(f, {"hermes_session": "s.1", "hermes_client": "web", "hermes_subagent": "2"})
        self.assertEqual(session_header_fields(headers({"user-agent": "x"})), {})


if __name__ == "__main__":
    unittest.main()
