"""Unit tests for request-id scheme v2 (issue stack/strata#26, SPEC.md 2-3)
-- stdlib only, no server, no GPU. Golden ids per family, the sanitizer
guards, the fallback ladder, and v1 (chatcmpl-) back-compat parsing."""

import unittest
from email.message import Message

from serve.session_request_id import (TAGGED_ID_RE, TAGGED_ID_RE_V1, completions_request_id,
                                      prompt_hash, session_header_fields,
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


class ChatGolden(unittest.TestCase):
    def test_golden_tagged_id(self):
        rid = session_request_id(headers({"X-Client-Name": "hermes-agent",
                                          "X-Session-Id": "b8b186bd69fc"}), req(43))
        m = TAGGED_ID_RE.match(rid)
        self.assertTrue(m, rid)
        self.assertEqual(m["family"], "chat")
        self.assertEqual(m["client"], "hermes-agent")
        self.assertEqual(m["session"], "b8b186bd69fc")
        self.assertEqual(m["msgs"], "88")                     # system + 43 pairs + last user
        self.assertRegex(rid, r"^chat\.hermes-agent\.b8b186bd69fc-m88-[0-9a-f]{6}-[0-9a-f]{4}$")
        self.assertEqual(session_root(rid), "b8b186bd69fc")

    def test_golden_subagent_path(self):
        rid = session_request_id(headers({"X-Client-Name": "hermes-agent",
                                          "X-Session-Id": "b8b186bd69fc",
                                          "X-Subagent-Path": "2.1"}), req(6))
        self.assertRegex(rid, r"^chat\.hermes-agent\.b8b186bd69fc\.sub2\.1-m14-[0-9a-f]{6}-[0-9a-f]{4}$")
        m = TAGGED_ID_RE.match(rid)
        self.assertEqual(m["session"], "b8b186bd69fc.sub2.1")
        self.assertEqual(session_root(rid), "b8b186bd69fc")   # subagents share the root

    def test_untagged_is_chat_plain(self):
        rid = session_request_id(headers({}), req())
        self.assertRegex(rid, r"^chat\.plain\.[0-9a-f]{12}$")
        self.assertTrue(TAGGED_ID_RE.match(rid))
        self.assertIsNone(session_root(rid))                  # no session to color by
        self.assertIsNone(session_request_id(None, req()))    # no handler at all

    def test_hyphenated_client_survives(self):
        rid = session_request_id(headers({"X-Client-Name": "hindsight-extract",
                                          "X-Session-Id": "b8b186bd69fc"}), req())
        self.assertRegex(rid, r"^chat\.hindsight-extract\.b8b186bd69fc-m4-")
        self.assertEqual(TAGGED_ID_RE.match(rid)["client"], "hindsight-extract")

    def test_m_run_guard(self):
        # A client token with a "-m<digits>" run has that hyphen collapsed, so
        # the first "-m<digits>" of the id is always the per-request tail.
        rid = session_request_id(headers({"X-Client-Name": "hermes-m4x-agent",
                                          "X-Session-Id": "s1"}), req())
        m = TAGGED_ID_RE.match(rid)
        self.assertTrue(m, rid)
        self.assertEqual(m["client"], "hermesm4x-agent")
        self.assertEqual(m["session"], "s1")
        self.assertEqual(m["msgs"], "4")
        rid = session_request_id(headers({"X-Client-Name": "a-m1", "X-Session-Id": "s1"}), req())
        m = TAGGED_ID_RE.match(rid)
        self.assertTrue(m, rid)
        self.assertEqual(m["client"], "am1")                  # "a-m1" -> "am1"
        self.assertEqual(m["msgs"], "4")

    def test_case_insensitive_and_legacy_headers(self):
        rid = session_request_id(headers({"x-client-name": "WebUI", "X-SESSION-ID": "abc"}), req())
        self.assertTrue(rid.startswith("chat.webui.abc-m4-"), rid)
        # Legacy X-Hermes-* names keep working (migration, SPEC 3).
        legacy = session_request_id(headers({"X-Hermes-Client": "webui", "X-Hermes-Session": "abc"}), req())
        self.assertTrue(legacy.startswith("chat.webui.abc-m4-"), legacy)
        # New names win when both are present.
        both = session_request_id(headers({"X-Client-Name": "new", "X-Hermes-Client": "old",
                                           "X-Session-Id": "ns", "X-Hermes-Session": "os"}), req())
        self.assertTrue(both.startswith("chat.new.ns-m4-"), both)

    def test_opencode_ua_fallback(self):
        # opencode sends no tag headers but x-session-id natively (SPEC 3 ladder).
        rid = session_request_id(headers({"user-agent": "opencode/1.18.31",
                                          "x-session-id": "ses_e3a0be8ccffe"}), req())
        self.assertRegex(rid, r"^chat\.opencode\.e3a0be8ccffe-m4-[0-9a-f]{6}-[0-9a-f]{4}$")
        # UA alone (no x-session-id) stays plain; x-session-id alone gets the
        # unknown client slot.
        self.assertRegex(session_request_id(headers({"user-agent": "opencode/1.18.31"}), req()),
                         r"^chat\.plain\.[0-9a-f]{12}$")
        self.assertTrue(session_request_id(headers({"x-session-id": "s1"}), req())
                        .startswith("chat.unknown.s1-m4-"))

    def test_sanitize_untrusted(self):
        rid = session_request_id(headers({"X-Client-Name": "We-b/../ui_x",
                                          "X-Session-Id": "20261018_075512-a1b2/../c" + "z" * 200,
                                          "X-Subagent-Path": "3; rm -rf"}), req())
        m = TAGGED_ID_RE.match(rid)
        self.assertTrue(m, rid)
        self.assertEqual(m["client"], "we-buix")              # "/" "." "_" all fall out
        self.assertEqual(m["session"], "20261018075512a1b2c" + "z" * 21 + ".sub3")  # cap 40, [a-z0-9]
        self.assertEqual(session_root(rid), "20261018075512a1b2c" + "z" * 21)
        for ch in ("/", "_", " ", ";"):
            self.assertNotIn(ch, rid)

    def test_msgs_and_hash(self):
        h = {"X-Client-Name": "webui", "X-Session-Id": "s"}
        a1 = TAGGED_ID_RE.match(session_request_id(headers(h), req(1)))
        a2 = TAGGED_ID_RE.match(session_request_id(headers(h), req(1)))
        self.assertEqual(a1["hash"], a2["hash"])              # re-send: same prompt, same hash
        rids = {session_request_id(headers(h), req(1)) for _ in range(8)}
        self.assertGreater(len(rids), 1)                      # nonce keeps identical re-sends apart
        b = TAGGED_ID_RE.match(session_request_id(headers(h), req(2)))
        self.assertEqual(b["msgs"], "6")
        self.assertNotEqual(b["hash"], a1["hash"])
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        c = TAGGED_ID_RE.match(session_request_id(headers(h), req(1, tools=tools)))
        self.assertEqual(c["msgs"], "4")
        self.assertNotEqual(c["hash"], a1["hash"])            # same count, other prompt

    def test_hash_stability_and_fields(self):
        # hash6 covers exactly messages/tools/reasoning_effort/chat_template_kwargs.
        h1 = prompt_hash(req(1, reasoning_effort="low"))
        h2 = prompt_hash(req(1, reasoning_effort="low", chat_template_kwargs={"c": 1}))
        self.assertNotEqual(h1, h2)
        self.assertEqual(h1, prompt_hash(req(1, reasoning_effort="low", max_tokens=5)))  # other keys ignored
        self.assertRegex(h1, r"^[0-9a-f]{6}$")


class CompletionsGolden(unittest.TestCase):
    def test_golden_tagged_id(self):
        rid = completions_request_id(headers({"X-Client-Name": "nova-decide",
                                              "X-Session-Id": "4f2a91c0d7e3"}))
        self.assertRegex(rid, r"^cmpl\.nova-decide\.4f2a91c0d7e3-[0-9a-f]{4}$")
        m = TAGGED_ID_RE.match(rid)
        self.assertEqual(m["family"], "cmpl")
        self.assertEqual(m["client"], "nova-decide")
        self.assertEqual(m["session"], "4f2a91c0d7e3")
        self.assertIsNone(m["msgs"])                          # no m/hash for this family
        self.assertEqual(session_root(rid), "4f2a91c0d7e3")

    def test_untagged_is_cmpl_plain(self):
        rid = completions_request_id(headers({}))
        self.assertRegex(rid, r"^cmpl\.plain\.[0-9a-f]{12}$")
        self.assertTrue(TAGGED_ID_RE.match(rid))
        self.assertIsNone(session_root(rid))
        self.assertIsNone(completions_request_id(None))

    def test_opencode_ua_fallback(self):
        rid = completions_request_id(headers({"user-agent": "opencode/1.18.31",
                                              "x-session-id": "ses_e3a0be8ccffe"}))
        self.assertRegex(rid, r"^cmpl\.opencode\.e3a0be8ccffe-[0-9a-f]{4}$")


class V1BackCompat(unittest.TestCase):
    """Historical logs keep parsing (log coloring, session_root, prefill-progress)."""

    def test_v1_id_parses(self):
        old = "chatcmpl-webui.0580615c1c52-m87-3fa2c1-9e81"
        self.assertIsNone(TAGGED_ID_RE.match(old))            # v2 regex is family-anchored
        m = TAGGED_ID_RE_V1.match(old)
        self.assertTrue(m, old)
        self.assertEqual(m["client"], "webui")
        self.assertEqual(m["session"], "0580615c1c52")
        self.assertEqual(m["msgs"], "87")
        self.assertEqual(session_root(old), "0580615c1c52")

    def test_v1_subagent_id_parses(self):
        old = "chatcmpl-subagent.0580615c1c52.sub2-m4-3fa2c1-9e81"
        m = TAGGED_ID_RE_V1.match(old)
        self.assertEqual(m["session"], "0580615c1c52")
        self.assertEqual(m["sub"], "2")
        self.assertEqual(session_root(old), "0580615c1c52")

    def test_plain_hex_and_garbage_are_not_tagged(self):
        for rid in ("13cdafc05768", "chatcmpl-abc-def", "chat.plain", "nope"):
            self.assertIsNone(session_root(rid))


class RecordFields(unittest.TestCase):
    def test_v2_and_legacy_keys(self):
        f = session_header_fields(headers({"X-Client-Name": "we-b", "X-Session-Id": "s1",
                                           "X-Subagent-Path": "2; x"}))
        self.assertEqual(f, {"client": "we-b", "hermes_client": "we-b",
                             "session": "s1", "hermes_session": "s1",
                             "subagent_path": "2", "hermes_subagent": "2"})
        self.assertEqual(session_header_fields(headers({"user-agent": "x"})), {})


if __name__ == "__main__":
    unittest.main()
