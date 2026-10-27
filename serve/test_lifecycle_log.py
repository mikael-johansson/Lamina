"""Golden-format tests for serve/lifecycle_log.py (API-PARITY-PLAN.md section
10): the exact line formats from the vLLM fork's request-lifecycle-log
patches, the color gating (VLLM_LOGGING_COLOR), and the per-request crc32 id
coloring -- stdlib only, no server, no GPU."""

import contextlib
import importlib
import io
import os
import unittest
import zlib


def load(color=None, off=False):
    """Import the module with the env fixed, like a deployment would."""
    env = dict(os.environ)
    if color is None:
        env.pop("VLLM_LOGGING_COLOR", None)
    else:
        env["VLLM_LOGGING_COLOR"] = color
    if off:
        env["VLLM_REQUEST_LIFECYCLE_LOG"] = "0"
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        import serve.lifecycle_log as mod
        return importlib.reload(mod)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def capture(mod, fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(*args, **kw)
    return out.getvalue()


class GoldenPlain(unittest.TestCase):
    """Colors off (VLLM_LOGGING_COLOR=0): the exact text of each line."""

    def setUp(self):
        self.mod = load(color="0")

    def test_new_request(self):
        line = capture(self.mod, self.mod.log_new_request,
                       "chatcmpl-webui.0580615c1c52-m87-3fa2c1-9e81", 83746)
        self.assertEqual(line, "NEW REQUEST chatcmpl-webui.0580615c1c52-m87-3fa2c1-9e81: 83746 prompt tokens\n")

    def test_admitted(self):
        line = capture(self.mod, self.mod.log_admitted_request,
                       "chatcmpl-webui.0580615c1c52-m87-3fa2c1-9e81", 83746, 81746, 0.4)
        self.assertEqual(
            line,
            "ADMITTED chatcmpl-webui.0580615c1c52-m87-3fa2c1-9e81: 83746 prompt, 97.6% cached "
            "(cached 81746), 2000 to compute, waited 0.4s\n")

    def test_finished(self):
        line = capture(self.mod, self.mod.log_finished_request,
                       "chatcmpl-webui.0580615c1c52-m87-3fa2c1-9e81",
                       num_prompt_tokens=83746, num_uncached_tokens=2000, prefill_time=7.3,
                       ttft=7.5, num_generation_tokens=237, decode_time=2.7,
                       queue_time=0.4, e2e_time=10.2)
        self.assertEqual(
            line,
            "FINISHED chatcmpl-webui.0580615c1c52-m87-3fa2c1-9e81: 83746 prompt, WAIT: 0.4s, "
            "PP: 7.3s (2000 tokens 274.0 PPtok/s) TTFT: 7.5s, GEN: 237 tokens in 2.7s (87.8 tok/s). "
            "E2E: 10.2s\n")

    def test_finished_missing_times(self):
        line = capture(self.mod, self.mod.log_finished_request,
                       "abc123def456", num_prompt_tokens=10, num_uncached_tokens=10,
                       prefill_time=0.0, ttft=0.5, num_generation_tokens=0, decode_time=0.0)
        self.assertEqual(
            line,
            "FINISHED abc123def456: 10 prompt, WAIT: ?, PP: 0.0s (10 tokens ? PPtok/s) TTFT: 0.5s, "
            "GEN: 0 tokens in 0.0s (? tok/s). E2E: ?\n")


class Colors(unittest.TestCase):
    def setUp(self):
        self.mod = load(color="1")

    def test_labels_and_segments(self):
        out = capture(self.mod, self.mod.log_finished_request,
                      "chatcmpl-webui.0580615c1c52-m87-3fa2c1-9e81",
                      num_prompt_tokens=83746, num_uncached_tokens=2000, prefill_time=7.3,
                      ttft=7.5, num_generation_tokens=237, decode_time=2.7,
                      queue_time=0.4, e2e_time=10.2)
        self.assertIn("\x1b[92mFINISHED\x1b[0m", out)          # label light green
        self.assertIn("\x1b[38;5;28m274.0 PPtok/s\x1b[0m", out)  # prefill rate dark green
        self.assertIn("\x1b[92m237\x1b[0m", out)               # gen tokens light green
        self.assertIn("\x1b[36m7.3s\x1b[0m", out)              # times dark cyan
        self.assertIn("\x1b[33m0.4s\x1b[0m", out)              # WAIT brown above 0.1s
        out2 = capture(self.mod, self.mod.log_admitted_request, "x", 100, 90, 0.05)
        self.assertIn("\x1b[32mADMITTED\x1b[0m", out2)         # label green
        self.assertIn("\x1b[92mcached 90\x1b[0m", out2)        # hit light green
        self.assertIn("\x1b[91m10 to compute\x1b[0m", out2)    # recompute red
        out3 = capture(self.mod, self.mod.log_new_request, "x", 10)
        self.assertTrue(out3.startswith("NEW REQUEST "))          # the label itself stays plain
        self.assertNotIn("\x1b[", out3[:len("NEW REQUEST ")])

    def test_wait_plain_below_threshold(self):
        out = capture(self.mod, self.mod.log_finished_request,
                      "x", num_prompt_tokens=10, num_uncached_tokens=10, prefill_time=1.0,
                      ttft=1.0, num_generation_tokens=1, decode_time=1.0, queue_time=0.05,
                      e2e_time=2.0)
        self.assertIn("WAIT: 0.1s", out)
        self.assertNotIn("\x1b[33m", out)

    def test_per_request_color_by_session(self):
        h = {"X-Hermes-Session": "0580615c1c52"}
        from serve.session_request_id import session_request_id
        from email.message import Message

        def hdrs(d):
            m = Message()
            for k, v in d.items():
                m[k] = v
            return m

        msgs = [{"role": "user", "content": "q"}]
        ids = [f"chatcmpl-{session_request_id(hdrs({**h, 'X-Hermes-Client': c}), {'messages': msgs})}"
               for c in ("webui", "aux")]
        colors = {self.mod._req(i).split("m", 1)[0] for i in ids}
        self.assertEqual(len(colors), 1, colors)               # one session, one color
        # The color is the crc32 of the session into the 12-color palette.
        want = self.mod._REQ_COLORS[zlib.crc32(b"0580615c1c52") % 12]
        self.assertIn(f"\x1b[{want}m", self.mod._req(ids[0]))
        # An untagged id colors by "chatcmpl-<head>" instead.
        self.assertIn(f"\x1b[{self.mod._REQ_COLORS[zlib.crc32(b'chatcmpl-abc') % 12]}m",
                      self.mod._req("chatcmpl-abc-def"))

    def test_id_cut_at_64(self):
        long_id = "chatcmpl-" + "a" * 100
        shown = self.mod._req(long_id)
        self.assertIn(long_id[:64], shown)
        self.assertIn("…", shown)


class ThroughRun(unittest.TestCase):
    """The real hook sites in Service._run: with a thread-local tagged id (what
    the handler sets from the X-Hermes-* headers) all three lifecycle lines
    carry it, and a plain MockEngine run still prints NEW REQUEST / ADMITTED /
    FINISHED with the plain id."""

    def setUp(self):
        self.mod = load(color="0")           # colors off: golden text

    def _svc(self):
        from pathlib import Path
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        return svc, tok

    def test_lines_carry_the_tagged_id(self):
        import threading
        svc, tok = self._svc()
        tag = "webui.0580615c1c52-m4-3fa2c1-9e81"
        svc.request_trace.tagged_id = tag
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for kind, _ in svc.run(tok.encode("hello"), False, None, 10, {}, threading.Event()):
                pass
        svc.request_trace.tagged_id = None
        lines = out.getvalue().splitlines()
        want = f"chatcmpl-{tag}"
        new = [l for l in lines if l.startswith("NEW REQUEST")]
        adm = [l for l in lines if l.startswith("ADMITTED")]
        fin = [l for l in lines if l.startswith("FINISHED")]
        self.assertEqual(len(new), 1, lines)
        self.assertEqual(len(adm), 1, lines)
        self.assertEqual(len(fin), 1, lines)
        for l in new + adm + fin:
            self.assertIn(want, l)
        # the MockEngine keeps no clock: PP/GEN times are 0.0s, "?" rates
        self.assertRegex(fin[0], r"FINISHED chatcmpl-webui\.0580615c1c52-m4-3fa2c1-9e81: \d+ prompt, "
                                  r"WAIT: \d+\.\ds, PP: 0\.0s \(\d+ tokens [?0-9.]* PPtok/s\) "
                                  r"TTFT: \d+\.\ds, GEN: \d+ tokens in 0\.0s \(\? tok/s\)\. E2E: \d+\.\ds")

    def test_untagged_run_keeps_plain_id(self):
        import threading
        svc, tok = self._svc()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for kind, _ in svc.run(tok.encode("hello"), False, None, 10, {}, threading.Event()):
                pass
        new = [l for l in out.getvalue().splitlines() if l.startswith("NEW REQUEST")]
        self.assertEqual(len(new), 1)
        # the plain run id (12 hex), not a chatcmpl- form
        self.assertRegex(new[0], r"NEW REQUEST [0-9a-f]{12}: \d+ prompt tokens")


class Gating(unittest.TestCase):
    def test_disabled_by_env(self):
        mod = load(color="0", off=True)
        self.assertFalse(mod.lifecycle_log_enabled())
        self.assertEqual(capture(mod, mod.log_new_request, "x", 1), "")
        self.assertEqual(capture(mod, mod.log_admitted_request, "x", 1, 0, 0), "")
        self.assertEqual(capture(mod, mod.log_finished_request,
                                 "x", num_prompt_tokens=1, num_uncached_tokens=1, prefill_time=1,
                                 ttft=1, num_generation_tokens=1, decode_time=1), "")

    def test_color_off_without_env_when_not_tty(self):
        # The tests run with stdout redirected: unset VLLM_LOGGING_COLOR means
        # auto-off (not a TTY), so no escapes appear.
        mod = load(color=None)
        self.assertFalse(mod._COLOR)
        out = capture(mod, mod.log_new_request, "chatcmpl-x.y-m1-abcdef-1234", 5)
        self.assertNotIn("\x1b[", out)


if __name__ == "__main__":
    unittest.main()
