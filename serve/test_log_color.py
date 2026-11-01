"""Golden-format tests for serve/log_color.py (issue stack/strata#5, API-PARITY-PLAN
section 10): the vLLM ColoredFormatter palette (grey stamp, level colors), the
env gating (VLLM_LOGGING_COLOR / NO_COLOR, TTY auto-off) and that plain output is
byte-identical when color is off -- stdlib only, no server, no GPU."""

import importlib
import os
import unittest


def load(color=None, no_color=None):
    """Import the module with the env fixed, like a deployment would."""
    env = dict(os.environ)
    for key, val in (("VLLM_LOGGING_COLOR", color), ("NO_COLOR", no_color)):
        if val is None:
            env.pop(key, None)
        else:
            env[key] = val
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        import serve.log_color as mod
        return importlib.reload(mod)
    finally:
        os.environ.clear()
        os.environ.update(saved)


class Gating(unittest.TestCase):
    def test_off_without_env_when_not_tty(self):
        # The tests run with stdout redirected: unset means auto-off (not a TTY).
        self.assertFalse(load().color_enabled())

    def test_on_with_env(self):
        self.assertTrue(load(color="1").color_enabled())

    def test_off_with_zero(self):
        self.assertFalse(load(color="0").color_enabled())

    def test_no_color_wins_over_one(self):
        # vLLM's rule: NO_COLOR checked before VLLM_LOGGING_COLOR=1.
        self.assertFalse(load(color="1", no_color="1").color_enabled())

    def test_empty_no_color_does_not_disable(self):
        self.assertTrue(load(color="1", no_color="").color_enabled())


class Plain(unittest.TestCase):
    """Colors off: every line byte-identical to today's output."""

    def setUp(self):
        self.mod = load(color="0")

    def test_stamped_line_untouched(self):
        line = "12:34:56.789 [strata] WARNING: RAM is tight - the model's experts take 55 GB"
        self.assertEqual(self.mod.console_line(line), line)

    def test_no_stamp_no_change(self):
        line = "[strata] stopping (Ctrl+C again to end the engine at once) ..."
        self.assertEqual(self.mod.console_line(line), line)

    def test_level_untouched(self):
        self.assertEqual(self.mod.level("WARNING: no API key"), "WARNING: no API key")


class Colored(unittest.TestCase):
    def setUp(self):
        self.mod = load(color="1")

    def test_stamp_grey(self):
        self.assertEqual(
            self.mod.console_line("12:34:56.789 [strata] draining 2 inference call(s)"),
            "\033[90m12:34:56.789\033[0m \033[90m[strata]\033[0m draining 2 inference call(s)")

    def test_warning_after_tag(self):
        self.assertEqual(
            self.mod.console_line("12:34:56.789 [strata] WARNING: RAM is tight"),
            "\033[90m12:34:56.789\033[0m \033[90m[strata]\033[0m \033[33mWARNING\033[0m: RAM is tight")

    def test_warning_indented(self):
        # The "       WARNING: no API key ..." line is indented text whose body
        # still opens with the level word: colored after the indent.
        self.assertEqual(
            self.mod.console_line("       WARNING: no API key"),
            "       \033[33mWARNING\033[0m: no API key")

    def test_level_colors_match_vllm(self):
        want = {"DEBUG": "37", "INFO": "32", "WARNING": "33", "ERROR": "31",
                "CRITICAL": "35"}
        for word, code in want.items():
            self.assertEqual(
                self.mod.level(f"[strata] {word}: x"),
                f"[strata] \033[{code}m{word}\033[0m: x", word)

    def test_level_needs_a_delimiter(self):
        # "ERRORS ..." is a sentence, not a level.
        self.assertEqual(self.mod.level("[strata] ERRORS happened"), "[strata] ERRORS happened")

    def test_level_word_mid_line_untouched(self):
        self.assertEqual(self.mod.level("[strata] the engine ERRORed once"),
                         "[strata] the engine ERRORed once")

    def test_lifecycle_lines_only_get_the_grey_stamp(self):
        # The lifecycle palette is applied inside serve/lifecycle_log.py; console_line
        # must not double-color its labels (NEW REQUEST / ADMITTED / FINISHED are not
        # level words, so only the stamp changes).
        line = "12:34:56.789 ADMITTED chatcmpl-x.y-m1-abcdef-1234: 100 prompt"
        self.assertEqual(
            self.mod.console_line(line),
            "\033[90m12:34:56.789\033[0m ADMITTED chatcmpl-x.y-m1-abcdef-1234: 100 prompt")

    def test_unstamped_line_still_levels(self):
        # A TTY run has no stamp: the tag and the level word are still colored.
        self.assertEqual(self.mod.console_line("[strata] ERROR: the engine died"),
                         "\033[90m[strata]\033[0m \033[31mERROR\033[0m: the engine died")


class ServerWiring(unittest.TestCase):
    """serve/server.py's console chain (main): _StampedOut over _ColoredOut over
    the real stdout - the stamp is added first, then greyed and the level word
    colored. The repo's other serve tests import serve.server too."""

    @classmethod
    def setUpClass(cls):
        import serve.server as srv
        cls.srv = srv

    def test_stamped_line_colored_through_the_chain(self):
        import io
        from serve import log_color as lc
        buf = io.StringIO()
        chain = self.srv._StampedOut(self.srv._ColoredOut(buf))
        saved = lc._COLOR
        lc._COLOR = True                      # like VLLM_LOGGING_COLOR=1 at import
        try:
            chain.write("[strata] WARNING: RAM is tight\n")   # print writes text + newline separately
            chain.write("plain line\n")
        finally:
            lc._COLOR = saved
        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[0],
                         r"^\x1b\[90m\d\d:\d\d:\d\d\.\d\d\d\x1b\[0m \x1b\[90m\[strata\]\x1b\[0m "
                         r"\x1b\[33mWARNING\x1b\[0m: RAM is tight$")
        self.assertRegex(lines[1], r"^\x1b\[90m\d\d:\d\d:\d\d\.\d\d\d\x1b\[0m plain line$")

    def test_partial_line_waits_then_flush_passes_it(self):
        import io
        from serve import log_color as lc
        buf = io.StringIO()
        out = self.srv._ColoredOut(buf)
        saved = lc._COLOR
        lc._COLOR = True
        try:
            out.write("no newline yet")
            self.assertEqual(buf.getvalue(), "")     # held until the newline
            out.flush()
        finally:
            lc._COLOR = saved
        self.assertEqual(buf.getvalue(), "no newline yet")   # unfinished text passes through as it is


if __name__ == "__main__":
    unittest.main()

