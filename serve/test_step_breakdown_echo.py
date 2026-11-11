"""STRATA_STEP_BREAKDOWN: the engine's "strata steps:" stderr lines reach the console as well, the status line with
the queue only the server sees (WAITING) - stdlib only, no engine, no GPU."""

import contextlib
import io
import unittest

from serve import server


class StepsEcho(unittest.TestCase):
    def setUp(self):
        self.eng = server.StrataEngine.__new__(server.StrataEngine)   # no engine process: the echo needs none
        self.eng.waiting, self.eng.admitting = 2, 1

    def echo(self, line):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.eng._steps_echo(line)
        return out.getvalue()

    def test_status_line_gets_waiting(self):
        self.assertEqual(self.echo("strata steps: 10.0 s: RUNNING: 2 reqs (2 gen, 0 prefill)\n"),
                         "[strata] 10.0 s: RUNNING: 2 reqs (2 gen, 0 prefill), WAITING: 3 reqs\n")

    def test_category_line_as_is(self):
        self.assertEqual(self.echo("strata steps:   - 55.1% Gen @ 187.4 tg/s (1 req: 40 it @ 201.2 tg/s)\n"),
                         "[strata]   - 55.1% Gen @ 187.4 tg/s (1 req: 40 it @ 201.2 tg/s)\n")

    def test_other_engine_lines_stay_in_the_log(self):
        self.assertEqual(self.echo("strata serve: slot 0: 12 generated in 147 ms\n"), "")

    def test_stamp_lines_calls_echo(self):
        seen, dst = [], io.StringIO()
        server._stamp_lines(io.StringIO("a\nstrata steps: x\n"), dst, seen.append)
        self.assertEqual(seen, ["a\n", "strata steps: x\n"])
        self.assertIn("strata steps: x", dst.getvalue())


if __name__ == "__main__":
    unittest.main()
