"""serve/test_completions.py - POST /v1/completions (serve/completions.py) against a mock engine (no GPU, no pack);
the --batch side (the engine's LP lines in admissions, a reply in a slot) is in test_parallel.py.

    python -m unittest serve.test_completions -v
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import IM_END, ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A, B, C, BAR = ord("A"), ord("B"), ord("C"), ord("|")


class ScriptEngine(MockEngine):
    """MockEngine that records each request, scores like the engine (score= -> `scores` in its figures), reports a
    reused prefix, and can hold a request until released (for the queue tests)."""

    def __init__(self, tok, script):
        super().__init__(tok, script)
        self.calls, self.logprobs, self.reused = [], {A: math.log(0.6), B: math.log(0.3), C: math.log(0.05)}, 7
        self.gate = None                                  # threading.Event: the next request waits for it
        self.started = threading.Event()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls.append((list(ids), max_new, dict(sampling)))
        self.started.set()
        if self.gate is not None:
            gate, self.gate = self.gate, None
            gate.wait(10)
        last = {"generated": 1, "prompt_tokens": len(ids), "finish": "length", "reused": self.reused}
        if sampling.get("score"):
            last["scores"] = {t: self.logprobs.get(t, -30.0) for t in sampling["score"]}
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)
        self.last = last


class Completions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = ScriptEngine(cls.tok, "Hi there|after")
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.im_end = cls.tok.encode(IM_END, parse_special=True)[0]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body, stream=False):
        req = urllib.request.Request(self.base + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
                if not stream:
                    return r.status, json.loads(raw)
                events = [json.loads(line[6:]) for line in raw.splitlines()
                          if line.startswith("data: ") and line != "data: [DONE]"]
                return r.status, events
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_decision_scores_the_allowed_ids(self):
        s, b = self.post({"prompt": [1, 2, 3], "max_tokens": 1, "temperature": 0.0, "allowed_token_ids": [A, B, C],
                          "logprobs": 20, "return_tokens_as_token_ids": True, "priority": -20})
        self.assertEqual(s, 200, b)
        ids, max_new, sampling = self.engine.calls[-1]
        self.assertEqual((ids, max_new, sampling), ([1, 2, 3], 1, {"score": [A, B, C], "priority": -20}))
        lp = b["choices"][0]["logprobs"]
        self.assertEqual(lp["tokens"], [f"token_id:{A}"])
        top = {int(k.split(":")[1]): v for k, v in lp["top_logprobs"][0].items()}
        z = 0.6 + 0.3 + 0.05                              # renormalized over the allowed ids, as vLLM's mask does
        for t, p in ((A, 0.6), (B, 0.3), (C, 0.05)):
            self.assertAlmostEqual(math.exp(top[t]), p / z, places=6)
        self.assertEqual(b["choices"][0]["text"], "A")
        self.assertEqual(b["usage"]["prompt_tokens_details"]["cached_tokens"], 3)   # min(reused, prompt)

    def test_decision_logit_bias_and_text_keys(self):
        s, b = self.post({"prompt": [1, 2], "max_tokens": 1, "allowed_token_ids": [A, B], "logprobs": 2,
                          "logit_bias": {str(B): 5.0}})
        self.assertEqual(s, 200, b)
        lp = b["choices"][0]["logprobs"]
        self.assertEqual(lp["tokens"], ["B"])            # the bias moves the choice; keys are the tokens' text
        self.assertEqual(set(lp["top_logprobs"][0]), {"A", "B"})

    def test_stream_stops_on_a_requested_id(self):
        s, ev = self.post({"prompt": [5, 6], "max_tokens": 50, "stop_token_ids": [BAR], "stream": True,
                           "skip_special_tokens": False, "stream_options": {"include_usage": True}}, stream=True)
        self.assertEqual(s, 200, ev)
        text = "".join(c["choices"][0]["text"] for c in ev)
        self.assertEqual(text, "Hi there")               # the stop token ends the text, as vLLM's
        last = ev[-1]
        self.assertEqual((last["choices"][0]["finish_reason"], last["choices"][0]["stop_reason"]), ("stop", BAR))
        self.assertEqual(last["usage"]["completion_tokens"], len("Hi there|"))
        self.assertEqual(last["usage"]["prompt_tokens_details"]["cached_tokens"], 2)

    def test_end_of_turn_is_a_stop(self):
        s, b = self.post({"prompt": "<|im_start|>user\nhi", "max_tokens": 50})
        self.assertEqual(s, 200, b)
        self.assertEqual(b["choices"][0]["text"], "Hi there|after")
        self.assertEqual((b["choices"][0]["finish_reason"], b["choices"][0]["stop_reason"]), ("stop", self.im_end))
        self.assertEqual(self.engine.calls[-1][0][0], 256)   # the string prompt's specials are parsed

    def test_length(self):
        s, b = self.post({"prompt": [1], "max_tokens": 2, "temperature": 0.0, "top_p": 0.9})
        self.assertEqual(s, 200, b)
        self.assertEqual((b["choices"][0]["text"], b["choices"][0]["finish_reason"]), ("Hi", "length"))
        self.assertEqual(self.engine.calls[-1][2], {"temperature": 0.0, "top_p": 0.9})

    def test_logit_bias_and_priority_together(self):
        s, b = self.post({"prompt": [1], "max_tokens": 2, "logit_bias": {str(A): -100}, "priority": -10})
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.calls[-1][2], {"logit_bias": {str(A): -100}, "priority": -10})

    def test_bad_requests_are_400(self):
        for body in ({"prompt": []}, {"prompt": [1], "max_tokens": 0}, {"prompt": [1], "n": 2},
                     {"prompt": [1], "ignore_eos": True}, {"prompt": [1], "max_tokens": 2, "allowed_token_ids": [A]},
                     {"prompt": [1], "logprobs": 5}):
            s, b = self.post(body)
            self.assertEqual(s, 400, body)
        from serve import completions                     # the vocabulary check (ByteTokenizer has no vocab list)
        for body in ({"prompt": [300]}, {"prompt": [1], "stop_token_ids": [300]}):
            self.assertRaises(ValueError, completions.parse, body, self.tok, 300, 4096)

    def test_cache_prefix_is_a_rolling_checkpoint_for_the_engine(self):
        """"strata_cache_prefix": N reaches the engine as `roll=N` (a decision's context, its probe after it), also
        beside a decision's scores; a value outside the prompt is a 400."""
        s, b = self.post({"prompt": [1, 2, 3, 4], "max_tokens": 1, "allowed_token_ids": [A, B],
                          "strata_cache_prefix": 2, "strata_checkpoint": False})
        self.assertEqual(s, 200, b)
        sampling = self.engine.calls[-1][2]
        self.assertEqual((sampling.get("strata_cache_prefix"), sampling.get("score")), (2, [A, B]))
        from serve.server import StrataEngine
        keys = StrataEngine.sampling_keys(sampling)
        self.assertIn(" roll=2", keys)
        self.assertIn(" ckpt=0", keys)
        for cp in (0, 4, -1, True, "2"):
            s, b = self.post({"prompt": [1, 2, 3, 4], "max_tokens": 1, "strata_cache_prefix": cp})
            self.assertEqual(s, 400, cp)


if __name__ == "__main__":
    unittest.main()
