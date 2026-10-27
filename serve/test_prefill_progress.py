"""serve/test_prefill_progress.py - GET /v1/prefill_progress: the route gate (PREFILL_PROGRESS=1), the auth,
and the payload mapped from terminal_requests + the slot snapshot.

    python -m unittest serve.test_prefill_progress -v
"""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.prefill_progress import payload  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Route(unittest.TestCase):
    """The route exists only with PREFILL_PROGRESS=1 (read per request, like the vLLM launcher gate)."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "</think>\n\nhello there", max_context=4096), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path, headers=None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())

    def test_404_without_the_env_200_with_it(self):
        os.environ.pop("PREFILL_PROGRESS", None)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.get("/v1/prefill_progress")
        self.assertEqual(e.exception.code, 404)
        os.environ["PREFILL_PROGRESS"] = "1"
        try:
            code, body = self.get("/v1/prefill_progress")
            self.assertEqual(code, 200)
            for key in ("ts", "running", "waiting", "kv_cache_pct", "requests"):
                self.assertIn(key, body)
            self.assertIsInstance(body["requests"], list)
        finally:
            os.environ.pop("PREFILL_PROGRESS", None)

    def test_auth_like_other_v1_routes(self):
        self.svc.api_key = "secret"
        os.environ["PREFILL_PROGRESS"] = "1"
        try:
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.get("/v1/prefill_progress")
            self.assertEqual(e.exception.code, 401)
            code, _ = self.get("/v1/prefill_progress", headers={"Authorization": "Bearer secret"})
            self.assertEqual(code, 200)
            code, _ = self.get("/v1/prefill_progress", headers={"x-api-key": "secret"})
            self.assertEqual(code, 200)
        finally:
            self.svc.api_key = ""
            os.environ.pop("PREFILL_PROGRESS", None)


class _Engine:
    batch, waiting = 2, 1
    info = {"max_context": 4096}

    def slots_view(self):
        return [{"slot": 0, "state": "reading", "prompt_tokens": 500, "prompt_read": 120, "generated": 0,
                 "elapsed_s": 2.0, "tok_s": None},
                {"slot": 1, "state": "generating", "prompt_tokens": 300, "prompt_read": None, "generated": 7,
                 "elapsed_s": 3.0, "tok_s": 40.0}]


class _Svc:
    """Mock Service state: the same shape Service keeps under status_lock."""

    def __init__(self):
        self.status_lock = threading.Lock()
        self.status = {"busy": True, "queued": 1, "prompt_tokens": 300, "generated": 7}
        self.live_reqs = {1: ({}, None)}
        self.engine = _Engine()
        now = 1000.0
        self.terminal_requests = {
            "aaa": {"id": "aaa", "state": "queued", "queued_at": now - 9, "admitted_at": None,
                    "prompt_tokens": 800},
            "bbb": {"id": "bbb", "state": "admitted", "queued_at": now - 8, "admitted_at": now - 6,
                    "prompt_tokens": 500},
            "ccc": {"id": "ccc", "state": "waiting", "queued_at": now - 7, "admitted_at": now - 5,
                    "prompt_tokens": 300, "first_engine_token_s": 1.25},
        }

    def reported_ctx(self):
        return 4096


class Payload(unittest.TestCase):
    def setUp(self):
        self.body = payload(_Svc())

    def test_per_request_schema(self):
        by_id = {r["request_id"]: r for r in self.body["requests"]}
        self.assertEqual(set(by_id), {"aaa", "bbb", "ccc"})
        for r in by_id.values():
            for key in ("request_id", "phase", "admitted", "arrived_ms", "admitted_ms", "prompt_tokens",
                        "prefill_started_at", "first_token_at"):
                self.assertIn(key, r)
        # queued: not admitted, has a position in the queue
        a = by_id["aaa"]
        self.assertEqual(a["phase"], "queued")
        self.assertFalse(a["admitted"])
        self.assertIsNone(a["prefill_started_at"])
        self.assertEqual(a["position"], 0)
        # admitted, no first token yet: prefilling, with the slot's read progress
        b = by_id["bbb"]
        self.assertEqual(b["phase"], "prefill")
        self.assertTrue(b["admitted"])
        self.assertEqual(b["computed_tokens"], 120)
        self.assertEqual(b["remaining_prompt_tokens"], 380)
        self.assertIsNotNone(b["prefill_started_at"])
        self.assertIsNone(b["first_token_at"])
        # first token fired: decoding
        c = by_id["ccc"]
        self.assertEqual(c["phase"], "decode")
        self.assertEqual(c["generated_tokens"], 7)
        self.assertIsNotNone(c["first_token_at"])

    def test_no_disk_restore_fields(self):
        # the vLLM payload's restore keys are ABSENT on Strata (no disk tier), not zeroed
        for r in self.body["requests"]:
            self.assertNotIn("restoring_kv", r)
            self.assertNotIn("restore_ms", r)

    def test_top_level_counts(self):
        self.assertEqual(self.body["running"], 1)
        self.assertEqual(self.body["waiting"], 1 + 1)          # status.queued + engine.waiting
        self.assertAlmostEqual(self.body["kv_cache_pct"], round(min(1.0, 307 / 4096) * 100, 1))

    def test_never_raises(self):
        class Broken:
            @property
            def status_lock(self):
                raise RuntimeError("no lock")
        body = payload(Broken())
        self.assertEqual(body["requests"], [])
        self.assertIn("ts", body)


if __name__ == "__main__":
    unittest.main()
