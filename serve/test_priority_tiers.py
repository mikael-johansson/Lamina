"""serve/test_priority_tiers.py - priority tiers on the admission path (API-PARITY-PLAN.md section 5; the ada
vLLM fork's short-prefill-priority-tiers.patch semantics): body `priority` (lower admits earlier, default 0),
tiers T0 <= -20 / T1 -19..-1 / T2 0..9 / T3 >= 10, gate PRIORITY_SCHED=1 (off = the field is accepted and
inert, never a 400), the T0/T1 reserved slot (PRIO_RESERVED_SLOTS) and short-first kept as the within-tier
tiebreak.  Mock-engine harness (serve/test_admit_order.py's engine_without_process): no process, no GPU.

    python -m unittest serve.test_priority_tiers -v
"""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, prio_tier, read_priority  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.test_admit_order import engine_without_process  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class TierBoundaries(unittest.TestCase):
    """The vLLM patch's tier boundaries: T0 <= -20, T1 -19..-1, T2 0..9 (default), T3 >= 10."""

    def test_boundaries(self):
        self.assertEqual(prio_tier(-20), 0)
        self.assertEqual(prio_tier(-1000), 0)
        self.assertEqual(prio_tier(-19), 1)
        self.assertEqual(prio_tier(-1), 1)
        self.assertEqual(prio_tier(0), 2)
        self.assertEqual(prio_tier(9), 2)
        self.assertEqual(prio_tier(10), 3)
        self.assertEqual(prio_tier(100), 3)

    def test_the_body_field_is_read_as_an_int_never_rejected(self):
        # vLLM's off-gate quirk: the field is accepted whatever it is; a non-number is T2 (0), not a 400
        self.assertEqual(read_priority(-20), -20)
        self.assertEqual(read_priority(None), 0)
        self.assertEqual(read_priority("voice"), 0)
        self.assertEqual(read_priority(True), 0)
        self.assertEqual(read_priority(7.9), 7)


class ParseRead(unittest.TestCase):
    """The priority read at request parse (with_shared): every API passes through it; the value is normalised
    and rides the request dict (the sampling dict) down to the engine; the request is not mutated."""

    def svc(self):
        tok = ByteTokenizer()
        return Service(MockEngine(tok, "xy" * 10, max_context=4096), tok,
                       ChatTemplate(ROOT / "serve" / "chat_template.jinja"))

    def test_priority_is_normalised_on_the_way_through(self):
        s = self.svc()
        out = s.with_shared({"messages": [], "priority": -20}, "openai")
        self.assertEqual(out["priority"], -20)
        out = s.with_shared({"messages": [], "priority": "voice"}, "openai")
        self.assertEqual(out["priority"], 0)                      # a non-number: T2, not a 400
        req = {"messages": [], "priority": "10"}
        self.assertEqual(s.with_shared(req, "anthropic")["priority"], 10)
        self.assertEqual(req["priority"], "10")                   # (untouched: the copy carries the int)
        self.assertNotIn("priority", s.with_shared({"messages": []}, "openai"))   # none sent: none added


class AdmitTierOrder(unittest.TestCase):
    """With PRIORITY_SCHED on, _take_control / _admit_next admit by tier first; short-first stays the tiebreak
    inside a tier (and arrival order when short-first is off).  Same mock-engine harness as test_admit_order."""

    def setUp(self):
        self.e = engine_without_process(16)
        self.order, self.cancels = [], []
        self.order_lock = threading.Lock()

    def tearDown(self):
        for c in self.cancels:
            c.set()

    def take(self, name, plen, **kw):
        c = threading.Event()
        self.cancels.append(c)

        def go():
            gen = self.e._take_control(c, plen, **kw)
            try:
                while True:
                    next(gen)
            except StopIteration as s:
                if s.value:
                    with self.order_lock:
                        self.order.append(name)
                    time.sleep(0.05)                # its read
                    self.e.ctl.release()
        threading.Thread(target=go, daemon=True).start()

    def until(self, cond, s=5.0):
        end = time.time() + s
        while time.time() < end and not cond():
            time.sleep(0.01)
        self.assertTrue(cond())

    def queue(self, *reqs):
        """(name, plen, priority): while a read holds the control lines the requests arrive in this order;
        then the read ends and each takes its turn."""
        self.assertTrue(self.e.ctl.acquire(timeout=1))
        for name, plen, prio in reqs:
            self.take(name, plen, new=True, priority=prio)
            self.until(lambda: self.e.waiting >= 1)
            time.sleep(0.02)                        # (distinct arrival times)
        self.until(lambda: self.e.waiting == len(reqs))
        self.e.ctl.release()
        self.until(lambda: len(self.order) == len(reqs), s=15.0)
        return self.order

    def test_tier_ordering_beats_length_and_arrival(self):
        """A long T0 (the after_reply decision carries the whole spoken reply) goes before everything; T1 before
        T2 before T3 - even a short T3 arriving first."""
        with mock.patch.object(server, "PRIO_SCHED", True), \
             mock.patch.object(server, "ADMIT_SHORT_FIRST", 0), \
             mock.patch.object(server, "ADMIT_MAX_WAIT_S", 60.0):
            order = self.queue(("bg", 100, 10), ("agent", 500, 0), ("reply", 300, -10),
                               ("decision", 190000, -20))
        self.assertEqual(order, ["decision", "reply", "agent", "bg"])

    def test_within_tier_short_first_is_kept(self):
        with mock.patch.object(server, "PRIO_SCHED", True), \
             mock.patch.object(server, "ADMIT_SHORT_FIRST", 8192), \
             mock.patch.object(server, "ADMIT_MAX_WAIT_S", 60.0):
            order = self.queue(("agent1", 120000, 0), ("chat", 500, 0), ("agent2", 190000, 0))
        self.assertEqual(order, ["chat", "agent1", "agent2"])   # short-first inside T2, arrival among the long

    def test_within_tier_arrival_order_when_short_first_off(self):
        with mock.patch.object(server, "PRIO_SCHED", True), \
             mock.patch.object(server, "ADMIT_SHORT_FIRST", 0):
            order = self.queue(("a", 190000, 0), ("b", 500, 0))
        self.assertEqual(order, ["a", "b"])                     # pure arrival order inside one tier

    def test_gate_off_the_field_is_inert(self):
        """PRIORITY_SCHED off (the default): the tier heap is not consulted, so a T0 does not cut the line -
        the request still runs (the HTTP 200 half is in HttpInertWhenGateOff)."""
        with mock.patch.object(server, "PRIO_SCHED", False), \
             mock.patch.object(server, "ADMIT_SHORT_FIRST", 0):
            order = self.queue(("agent", 500, 0), ("decision", 190000, -20))
        self.assertEqual(sorted(order), ["agent", "decision"])  # both get their turn, arrival order decides


class ReservedSlot(unittest.TestCase):
    """The T0 reserved slot (vLLM patch rule 3): T2/T3 together hold at most batch - PRIO_RESERVED_SLOTS slots;
    a lower-tier request that would take a reserved one is refused (parks), a T0 behind it still gets in."""

    def engine(self, n):
        e = engine_without_process(n)
        e.slot_order = list(range(n))
        e.slot_gs, e.slot_groups = 0, 1
        e.slot_group = [0] * n
        e.slot_held = [[] for _ in range(n)]
        e.slot_used = [0.0] * n
        return e

    def claim(self, e, slot, tier):
        e.slot_busy[slot] = True
        e.slot_claims.add(slot)
        e.slot_tier[slot] = tier

    def test_t2_is_refused_the_reserved_slot_t0_gets_it(self):
        e = self.engine(2)
        with mock.patch.object(server, "PRIO_SCHED", True), \
             mock.patch.object(server, "PRIO_RESERVED_SLOTS", 1):
            self.claim(e, 0, 2)                                 # one agent holds its slot
            self.assertEqual(e.pick_slot([1, 2, 3], 2), None)   # a second agent: parked, the slot is reserved
            self.assertEqual(e.pick_slot([1, 2, 3], 3), None)   # ... same for T3
            self.assertEqual(e.pick_slot([1, 2, 3], 0), 1)      # the voice decision behind it still gets in
            self.claim(e, 1, 0)
            self.assertEqual(e.pick_slot([1, 2, 3], 0), None)   # the pool really is full now

    def test_two_reserved_slots(self):
        e = self.engine(4)
        with mock.patch.object(server, "PRIO_SCHED", True), \
             mock.patch.object(server, "PRIO_RESERVED_SLOTS", 2):
            self.claim(e, 0, 2)
            self.claim(e, 1, 3)
            self.assertEqual(e.pick_slot([1], 2), None)         # 2 held >= 4-2: T2 parked
            self.assertEqual(e.pick_slot([1], 1), 2)            # T1 may use the reserved slots too

    def test_gate_off_no_cap(self):
        e = self.engine(2)
        with mock.patch.object(server, "PRIO_SCHED", False), \
             mock.patch.object(server, "PRIO_RESERVED_SLOTS", 1):
            self.claim(e, 0, 2)
            self.assertEqual(e.pick_slot([1, 2, 3], 2), 1)      # inert: the cap is not applied

    def test_untiered_busy_counts_as_lower(self):
        """A busy flag with no recorded tier (parked after a BGEN refusal, an orphan) counts against the lower
        share, so a leak can never eat the reserved slot."""
        e = self.engine(2)
        e.slot_busy[0] = True                                   # claimed by nobody, tiered by nobody
        with mock.patch.object(server, "PRIO_SCHED", True), \
             mock.patch.object(server, "PRIO_RESERVED_SLOTS", 1):
            self.assertEqual(e.pick_slot([1], 2), None)
            self.assertEqual(e.pick_slot([1], 0), 1)


class HttpInertWhenGateOff(unittest.TestCase):
    """The parity quirk over HTTP: a priority on a gate-off server is accepted and inert (200, never 400), so
    clients never branch on the backend; the same body runs fine with the gate on."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "xy" * 100, max_context=4096), tok,
                          ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        cls.httpd = server.serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def call(self, extra):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra}
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())

    def test_priority_is_accepted_and_runs_with_the_gate_off(self):
        s, b = self.call({"priority": -20})
        self.assertEqual(s, 200, b)
        self.assertIn("choices", b)

    def test_a_non_number_priority_is_not_a_400(self):
        s, b = self.call({"priority": "voice"})
        self.assertEqual(s, 200, b)                             # vLLM off-gate quirk: inert, not rejected

    def test_the_same_body_with_the_gate_on(self):
        with mock.patch.object(server, "PRIO_SCHED", True):
            for prio in (-20, -10, 0, 10):
                s, b = self.call({"priority": prio})
                self.assertEqual(s, 200, b)


if __name__ == "__main__":
    unittest.main()
