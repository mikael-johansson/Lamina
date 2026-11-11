// The serve engine's step-time breakdown (STRATA_STEP_BREAKDOWN=1): the sweep-line ledger over overlapping threads,
// the window's end while a read is in flight, and the status lines (CPU only, synthetic times in seconds).
#include "strata/core/step_breakdown.hpp"

#include <cmath>
#include <cstdio>
#include <string>

using strata::core::StepBreakdown;

namespace {
int failures = 0;

void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}
bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }
bool has(const std::vector<std::string>& lines, const std::string& s) {
    for (const auto& l : lines)
        if (l.find(s) != std::string::npos) return true;
    return false;
}
void dump(const std::vector<std::string>& lines) {
    for (const auto& l : lines) std::printf("      | %s\n", l.c_str());
}
}  // namespace

int main() {
    std::printf("step_breakdown_test\n");

    {   // an idle window prints nothing
        StepBreakdown bd(0.0);
        StepBreakdown::Closed c;
        check(bd.close(10.0, c) && near(c.window_s, 10.0), "an idle window closes at the tick");
        check(StepBreakdown::format(c).empty(), "an idle window prints nothing");
    }
    {   // one solo request: a read 2 s, other 0.5 s, decode 4 s of 400 tokens in 100 windows
        StepBreakdown bd(0.0);
        bd.main_busy(1.0, true);
        bd.reading(1.5, true, 100);
        bd.read_at(2.5, 1100);
        bd.read_at(3.5, 2100);
        bd.reading(3.5, false, 2100, 2001);   // the prompt's last token goes through the first decode window
        bd.solo_decoding(3.5, true);
        for (int i = 1; i <= 100; ++i) bd.window_done(3.5 + 0.04 * i, 4);
        bd.solo_decoding(7.5, false);
        bd.main_busy(7.5, false);
        StepBreakdown::Closed c;
        check(bd.close(10.0, c), "a window with no read in flight closes at the tick");
        const auto& a = c.acc;
        check(near(a.prefill_s, 2.0) && a.prefill_tok == 2000 && a.prefill_it == 2, "the read: 2 s, 2000 tok, 2 it");
        check(a.done_reads == 1 && a.done_read_tok == 2001, "the read is completed with the prompt's fresh tokens");
        check(near(a.other_s, 0.5), "the request's setup before the read is Other");
        check(near(a.gen_s[1], 4.0) && a.gen_tok[1] == 400 && a.gen_it[1] == 100, "the decode: 1 req, 4 s, 400 tok");
        check(near(a.idle_s, 3.5) && near(c.window_s - a.busy_s(), 3.5), "the rest is idle");
        const auto lines = StepBreakdown::format(c);
        dump(lines);
        check(has(lines, "10.0 s: RUNNING: 0 reqs (0 gen, 0 prefill)"), "status line");
        check(has(lines, "- 20.0% Prefill @ 1000 PPt/s (2 it). Completed: 1 req / 2001 tok"), "prefill line");
        check(has(lines, "- 40.0% Gen @ 100.0 tg/s (1 req: 100 it @ 100.0 tg/s)"), "gen line");
        check(has(lines, "- 5.0% Other"), "other line");
        check(!has(lines, "Stalled"), "no stall, no stalled line");
        check(has(lines, "- 35.0% Idle"), "idle line");
    }
    {   // two slots overlapping: slot 0 alone 2 s, both 3 s, slot 1 alone 1 s; their A/B stages never double-count
        StepBreakdown bd(0.0);
        bd.slot_active(0.0, 0, true);
        bd.slot_gen(0.0, 0, true);
        bd.window_done(1.0, 50);
        bd.slot_active(2.0, 1, true);
        bd.slot_gen(2.0, 1, true);
        bd.slot_gen(2.0, 1, true);   // a second pick of a running slot changes nothing
        bd.window_done(3.0, 40);
        bd.window_done(4.0, 40);
        bd.slot_active(5.0, 0, false);   // its BDONE: its windows end too
        bd.window_done(5.5, 30);
        bd.slot_active(6.0, 1, false);
        StepBreakdown::Closed c;
        bd.close(10.0, c);
        const auto& a = c.acc;
        check(near(a.gen_s[1], 3.0) && near(a.gen_s[2], 3.0), "time per concurrency: 1 req 3 s, 2 req 3 s");
        check(a.gen_tok[1] == 80 && a.gen_tok[2] == 80 && a.gen_it[2] == 2, "tokens by the decoding count at the end");
        const auto lines = StepBreakdown::format(c);
        dump(lines);
        check(has(lines, "- 60.0% Gen @ 26.7 tg/s (1 req: 2 it @ 26.7 tg/s, 2 req: 2 it @ 26.7 tg/s)"), "gen buckets");
    }
    {   // a read beside a decoding slot: its time is prefill, with decode; the slot's tokens stay out of Gen
        StepBreakdown bd(0.0);
        bd.slot_active(0.0, 0, true);
        bd.slot_gen(0.0, 0, true);
        bd.window_done(1.0, 10);
        bd.main_busy(1.0, true);
        bd.reading(1.0, true, 0);
        bd.window_done(1.5, 7);
        bd.read_at(2.0, 512);
        bd.reading(2.0, false, 512);
        bd.main_busy(2.0, false);
        bd.window_done(3.0, 10);
        bd.slot_active(3.0, 0, false);
        StepBreakdown::Closed c;
        bd.close(10.0, c);
        const auto& a = c.acc;
        check(near(a.prefill_s, 1.0) && a.prefill_mixed == 1 && a.mixed_tok == 7, "read-over-window is mixed");
        check(near(a.gen_s[1], 2.0) && a.gen_tok[1] == 20, "the slot's time and tokens outside the read are Gen");
        const auto lines = StepBreakdown::format(c);
        dump(lines);
        check(has(lines, "(1 it, 1 with decode: 7 tok)"), "the mixed count in the prefill line");
    }
    {   // the slots paused for the main thread's turn with nothing running: stalled
        StepBreakdown bd(0.0);
        bd.slot_active(0.0, 0, true);
        bd.slot_gen(0.0, 0, true);
        bd.slots_stopped(1.0);   // paused
        bd.slot_gen(1.4, 0, true);   // resumed
        bd.slots_stopped(2.0);
        bd.slot_gen(2.1, 0, true);
        bd.slot_active(3.0, 0, false);
        StepBreakdown::Closed c;
        bd.close(10.0, c);
        check(near(c.acc.stalled_s, 0.5) && c.acc.stalls == 2, "two pauses: 0.5 s stalled in 2 stalls");
        const auto lines = StepBreakdown::format(c);
        dump(lines);
        check(has(lines, "- 5.0% Stalled (2 stalls"), "stalled line");
    }
    {   // a read in flight at the tick: the window ends at its last chunk boundary, the rest carries over
        StepBreakdown bd(0.0);
        bd.main_busy(0.0, true);
        bd.reading(0.0, true, 0);
        bd.read_at(4.0, 4000);
        bd.read_at(8.0, 8000);   // the next chunk runs 8 .. 13
        StepBreakdown::Closed c;
        check(bd.close(10.0, c) && near(c.window_s, 8.0), "closed at the last chunk boundary (8 s)");
        check(near(c.acc.prefill_s, 8.0) && c.acc.prefill_tok == 8000 && c.prefill == 1, "its 8 s and 8000 tok");
        const auto l1 = StepBreakdown::format(c);
        dump(l1);
        check(has(l1, "8.0 s: RUNNING: 1 reqs (0 gen, 1 prefill)") && has(l1, "100.0% Prefill @ 1000 PPt/s"),
              "the first window's lines");
        bd.read_at(13.0, 13000);
        bd.reading(13.0, false, 13000);
        bd.main_busy(13.0, false);
        check(bd.close(20.0, c) && near(c.window_s, 12.0), "the next window starts at 8 s");
        check(near(c.acc.prefill_s, 5.0) && c.acc.prefill_tok == 5000 && c.acc.done_reads == 1 &&
                  c.acc.done_read_tok == 13000, "the carried chunk lands whole in the next window");
    }
    {   // one chunk longer than the tick: no boundary in the window, so it is not closed
        StepBreakdown bd(0.0);
        bd.main_busy(0.0, true);
        bd.reading(0.5, true, 0);
        StepBreakdown::Closed c;
        check(bd.close(10.0, c) && near(c.window_s, 0.5), "a read that starts in the window cuts it at its start");
        check(!bd.close(20.0, c), "a chunk spanning the whole window: not closed");
        bd.read_at(25.0, 30000);
        check(bd.close(30.0, c) && near(c.window_s, 24.5) && near(c.acc.prefill_s, 24.5), "closed once it ends");
    }

    if (failures) std::printf("step_breakdown_test: %d FAILED\n", failures);
    else std::printf("step_breakdown_test: all ok\n");
    return failures ? 1 : 0;
}
