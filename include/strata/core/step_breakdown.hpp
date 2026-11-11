// strata/core/step_breakdown.hpp - where the serve engine's wall clock went, one logging window at a time
// (STRATA_STEP_BREAKDOWN=1), in the shape of the vLLM fork's step-time-breakdown status lines:
//
//   strata steps: 10.0 s: RUNNING: 2 reqs (2 gen, 0 prefill)
//   strata steps:   - 12.4% Prefill @ 2410 PPt/s (4 it, 2 with decode). Completed: 2 req / 312 tok
//   strata steps:   - 55.1% Gen @ 187.4 tg/s (1 req: 40 it @ 201.2 tg/s, 2 req: 18 it @ 170.5 tg/s)
//   strata steps:   - 0.9% Other (the main thread's request work besides reads and windows)
//   strata steps:   - 1.4% Stalled (3 stalls: requests present, nothing running)
//   strata steps:   - 30.2% Idle
//
// vLLM has one scheduler loop; this engine runs the main thread (prompt reads, the solo decode) beside the batch
// slots' two stage threads, on purpose at the same time.  So the time is not summed per thread: one ledger holds what
// is running now, and every event first charges the time since the previous event to exactly one category, from that
// state (a sweep line):
//
//   prefill   a prompt read on the main thread (decode windows beside it make its chunks "with decode"; their time
//             is the read's, as in vLLM, and their tokens are kept out of the Gen rates)
//   gen[n]    decode windows in flight, n = the requests decoding (the batch slots' plus the solo request's)
//   other     the main thread holds a request but neither reads nor decodes (setup, the K/V's growth, a slot's
//             handover copies) - vLLM charges that loop overhead to its steps; here it is shown, from 0.1%
//   stalled   requests present (a batch slot active) and nothing running: the slots paused for the main thread's
//             turn, a slot between windows with nothing due.  Shown from 0.1%
//   idle      the rest
//
// A window ends at the tick, except while a read is in flight: then it ends at the read's last chunk boundary and
// the rest carries into the next window (vLLM's rule: a chunk's tokens arrive with its end, so a window cut inside
// it would show its time without its tokens).  A window with no chunk boundary at all (one chunk longer than the
// tick) is not closed; the next tick tries again.  Decode windows (~10-20 ms) are cut at the tick.
//
// Host C++ only, no clock of its own (the caller passes `now` in seconds): the engine and the unit test drive it the
// same way.  Not thread-safe: the engine holds one mutex around every call.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::core {

class StepBreakdown {
public:
    static constexpr int kMaxN = 16;   // gen buckets 1..kMaxN (more requests share the last)
    static constexpr int kMaxSlots = 64;

    struct Acc {
        double prefill_s = 0, other_s = 0, stalled_s = 0, idle_s = 0;
        int64_t prefill_it = 0, prefill_mixed = 0, prefill_tok = 0, mixed_tok = 0;
        double gen_s[kMaxN + 1] = {};
        int64_t gen_it[kMaxN + 1] = {}, gen_tok[kMaxN + 1] = {};
        int64_t stalls = 0;
        int64_t done_reads = 0, done_read_tok = 0;   // reads that ended, and their fresh tokens

        double busy_s() const {
            double s = prefill_s + other_s + stalled_s;
            for (int n = 0; n <= kMaxN; ++n) s += gen_s[n];
            return s;
        }
        bool any() const {
            if (prefill_it || done_reads || prefill_s > 0 || other_s > 0 || stalled_s > 0) return true;
            for (int n = 0; n <= kMaxN; ++n)
                if (gen_it[n] || gen_s[n] > 0) return true;
            return false;
        }
        void minus(const Acc& b) {
            prefill_s -= b.prefill_s; other_s -= b.other_s; stalled_s -= b.stalled_s; idle_s -= b.idle_s;
            prefill_it -= b.prefill_it; prefill_mixed -= b.prefill_mixed; prefill_tok -= b.prefill_tok;
            mixed_tok -= b.mixed_tok; stalls -= b.stalls; done_reads -= b.done_reads; done_read_tok -= b.done_read_tok;
            for (int n = 0; n <= kMaxN; ++n) {
                gen_s[n] -= b.gen_s[n];
                gen_it[n] -= b.gen_it[n];
                gen_tok[n] -= b.gen_tok[n];
            }
        }
    };

    struct Closed {
        Acc acc;
        double window_s = 0;
        int running = 0, gen = 0, prefill = 0;   // at the close: requests held, decoding, reading
    };

    explicit StepBreakdown(double now) : last_(now), wstart_(now) {}

    // ---- the state's edges (each charges the time before it first)
    void main_busy(double now, bool on) { charge(now); main_ = on; }
    /// a prompt read starts (`on`, `pos` = its first fresh position) or ends (`pos` = where it ended; `fresh`: the
    /// prompt's tokens not from the cache, for "Completed" - the FINISHED line's count, which also holds the ones
    /// its first decode window reads; -1: the read's own)
    void reading(double now, bool on, int64_t pos, int64_t fresh = -1) {
        if (on) {
            charge(now);
            reading_ = true;
            read_pos_ = read_from_ = pos;
            mixed_ = false;
            snap_ = acc_;   // the read's start is a boundary too: none of its time is in the window yet
            snap_t_ = now;
        } else if (reading_) {
            read_at(now, pos);
            reading_ = false;
            acc_.done_reads += 1;
            acc_.done_read_tok += fresh >= 0 ? fresh : std::max<int64_t>(0, read_pos_ - read_from_);
        }
    }
    /// the read's first fresh position, once known (the prompt cache's restore decides it, after the read started)
    void read_base(double now, int64_t pos) {
        charge(now);
        if (reading_) read_pos_ = read_from_ = pos;
    }
    /// a read's progress: the chunk or feed window through `pos` is done (a chunk boundary: a window may end here)
    void read_at(double now, int64_t pos) {
        charge(now);
        if (!reading_) return;
        if (pos > read_pos_) {
            acc_.prefill_tok += pos - read_pos_;
            acc_.prefill_it += 1;
            if (mixed_ || gen_slots_ > 0) acc_.prefill_mixed += 1;
            read_pos_ = pos;
        }
        mixed_ = false;
        snap_ = acc_;
        snap_t_ = now;
    }
    void solo_decoding(double now, bool on) { charge(now); solo_ = on; }
    /// a batch slot's windows run (from its first window's pick until it ends or the slots pause)
    void slot_gen(double now, int slot, bool on) { charge(now); flip(gen_, gen_slots_, slot, on); }
    /// a batch slot holds a request (from its admission until its BDONE)
    void slot_active(double now, int slot, bool on) {
        charge(now);
        flip(active_, active_slots_, slot, on);
        if (!on) flip(gen_, gen_slots_, slot, false);
    }
    /// every slot's windows stopped (the slots paused, or the engine reset them)
    void slots_stopped(double now) {
        charge(now);
        for (int i = 0; i < kMaxSlots; ++i) gen_[i] = false;
        gen_slots_ = 0;
    }
    /// a decode window ended with `tokens` emitted
    void window_done(double now, int64_t tokens) {
        charge(now);
        if (reading_) {   // beside a read: its time is the read's, so its tokens stay out of the Gen rates
            acc_.mixed_tok += tokens;
            return;
        }
        const int n = std::clamp(decoding(), 1, kMaxN);
        acc_.gen_it[n] += 1;
        acc_.gen_tok[n] += tokens;
    }

    /// The window up to `now` (see the header for where it ends); false: not closed (a read's chunk spans it)
    bool close(double now, Closed& out) {
        charge(now);
        double end = now;
        Acc carry{};
        if (reading_) {
            if (snap_t_ <= wstart_) return false;
            end = snap_t_;
            carry = acc_;
            carry.minus(snap_);
            out.acc = snap_;
        } else {
            out.acc = acc_;
        }
        out.window_s = std::max(0.0, end - wstart_);
        out.running = active_slots_ + (main_ ? 1 : 0);
        out.gen = gen_slots_ + (solo_ ? 1 : 0);
        out.prefill = reading_ ? 1 : 0;
        acc_ = carry;
        snap_ = Acc{};
        wstart_ = snap_t_ = end;
        return true;
    }

    /// the lines after the status line's prefix; none for a window with no work in it (quiet when idle)
    static std::vector<std::string> format(const Closed& c) {
        std::vector<std::string> lines;
        const Acc& a = c.acc;
        if (c.window_s <= 0 || !a.any()) return lines;
        auto pct = [&](double s) { return 100.0 * s / c.window_s; };
        auto rate = [](double tok, double s) { return s > 0 ? tok / s : 0.0; };
        char buf[512];
        std::snprintf(buf, sizeof buf, "%.1f s: RUNNING: %d reqs (%d gen, %d prefill)", c.window_s, c.running, c.gen,
                      c.prefill);
        lines.emplace_back(buf);
        if (a.prefill_it || a.done_reads || a.prefill_s > 0) {
            std::string l;
            std::snprintf(buf, sizeof buf, "  - %.1f%% Prefill @ %.0f PPt/s (%lld it", pct(a.prefill_s),
                          rate((double) a.prefill_tok, a.prefill_s), (long long) a.prefill_it);
            l = buf;
            if (a.prefill_mixed) {
                std::snprintf(buf, sizeof buf, ", %lld with decode: %lld tok", (long long) a.prefill_mixed,
                              (long long) a.mixed_tok);
                l += buf;
            }
            l += ")";
            if (a.done_reads) {
                std::snprintf(buf, sizeof buf, ". Completed: %lld req / %lld tok", (long long) a.done_reads,
                              (long long) a.done_read_tok);
                l += buf;
            }
            lines.push_back(l);
        }
        double gs = 0;
        int64_t gt = 0, git = 0;
        std::string per;
        for (int n = 1; n <= kMaxN; ++n) {
            if (!a.gen_it[n] && a.gen_s[n] <= 0) continue;
            gs += a.gen_s[n];
            gt += a.gen_tok[n];
            git += a.gen_it[n];
            std::snprintf(buf, sizeof buf, "%s%d req: %lld it @ %.1f tg/s", per.empty() ? "" : ", ", n,
                          (long long) a.gen_it[n], rate((double) a.gen_tok[n], a.gen_s[n]));
            per += buf;
        }
        if (git || gs > 0) {
            std::snprintf(buf, sizeof buf, "  - %.1f%% Gen @ %.1f tg/s (%s)", pct(gs), rate((double) gt, gs), per.c_str());
            lines.emplace_back(buf);
        }
        if (pct(a.other_s) >= 0.05) {
            std::snprintf(buf, sizeof buf, "  - %.1f%% Other (request setup and handover on the main thread)",
                          pct(a.other_s));
            lines.emplace_back(buf);
        }
        if (pct(a.stalled_s) >= 0.05) {
            std::snprintf(buf, sizeof buf, "  - %.1f%% Stalled (%lld stalls: requests present, nothing running)",
                          pct(a.stalled_s), (long long) a.stalls);
            lines.emplace_back(buf);
        }
        std::snprintf(buf, sizeof buf, "  - %.1f%% Idle", std::max(0.0, pct(c.window_s - a.busy_s())));
        lines.emplace_back(buf);
        return lines;
    }

    int decoding() const { return gen_slots_ + (solo_ ? 1 : 0); }

private:
    enum Kind { kIdle, kPrefill, kGen, kOther, kStalled };
    Kind kind() const {
        if (reading_) return kPrefill;
        if (decoding() > 0) return kGen;
        if (main_) return kOther;
        if (active_slots_ > 0) return kStalled;
        return kIdle;
    }
    void charge(double now) {
        const double dt = now - last_;
        if (dt <= 0) return;   // (a clock read before the lock was taken: the order of the calls decides)
        last_ = now;
        const Kind k = kind();
        switch (k) {
        case kPrefill:
            acc_.prefill_s += dt;
            if (gen_slots_ > 0) mixed_ = true;
            break;
        case kGen: acc_.gen_s[std::clamp(decoding(), 1, kMaxN)] += dt; break;
        case kOther: acc_.other_s += dt; break;
        case kStalled:
            acc_.stalled_s += dt;
            if (prev_ != kStalled) acc_.stalls += 1;
            break;
        case kIdle: acc_.idle_s += dt; break;
        }
        prev_ = k;
    }
    static void flip(bool* set, int& count, int slot, bool on) {
        if (slot < 0 || slot >= kMaxSlots || set[slot] == on) return;
        set[slot] = on;
        count += on ? 1 : -1;
    }

    Acc acc_, snap_;
    double last_, wstart_, snap_t_ = 0;
    bool main_ = false, reading_ = false, solo_ = false, mixed_ = false;
    bool gen_[kMaxSlots] = {}, active_[kMaxSlots] = {};
    int gen_slots_ = 0, active_slots_ = 0;
    int64_t read_pos_ = 0, read_from_ = 0;
    Kind prev_ = kIdle;
};

}  // namespace strata::core
