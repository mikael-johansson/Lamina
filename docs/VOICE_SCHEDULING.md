# Voice beside agents: how the server schedules them

State as of 2026-10-28, 18:05: server `d631f5ca` on branch `layer-split`, with the engine from
`build/strata.dsplit-noleak` and config `configs/fork-iq3_s-split-prod.json`. The hardware is two RTX 3090s,
with the model split by layers across them.

The goal is that Nova's voice requests answer as fast as on an idle server while Hermes agents (one main agent plus
subagents, 25K–190K contexts) keep working. Agent long reads may wait while somebody talks; voice may not.

## 1. The engine's limits that everything below works around

- **Prompt reads use one pair of control lines.** The engine reads one prompt at a time (the solo path, or a batch
  slot's admission, which reads in the main session and then copies the state into the slot). Every request needs
  the control lines for its prompt read, even a 30-token voice decision whose context is cached.
- **Decoding runs in batch slots beside the prompt read.** `parallel` slots (`--batch N`) decode in parallel with
  each other and with the read on the control lines; they share the GPUs.
- **A read stops only between chunks.** `STOP` is checked before each chunk of `chunk=` tokens (the server's per-request
  key) or `--prefill` (16384). A chunk runs all 48 layers and updates the recurrent (GDN) states in place, so it can't
  be abandoned half way. Under load with two slots decoding, a 4K chunk takes about 2.4 s.
- **Progress lines (PP) arrive per 16K step, not per chunk.** The engine reads a prompt in steps of `--cache-every`
  (16384), one checkpoint each, and counts progress at each step's end. "16,384 of N" therefore doesn't show the chunk
  size.
- **A STOPped read pays a swap.** When another request takes the main session, the engine copies the live sequence's
  K/V to host RAM (a fresh, unpinned buffer) and copies it back on resume. The first voice request after an agent leg
  pays about 0.5 s of this, and the agent pays a similar cost when it resumes.
- **The batch slots share one paged K/V pool.** `STRATA_SLOT_POOL_CELLS` (default `--max-context`, 262144) is the
  cells all slots together can hold. Pages are lent from the expert cache only as used. Production sets 524288, so
  two ~165K conversations fit side by side; with 262144 one was refused ("no room for a conversation of 165506
  tokens") and waited for the other slot to finish.

## 2. Tiers

The request body's `priority` sets the tier (the vLLM patch's boundaries):

| Tier | priority | Who | Gets |
|---|---|---|---|
| T0 | ≤ -20 | Nova decisions (word checks, end of turn, silence, after_reply) | first on the control lines, the reserved slot, STOP rights |
| T1 | -19 … -1 | Nova spoken replies, continuations, cache warms (-10) | the same as T0, ordered after it |
| T2 | 0 … 9 | Hermes agents (default 0) | the remaining slots; can be STOPped, held, sliced |
| T3 | ≥ 10 | background | as T2, ordered after it |

`priority_sched: true` turns the tiers on. `prio_solo_max_tier: -1` means no tier takes the solo path beside busy
slots: voice goes through its slot like everyone else, since pausing the slots for it halved Hermes decode, 116 →
43 tok/s.

## 3. Voice activity: one timestamp drives everything

`voice_seen_at` is the moment voice last counted as active. Three windows run from it:

- **The reserve window** (`prio_reserve_while_voice_s` 600): the reserved slots stay out of the agents' reach.
- **The voice-chunk window**, the same 600 s: agent reads start in 4K chunks.
- **The hold** (`prio_voice_hold_s` 15): long agent reads wait.

It moves forward when:

- **A T0/T1 request arrives.** `voice_seen_at` becomes its arrival time.
- **A T0/T1 reply finishes.** It becomes `now + output_tokens × prio_voice_speech_s`, at 0.3 s per token, capped at
  60 s. That's when the reply has been spoken, so the hold doesn't lapse during playback. Nova measures 0.28 s per
  token at p50 over 800 replies.

It never moves backwards.

## 4. Slots

- **Three slots, one reserved.** `parallel` 3 and `prio_reserved_slots` 1: within the reserve window agents (T2/T3)
  hold at most 2 slots, and the third stays free for voice. Outside the window agents may use all 3.
- **Slot choice follows the conversation.** `pick_slot` prefers the free slot whose held tokens are the longest prefix
  of the prompt, so the engine reads only the rest. Otherwise it takes an empty slot, otherwise the one used longest
  ago.
- **A freed slot goes to the best waiter.** That's the best tier, then the earliest arrival (`_slot_turn`). A waiter
  takes the control lines only once it's its turn and a slot is free.
- **Slot memory is borrowed.** Each slot's own buffers (218 MiB on GPU 0, 282 MiB on GPU 1) and its K/V pages are lent
  from the expert cache as used. Everything goes back when no slot runs and a solo request comes.

## 5. The control lines: who reads next

- **Order.** Requests ready for the control lines are ordered by (tier, arrival); see `_line_next`. Short-first
  admission (`STRATA_ADMIT_SHORT_FIRST`) is off.
- **A release wakes everyone.** It notifies all waiters, so a hand-over doesn't wait for a poll.
- **Order after giving way.** A read that gave way to voice keeps its arrival order. One that gave way to its own
  slice goes to the back (§6).

## 6. Give-ways: when a running agent read is STOPped

A T2/T3 read on the control lines is sent `STOP`, and takes it at its next chunk boundary, when its own PP lines
show at least `VOICE_STOP_MIN` (4096) tokens left and one of these holds:

1. **Voice waits.** A T0/T1 request waits for the control lines (`_voice_waiting`). This is the voice STOP
   (`prio_voice_stop` on).
2. **The hold applies.** The voice hold is active for this read (§7).
3. **Its slice is over.** It has had the control lines for `prio_agent_slice_s` (6 s) and another request is ready for
   them. It then goes to the back of the line order.

The rules count only what the read's own PP lines report: a prompt the cache serves sends none, so a 178K prompt
with 5 tokens left is never STOPped during its restore. After the STOP, the read's part is in the prompt cache. The
read is sent again later and continues from there, and every later leg goes in 4K chunks (`chunk=4096`).

## 7. The hold

- **When it applies.** A STOPped agent read with at least 4096 tokens left waits without the control lines while
  voice has been active within the last `prio_voice_hold_s` (15 s). So voice decisions find the control lines free.
- **The cap.** It's never held longer than `prio_voice_hold_max_s` (300 s) after it arrived.
- **Past the cap** it reads again. While voice continues, each decision STOPs it at the next 4K boundary, so each one
  waits about 2.4 s and the read gains about 4K per decision. This is accepted as the worst case after a long stretch
  of talk.
- **No hold for a near-finished read.** Whether a STOP leaves enough to hold for is worked out again when the leg
  ends. The STOP may come during the read's last chunk, which the engine then finishes. The leftover is the last PP
  figure less the read's mean rate times the time since. Under 4096 means no hold (e8d2ea00: a 121K read with 9
  tokens left used to sit out the whole 300 s cap).

## 8. Voice chunks

A T2/T3 read that starts within the voice-chunk window (600 s, §3) is sent `chunk=4096`, and so is every leg after a
give-way. So a voice request arriving mid-read waits for at most one 4K chunk, not 16K. Outside the window, agent
reads use the configured 16K, the fastest. Measured: 4K chunks cost no visible prompt rate under load (2000 tok/s p50
during voice, 1756 alone). A 2K chunk was tried and reverted: each leg still cost 1.4–2 s, and agent prefill fell
from 2478 to 1439 tok/s.

## 9. What a voice decision costs, by situation

| Situation | Typical decision |
|---|---|
| Control lines free (hold in effect, or nothing reading) | 0.12–0.3 s |
| A fresh agent read in progress (one 4K chunk + STOP) | ~0.7 s light load, ~2.4 s with two slots decoding |
| An agent read past the 300 s cap, during continuous talk | ~2.4 s each, until the read is done |
| First request of a session (a read began in 16K chunks before voice) | up to one 16K chunk, ~5–10 s |
| A restore-heavy agent leg (most of its context from a sibling's cache) | seen 4.5–5.9 s: the STOP landed late (open) |

## 10. Measurements (voice-lab-74, replay of a real Nova session under the user's Hermes load)

| Setup | Voice decision p50 / p90 | Over 1 s | After 89 s of quiet | Hermes |
|---|---|---|---|---|
| (a) 2 slots, 1 reserved for 60 s | 231 / 410 ms | – | 7.1 s | 4.3 req/min during voice |
| (b) 2 slots, none reserved | 250 / 2820 ms | 69 of 209 | – | 8.7 req/min; GPUs idle while reads were held |
| (c) 3 slots, 1 reserved for 600 s | 204 / 420 ms | 13 of 209 | 1.2 s | short-read wait p50 1.6 s (was 23 s) |
| (c) + speech 0.3 | 248 / 972 ms | 20 of 209 | – | load rose mid-run; after_reply 0.2–3.4 s (was 3.9–4.7) |
| (c) + speech, e8d2ea00 | 224 / 759 ms | 17 of 209 | – | short p50 2.1 s |

The user's load varies from run to run, so the Hermes request counts don't compare well across runs.

## 11. Config (production)

```json
"parallel": 3,
"priority_sched": true,
"prio_reserved_slots": 1,
"prio_reserve_while_voice_s": 600,
"prio_solo_max_tier": -1,
"prio_voice_stop": true,
"prio_voice_pause": false,
"prio_voice_hold_s": 15,
"prio_voice_hold_max_s": 300,
"prio_agent_slice_s": 6,
"prio_voice_speech_s": 0.3,
"env": {"STRATA_SLOT_POOL_CELLS": "524288", "STRATA_RES_CENSUS": "1"}
```

`prio_voice_chunk` is the default, 4096. `prio_voice_hold_max_s` is the default, 300.

The engine flags that matter here are `--main-gpu 0 --second-gpu 1 --decode-split 23`, so GPU 1 runs the heavier
stage, layers 23–47 plus the head. Others are `--prefill 16384`, `--vram-reserve-mib 200` and
`--second-gpu-reserve-mib 450`. With `--batch 3` the engine adds its own per-slot reserve on top: 230 MiB a slot on
the main GPU, 130 on the second.

## 12. Reading the server log

- `WAIT` in a FINISHED line is time waiting for the control lines or a slot. `PP` is the prompt read once it had
  them.
- `reading the prompt: X of Y` comes only from the request that holds the control lines. The others print `waiting to
  read the prompt (...)`: behind another read, a voice hold or a slot (d631f5ca; before that, every waiting request
  echoed the engine's figures).
- `parallel: BGEN: no room for a conversation of N tokens` means the slots' shared K/V pool is full (§1).
- On startup the server prints its tiers, the reserve window, the hold (with the speech rate) and the slices:
  `[strata] priority tiers on ...`, `[strata] voice hold: ...`, `[strata] agent slices: ...`.

## 13. Open items

1. **A cheaper STOP.** About 2.4 s per word check while a long read runs is the chunk time plus a ~0.5 s swap.
   Copying the stash into a reusable pinned buffer should cut the swap. Stopping inside a chunk would need the
   recurrent states rolled back to the chunk start, which is engine work.
2. **Restore-heavy legs.** At 17:09 a 135K read leg read ~90K tokens in 11 s and the STOP landed only at the leg's end.
   That needs engine tracing to explain.
3. **The cold first decision of a session.** A read that began in 16K chunks before any voice can hold the first
   decision for one 16K chunk.
4. **The bigger K/V pool is new** (18:04). Watch the hit rate while two big conversations hold ~4 GiB of pages.
