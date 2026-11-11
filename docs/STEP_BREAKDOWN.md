# Step-time breakdown (`STRATA_STEP_BREAKDOWN=1`)

Where the serve engine's wall clock went, every 10 s, in the shape of the vLLM fork's
`step-time-breakdown` status lines (`patches/step-time-breakdown.patch` in the vLLM tree):

```
[strata] 11.1 s: RUNNING: 1 reqs (1 gen, 0 prefill), WAITING: 0 reqs
[strata]   - 54.5% Prefill @ 2662 PPt/s (1 it, 1 with decode: 215 tok). Completed: 1 req / 63535 tok
[strata]   - 45.5% Gen @ 102.9 tg/s (1 req: 195 it @ 100.0 tg/s, 2 req: 22 it @ 132.0 tg/s)
[strata]   - 0.3% Other (request setup and handover on the main thread)
[strata]   - 1.2% Stalled (3 stalls: requests present, nothing running)
[strata]   - 0.0% Idle
```

Switch: `"STRATA_STEP_BREAKDOWN": "1"` in the config's `env` (a number ≥ 2 is the interval in seconds).
The engine writes `strata steps: ...` lines to its log (the config's `log`). server.py also echoes them
to the console with `[strata]`, and adds `WAITING` to the status line (requests waiting for the control
lines or for a batch slot; only the server knows that queue). Windows with no work in them print nothing.

## Categories

The engine has no single scheduler loop. The main thread (prompt reads, the solo decode) runs beside the
batch slots' two stage threads, so the time is not summed per thread. One ledger
(`include/strata/core/step_breakdown.hpp`, behind one mutex) holds what is running now. At each event, the
time since the previous event goes to exactly one category:

| category | when |
|---|---|
| Prefill | a prompt read on the main thread (cache restore to the last fed token). Slot windows that run beside it make its chunks "with decode": their time is the read's (vLLM's rule), and their tokens are listed there and kept out of the Gen rates |
| Gen[n] | decode windows running. n = requests decoding: active, unpaused batch slots, plus the solo request |
| Other | the main thread holds a request but neither reads nor decodes: setup, K/V growth, a slot's handover copies, the TIERS/DONE bookkeeping. vLLM folds this into its steps; here it is shown from 0.1% |
| Stalled | a batch slot holds a request and nothing runs (the slots paused for the main thread, which is idle). Shown from 0.1% |
| Idle | the rest |

- **PPt/s** is fresh prompt tokens (the main path's batched chunks and feed windows) over Prefill time.
  **Completed** counts the reads that ended, with `n - reuse` tokens each (the FINISHED line's number).
  Short prompts mostly go through the first decode window, so their PPt/s is low and means little,
  as in vLLM.
- **tg/s** is emitted tokens (after stops, MTP acceptance included) over the time spent at that concurrency.
- **it**: a prompt chunk or feed window for Prefill, a verify window for Gen.

## Window ends

A window ends at the tick, except while a read is in flight: then it ends at the read's last chunk boundary,
and the rest carries into the next window (vLLM's rule, so a chunk's time and tokens land together). If no
boundary falls inside the window (one chunk longer than the tick), the window stays open until the next tick.
With `--prompt-split`, the second GPU's stage runs a chunk behind the first. So a chunk counts only once both
stages have it, and the split's last chunk counts at `psplit.finish`. Decode windows (~10-20 ms) are cut at
the tick.

## Cost

Per event: one `steady_clock::now()` and one uncontended lock (events come per chunk and per verify window).
Nothing runs per layer or per token, and nothing touches the GPU path. With the switch off, each hook is a
null-pointer test.

## Tests

- `step_breakdown_test` (CMake target, CPU only): solo request, two overlapping slots, read over a decode
  window, pauses, carry-over across the tick, one chunk spanning the window.
- `serve/test_step_breakdown_echo.py`: the console echo and WAITING.

## Not done (on purpose)

- vLLM's KV-cache % on the status line: Strata's elastic K/V and slot page pool don't map onto one number.
- `/metrics`: log only, like the vLLM patch.
- The dead BSTAT/PSTAT parser in server.py, a leftover of upstream's telemetry, is left alone.
