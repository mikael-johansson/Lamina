# Overnight optimization, 2026-10-25/26 (IQ3_S on 2x RTX 3090)

Criteria: C=1 first, then C=2, C=3/4, C=5+ last. Power limit 300 W.

## Result in one paragraph

Two config changes, both flags the original fork engine already has, raise IQ3_S decode by +4% (short
prompts) to +48% (150K context) and prefill by +1-2%. The big one is turning off the second GPU's expert
prefetch (`--second-gpu-prefetch-mb 0`); the other moves the prompt split from layer 26 to 28. The same
prefetch change gives Q4 +21-48% decode. The production engine binary is the original one; my experimental
engine code is kept as a patch, not built in. Batching (C>=2) and TP=2 turned out to be multi-day engine ports;
designs with measured numbers are below.

## Results (through the server, same session, 300 W)

IQ3_S decode (tok/s), real requests, default prefetch -> `--second-gpu-prefetch-mb 0`:

| request | before | after | change |
| --- | --- | --- | --- |
| code (short) | 232 | 242 | +4% |
| thinking | 193 | 200 | +4% |
| Hermes 22K | 173 | 186 | +8% |
| Claude Code 40K | 160 | 186 | +16% |
| Claude Code 90K | 161 | 185 | +15% |
| 90K follow-up turn | 178 | 208 | +17% |
| Claude Code 147K | 111 | 165 | +48% |

Prompt split 26 -> 28: prefill +1-2.3% on every long request (e.g. 90K: 4,178 -> 4,274 tok/s).

Concurrency test (`bench/load.py`, the fork queues requests), before -> after:

| | C=1 | C=2 | C=4 |
| --- | --- | --- | --- |
| short prompts, total tok/s | 198-200 -> 223-226 | 191-195 -> 217-221 | 187-191 -> 213 |
| 49K prompt, time to first token (C=1) / whole round (C=4) | 13.0 -> 12.6 s | | 50.9 -> 47.4 s |

Q4 with prefetch off: code 99 -> 133, Hermes 22K 86 -> 106, 40K 78 -> 94, 90K 73 -> 108 tok/s (+21-48%).

Tried and not kept (repeated A/B, within noise): `--spec 5`, `--spec-min-p 0.4`, fused pool phases
(`STRATA_POOL_FUSED=1`, +1.6%), pool workers 5/7, adaptive decay 0.5/0.9, 192 swaps, static second-GPU tier,
VRAM reserve 300 MiB, `--pcie-frac` 0.15/0.3 (kernel and DMA), second-GPU prefetch 4 MB (between 0 and 12).

## What limits decode now (IQ3_S, per verify window of ~4 tokens, short context)

Measured with the engine's `--stats --window-profile` (direct mode, code prompt):

| main GPU stage | ms / window |
| --- | --- |
| GDN/QSA mixer | 3.9 |
| hyper-connection reads (2x, one fused kernel each) | 2.2 |
| router + doorbell | 1.1 |
| VRAM experts | 3.2 |
| wait for the CPU's experts (~1.6 missed experts/layer) | 4.0 |
| wait for the second GPU | 0.3 (was 3.9 with prefetch on) |
| other | ~1 |
| **whole window** | **~15.7** (was ~17.3) |

- The CPU's missed experts are now the longest path. The pool reads ~24 GB/s, RAM-bound on 3 DDR4 sticks
  (part of the address space is likely single-channel with 3 sticks: the 4th stick should help).
  Not levers (all within noise): pool worker count, fused pool phases, adaptive-tier decay/swaps,
  VRAM reserve 300 MiB, `--pcie-frac` 0.15/0.3 in kernel or DMA mode.
- "99% GPU busy" on the main card was misleading: its verify graph spins on host flags while it waits for the
  CPU's and the second GPU's rows, and that counts as busy.
- Moving hot experts to the second GPU (an opt-in `STRATA_HOT_SPLIT` I added) made decode 8-17% slower: the
  second GPU's per-layer path has fixed costs (handoff through host memory, 4-5 small kernels).

### The prefetch finding

With prefetch on (default 12 MB per layer), the second GPU copied ~74 experts (~150 MB) per window over its x8
link and used only ~39 of them. That traffic doubled its own per-layer time (~180 -> ~90 us) and made the
main GPU wait 3.9 ms per window for it. Off, the experts it would have prefetched are computed on the CPU,
which is faster than streaming them.

### The second GPU's path (diagnostics, in the patch)

`STRATA_GPU2_PROF=1`: per layer share, host submit ~15 us, GPU graph ~85-90 us (plan/activation copies from
mapped memory ~20-28, experts ~11-33, rows out ~7-23) plus ~100 us of graph launch latency.
`STRATA_GPU2_ARM=1` hides the launch (the next layer's graph waits behind a kernel spinning on a mapped word):
it works (~90% of shares take the armed graph) but is neutral end to end once prefetch is off, because the
second GPU is no longer on the critical path. Kept in the patch only.

## Prefill (where it is limited)

64K-token prompt, direct mode, `--prefill-profile`: ~4,200-4,500 tok/s. With `--prompt-split` the second GPU
runs the last layers whole and also computes the main GPU's missing experts; it streams ~49K expert copies
(~100 GB, ~6.5 GB/s average over its x8 link, mostly ahead of use). Both GPUs are compute-bound
(GDN projections, attention, expert GEMMs ~1-1.3 s per section over the prompt); the second stage was the
slower one (12.4 s vs 11.0 s at split 26), which is why moving the split to 28 helps (+1-2% through the server;
direct mode kept improving to 30, the server did not). Not levers: a bigger PLE row cache / more PLE reads in
flight (the "blocked 2.8 s" counter is not on prefill's critical path), `--no-prompt-offload` (turns the split
off: -27%). Q4 keeps split 26 (28 was mixed, 30 collapsed short prompts).

Further prefill speed would need kernel work (the GDN projection and expert GEMM kernels) or a third GPU.

## Batching (C>=2): design and expected gain

Measured cost model (fork, `--spec` 2/4/6/8 at min-p 0): a window costs ~8.3 ms fixed + ~1.75 ms per verified
row (short context), ~8.4 + ~2.4 ms at 40K. A 2-slot batch with 3 drafts each is 8 rows: measured 22.1 ms
(short) / 25.7 ms (40K) for ~6.7 / ~5.8 tokens. Expected total throughput vs today's queued C=2:
~1.35x at short context, ~1.2x at 40K; C=4 ~1.65x / ~1.35x. MoE batching scales worse than the dense 27B
because rows rarely share experts (~1.4 rows per distinct expert per window; per-request distinct experts
fall only 15% at C=4).

Upstream's batching does not get this on our box (measured: C=2 136-162 tok/s total, C=4 151-170, below the
fork's single stream ~200): it verifies one MTP proposal per slot (hard-coded), and its layer-split window is
~45% slower per row than the fork's.

What a port into the fork needs (the fork's `custom` branch never had batch slots; upstream's arrived via
merges where the fork kept its own engine):
1. Per-slot `SessionState` (GDN state, QSA K/V, PLE history), carved like the main session; slot state must not
   come out of the expert cache (the miss cliff: 2,400 fewer resident experts = ~10x the misses).
2. A verify window over rows of several sessions: per-row state pointers in the QSA/GDN kernels, the GDN
   recurrence restarting per slot group, commit of an accepted prefix per slot (upstream's
   `run_slot_rows` / `commit_slot_prefixes` as the model), k drafts per slot (not upstream's 1).
3. Per-slot MTP drafters (shared weights, own K/V ring), drafts of all slots batched per depth.
4. The serve loop: admission, per-slot prompt reading (prefill interleaved with decode, or long prompts
   stall the other streams: upstream's slots fell to ~13 tok/s during a 49K prompt), sampling per slot.
5. Tests: a slot's greedy tokens equal its solo tokens in deterministic mode (`--deterministic-adapt`).
Estimate: several days of engine work.

## TP=2: design and expected gain

P2P between the two cards works (driver reports peer access OK). Microbenchmark
(scratchpad `p2p.cu`): pipelined peer copies 10-13 GB/s; a kernel pushing 10-40 KB into the other card's memory
and raising a flag ~11-21 us one way (host-launched; inside a captured graph it would be lower). TP needs ~2
exchanges per layer (~96 per window) of ~40 KB at 4 rows: ~1-2 ms per window, against ~11 ms of main-GPU work
that would be split. Rough expected gain at C=1: +25-35% (estimate). Experts would be split by rows (each card
half of every cached expert: same capacity as today's two tiers, twice the miss bandwidth over two links).
Cost: the GDN/QSA state and kernels split by heads, the hit path, the CPU pool's rows, MTP and the prompt path
all change. Weeks rather than days in this engine.

## Files

- configs: `fork-iq3_s-tuned.json`, `fork-iq3_s-abliterated.json` (prefetch 0, split 28),
  `fork-ud-q4_k_xl-tuned.json` (prefetch 0). Backups: `*.bak-preprefetch`.
- `patches/fork-gpu2-experiments.patch`: `STRATA_HOT_SPLIT`, `STRATA_GPU2_PROF`, `STRATA_GPU2_ARM` (all opt-in;
  `git apply` in `eddoursul/`, rebuild with `cmake --build build --target strata`). The experimental binary is
  `eddoursul/build/strata.hotsplit-exp`; `build/strata` is the original.
- bench tools: `bench/trace_run.py` (routing traces), `bench/route_analysis.py`, `bench/load.py` (C clients),
  `bench/concurrency.sh`; results in `bench/runs/`.
