# GPU kernels on the 3090: profile and experiments (2026-10-26)

IQ3_S, the layer split (`configs/fork-iq3_s-split-batch.json`, `--decode-split 23`), one request, the `code`
prompt, 600 tokens.  Unprofiled: 225 tok/s, 184 windows, 14.5 ms a window.

## Tools (`ada/bench/kernels/`)

| tool | what |
| --- | --- |
| `ktrace.cpp` (+ `ktrace.sh`) | CUPTI activity tracer, `LD_PRELOAD`: every kernel's GPU time, also inside graphs; per (grid, kernel) table; `KTRACE_RAW` dumps each kernel; `<KTRACE_OUT>.go` brackets a request on a running server. No root. |
| `ktline.py`, `ktgaps.py` | from the raw dump: per device the time each kernel ran alone; the gaps with no kernel on either GPU and the kernels around them |
| `nsgaps.py` | from an nsys sqlite export: the GPU-idle gaps, the CUDA API calls and CPU samples inside them |
| `glaunch.cpp` | `cudaGraphLaunch`'s host time without a profiler (`LD_PRELOAD`) |
| `gk.cu` (+ `gk.sh`) | the HC read, the verify router and gdn alpha/beta alone at the model's shapes over 48 layers of distinct weights, in a graph; matches the engine's times |
| `floor.cu` | an empty kernel's cost in a graph, and a plain 3.3 MB read |
| `ncu_gk.sh` | Nsight Compute on `gk` (root while `NVreg_RestrictProfilingToAdminUsers=1`) |
| `ktrun.py` | one request against a server started with `ktrace` preloaded |

**Do not run ncu on the engine's decode**: ncu serializes kernels, and the verify graph has kernels that spin on
flags the host or the other GPU raise; it deadlocks at the first window (seen: GPU 1 at 100%, no report).
`ncu_decode.sh` is that attempt, kept as a warning.  Profilers also inflate graph launches: under CUPTI/nsys
node tracing a ~700-node graph's launch took 0.6-0.9 ms, unprofiled 3-10 us.

## Where a window goes (the engine's `--window-profile`, unprofiled)

GPU 6.1 ms (first stage) + 6.8 ms (second stage and the head) + MTP drafting 0.9 ms + host ~0.6 ms.

| per layer and stage | us | share of the window |
| --- | --- | --- |
| mixer (GDN / QSA projections, conv, recurrence) | ~80 | ~28% |
| VRAM experts | ~82 | ~28% |
| HC reads (two: down + up, 8-bit) | ~42 | ~15% |
| router, ring, hits | ~21 | ~7% |
| wait for the CPU pool | 6-23 | ~5% |
| head (Q6_K, 521 MB at ~850 GB/s) | | 0.65 ms, ~4.5% |

The big streams are at the limit: the mixer's projections (qkv and gate side by side ~900 GB/s, ssm_out ~670),
the experts ~620-690, the head ~850.  The small kernels are not:

| kernel (T = 4) | us | bytes | floor | ncu |
| --- | --- | --- | --- | --- |
| HC down (`gr_down_q8_split`) | 11.4 | 3.3 MB | ~5 | DRAM 34%, 1 block/SM (153 regs), stalls: barrier 17%, issue 21%, icache 11%, memory 8% |
| HC up (`gr_up_q8_multi`) | 10.3 | 3.3 MB | ~5 | DRAM 33%, IPC 1.8 at 33% occupancy, stalls: issue/wait ~40%, memory 7% |
| verify router | 12.5 alone, ~19 in the engine | 2.6 MB | ~4 | stalls: barrier 26%, icache 12%, membar 10% |
| gdn alpha/beta | 10.6 alone, ~20 in the engine | 0.5 MB | ~1.5 | 12 blocks, memory latency; beside qkv (not on the critical path) |

Floor (`floor.cu`): an empty dependent kernel in a graph costs 0.7-1.9 us; a plain 3.3 MB read 4.9-5.7 us.
The HC reads and the router above their floors are ~35 us a layer, ~1.7 ms a window (~12%).

## Tried (none kept)

| experiment | result |
| --- | --- |
| HC down prefetches the up kernel's codes into L2 (`prefetch.global.L2::evict_last`) | no change (22.2 vs 21.4 us) |
| HC up: every global read hoisted above its barrier | slower: 10.3 -> 11.4 us |
| HC up: lo in 8 bits and `dp4a` (q8_1-style) | kernel 10.3 -> 9.5 us alone; in the engine no change (222 vs 224.5 tok/s), and the output differs from token 96 |
| HC down without its last-block reduction (timing only) | 11.0 -> 8.9 us: the reduction is ~2.3 us |

INT8: decode already runs its matvecs on q8_1 activations and prefill its GEMMs on int8 tensor cores (MMQ);
decode is latency-bound, not arithmetic-bound.

## Prefill and gpu-unlock

A 20K-token prompt by kernel kind: int8 MMQ 48%, HC 13% (about half BF16 mma), cuBLAS/cutlass BF16/FP16 GEMM
with FP32 accumulate 11%, QSA attention (FP16 mma, FP32 accumulate) 10%, QSA indexing 6%, GDN 5%.  About 28% of
the kernel time is on the FP16/BF16-into-FP32 tensor-core path that gpu-unlock doubles: expected +8-15% on long
prompts, nothing on decode.

## Kernel fusion: the HC read in one kernel (tried, not kept)

`bench/kernels/hc-fused-experiment.patch` (`STRATA_HC_FUSED=1`, windows of 1-4 tokens): one kernel of 80 blocks,
one an SM.  Each block starts `cp.async` copies of its up rows' codes and scales (43.5 KB) and of the epilogue's
inputs into shared memory, does the down slice as `gr_down_q8_split`, then crosses a grid-wide barrier (atomic
counter and generation; traps after ~1 s instead of hanging).  The (token, row) sums are shared out between the
blocks, then a second barrier, then each block does two old up blocks' work from shared memory.  Bitwise equal to
the two kernels at T = 1..4 (`gk T 1 <dump>` with and without the variable).  Deadlock-safe in the engine's graph:
80 blocks of 82 SMs at one an SM, and the only kernels that spin forever are a few threads.

It is slower: T=4 25.4 us against 20.6 (T=1 19.6 against 16.2).  By phase (T=4):

| phase | ends at | |
| --- | --- | --- |
| down slice + every prefetch | 12.95 us | the two-kernel down: 11.4 (8.9 without its reduction) |
| barrier 1 | 14.4 | +1.4 |
| shared-out sums + barrier 2 | 17.7 | +3.3 (each block adding all 26K partials itself: +7.1) |
| lo back | 18.9 | +1.2 |
| up from shared memory | 25.4 | +6.6: 8 warps an SM instead of the up kernel's 16, issue-bound |

The overlap of the reads gains ~4 us; the two barriers and the cross-block sums cost ~5, and the up arithmetic at
half the occupancy ~3 more.  Two blocks an SM (160 blocks, finer slices) would restore the occupancy but change the
summation order (no longer bitwise) and, by the phases above, reach ~19 us at best against 20.6: not worth it.

The router is ~12.5 us alone and ~19 in the engine: the rest is publishing each layer's activations to mapped host
memory for the CPU pool and the system-scope fence before the ring.  The fork already sends one form per layer
(floats or Q8_K, the q8_1 rows only with a second-GPU tier).

## What is left

On this card the small kernels' cost is mostly their own arithmetic and reductions at low occupancy, not the
gaps between them: fusing the HC read did not pay (above).  Without programmatic dependent launch (Hopper on) there
is no cheap way to start a kernel's reads before its predecessor ends.  The cheap remaining levers are settings:
VRAM overclock (the big streams are at the bandwidth limit), gpu-unlock for prefill, CPU PBO, `--spec 5`.

## The CPU expert pool: limited by the package power (later the same day)

`kbench` (`ada/bench/kbench.cpp`), gate/up with the weights in L3 (no RAM in the way):

| | 1 thread | 7 threads |
| --- | --- | --- |
| core clock (median) | 4650 MHz | 4025 MHz (-13.5%) |
| GB/s of weights a thread | ~4.4 | ~4.0 (-10%) |

7 threads reach ~27-28 GB/s; the engine's pool phases stream 27.3 GB/s; the fast (dual-channel) places in which
`--arena-place` keeps every CPU-computed expert time ~35 GB/s on 8 cores.  So gate/up is compute-bound, below the RAM
it reads (kbench's "RAM" mode allocates anywhere, single-channel memory included, so it is not that ceiling).  The
compute is clock-bound: `ada/bench/cpu/cpu_power.sh` (root, RAPL) with `avxburn` (the kernels' maddubs / madd / add
chains) on every core: **77 W at 3975-4000 MHz**, the 5700X's 76 W PPT; one core at 4650 MHz takes 47 W; idle
17 W; Tctl at most 65 C under load (throttling at 90).

Kernel experiments (reverted): gate and up rows decoded in one loop (shared activation loads, two chains): +7% IQ2_S
on one thread, +1.7% on 7, -2.6% IQ3_XXS, 0 IQ3_S.  On Zen 3 the gathered decodes (Intel-only in the engine) are +22%
for IQ3_S rows of 2 tokens and slower for IQ2_S / IQ3_XXS, but the pool's experts get ~1.1 tokens each.  Power-bound,
rearranging the same instructions buys little.

The lever is PBO (step by step: `PBO-GUIDE-2026-10-26.md`): PPT ~120 W, TDC 80 A, EDC 120 A, Curve Optimizer all-core
-10.  All-core 4.0 -> ~4.4-4.5 GHz would be ~+10-12% pool throughput: estimated ~+5-8% at C=2-4 (the pool busy
60-72%), ~+0.5-1% at C=1 (the CPU wait ~5% of a window).  Idle power unchanged (limits only).  Not yet applied.

`ada/bench/cpu/segv_bt.c`: an `LD_PRELOAD` SIGSEGV backtrace (it found the decode split's checkpoint-restore crash
without gdb).

## A 5900X / 5950X instead (estimate, not measured)

The pool's compute today: 7 threads x ~4 GB/s = ~27 GB/s (power-bound, 76 W).  Its RAM ceiling: the fast places,
~35 GB/s timed on 8 cores, maybe ~38-42 with more cores (DDR4-3200 dual channel: 51.2 in theory).  A 5900X (11 pool
threads at ~4.2 GB/s, 142 W) is ~46 GB/s of compute, a 5950X ~60: both RAM-bound at ~35-40, so +30-45% pool
throughput for either; the 5950X buys nothing over the 5900X here.  The 4th RAM stick does not raise that ceiling.

| | 5700X + PBO | 5900X | 5950X |
| --- | --- | --- | --- |
| CPU pool | +10-12% | +30-45% (RAM-bound) | as the 5900X |
| C=1 | ~+0.5-1% | ~+2-3% | ~+2-3% |
| C=2 total | ~+5% | ~+12-20% | ~+12-20% |
| C=3/4 total | ~+5-8% | ~+15-25% | ~+15-25% |
| prefill | ~0 | ~0 | ~0 |

C=1 gains little: on the split the CPU's work mostly runs beside the GPUs' and only ~0.66 ms of a 14.5 ms window
waits for it (this replaces the earlier C=1 +8..+20%, from before the split).  At C>=2 the pool is busy 60-72% and
the layers wait 55-93 us for it.  The two-chiplet parts add cross-CCD latency to the pool's per-layer syncs (a few
us a layer, inside the ranges above).  With a 3rd card fewer experts are on the CPU and a CPU upgrade is worth
less still.  Order: PBO first (free; it also shows how the pool scales with clock), a used 5900X if C>=2 matters
before the 3rd card, not a 5950X.

Status: PBO and the CPU swap are on hold (no physical access for now); the guide is `PBO-GUIDE-2026-10-26.md`.

## SMT under the power limit

`kbench` with 14 threads, the pool's 7 cores (1-7) and their siblings (9-15), against 7 threads on the same cores,
weights in L3: IQ2_S gate/up 28.3 -> 30.4 GB/s (+7%), IQ3_XXS 26.3 -> 27.8 (+6%), IQ3_S 26.6 -> 28.3 (+6%), IQ4_NL down
68.6 -> 72.0 (+5%), Q2_0 down 40.8 -> 43.9 (+7%).  The second hardware thread fills the core's stalls for the same
watts.  The engine pins one worker a physical core (`pool.cpp`); adding the siblings is a small switch, then a
server A/B at C=1/C=2 (the per-layer sync of 13 workers instead of 6 may eat part of it).  Not tried yet.

Are the CPU kernels optimal?  Not provably.  They are the fork author's tuned multi-token AVX2 kernels (codebook
decoded once per 32 values for every token, ggml's sign tricks, gathered variants per CPU); two rearrangements here
gained nothing while power-bound.  What is left for them: SMT (above), PBO (more watts), the per-format gather choice
on Zen 3 (<1%), and a cheaper lossless layout of the CPU experts (fewer uops a weight, more bytes; speculative).
