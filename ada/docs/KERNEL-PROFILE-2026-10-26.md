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

## What is left

Fusing the chains of small dependent kernels (HC down + up, the router) so their reads overlap: up to ~12% of a
window.  On Ampere (no programmatic dependent launch) that needs a grid-wide wait inside the graph, beside kernels
that spin on CPU flags: co-residency has to be guaranteed or it can deadlock.  It carries over to a 3-stage split.
