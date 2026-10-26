# Strata on ada: status report (2026-10-26)

Qwen3.8-Flash-Next (512-expert MoE, 10 active) served by the eddoursul fork of Strata on one AM4 box. This branch
(`ada`) is the fork's `custom` branch plus our work:

- commit 1: opt-in engine experiments (`STRATA_HOT_SPLIT`, `STRATA_GPU2_PROF`, `STRATA_GPU2_ARM`); with none of
  the variables set the engine behaves exactly like `custom`.
- commit 2: this `ada/` directory: configs, start script, benchmark tools, results and reports.

Production runs `custom` plus the `--arena-place` fix below (`eddoursul/build/strata`; the plain `custom` build is
`build/strata.orig`).

## Machine

| part | detail |
| --- | --- |
| CPU | Ryzen 7 5700X (8 cores, AVX2, no AVX-512) |
| RAM | 96 GB DDR4, 3 sticks (a 4th is on the way: 128 GB, both channels fully populated) |
| GPUs | 2x RTX 3090 24 GB, PCIe 4.0 x8/x8, power limit 300 W. CUDA 1 = main card, CUDA 0 = second card |
| spare | a 3rd and 4th 3090, not installed (waiting for a riser; PCIe 4.0 x4 slot) |
| storage | experts and packs on `/d/nvme_512/strata-data` |

Priorities (from the user): raw API speed; C=1 most important, then C=2, C=3/4, C=5+ least. IQ3_S is the focus
model; UD-Q4_K_XL is deferred until a 3rd card is installed.

## How it runs

```
./strata.sh start            # IQ3_S, port 8091 (configs/fork-iq3_s-tuned.json)
./strata.sh start <config>   # other configs; stop/status likewise
```

The configs in `ada/configs/` are copies of the live ones in `~/git/strata/configs/` with the API key replaced
by a placeholder (the live key lives in `~/git/strata/api-key.txt`, not in git).

| config | port | model | notes |
| --- | --- | --- | --- |
| `fork-iq3_s-tuned.json` | 8091 | IQ3_S | the production config |
| `fork-ud-q4_k_xl-tuned.json` | 8092 | UD-Q4_K_XL | prompt split 26 |
| `fork-iq3_s-abliterated.json` | 8093 | IQ3_S abliterated | same flags as 8091, starts from SATA (~150 s) |
| `up-*.json` | 8094/8095 | upstream niko1221/strata | A/B only; `-mtp` variants add `--batch-mtp --batch-groups 1` |
| `up-ud-q4_k_xl-p4.json.disabled` | | | 4 slots of Q4 overran RAM and swapped the machine; do not use |

Key flags in the tuned configs and why:

- `--second-gpu-prefetch-mb 0`: the overnight win (below).
- `--prompt-split 28` (IQ3_S; Q4 keeps 26): the second card runs layers 28-47 of the prompt.
- `--second-gpu-reserve-mib 512`: more VRAM for expert slots (prefill +1-3.5%).
- `--vision` with the CPU reader (`build-vision/bin/strata-vision`, mmproj BF16, `gpu: false`, 1,024 image tokens):
  ~15 s for a new 1080p screenshot, ~1 s when the same image comes again. Keeps all VRAM for experts.
- `expert_profile_save`: the adaptive tier swaps experts between requests (`--adapt-every 1`, a few hundred swaps,
  70-170 ms) and saves what it learned to `/d/nvme_512/strata-data/profiles/iq3_s-learned.bin`; the next start
  loads that instead of the shipped profile. So the cache follows real traffic by itself.

## Current numbers (IQ3_S, through the server, 300 W)

Decode (tok/s) on recorded real requests, default prefetch -> prefetch 0:

| request | before | now |
| --- | --- | --- |
| code (short) | 232 | 242 |
| thinking | 193 | 200 |
| Hermes 22K | 173 | 186 |
| Claude Code 40K | 160 | 186 |
| Claude Code 90K | 161 | 185 |
| 90K follow-up turn | 178 | 208 |
| Claude Code 147K | 111 | 165 |

Prefill: ~4,200-4,500 tok/s on long prompts (90K: 4,274 tok/s; 49K prompt: time to first token 12.6 s).

Concurrency (`bench/load.py`, short prompts, total tok/s): C=1 223-226, C=2 217-221, C=4 213. The fork serves one
request at a time and queues the rest, so the total stays flat; C=4 on 49K prompts: 47.4 s a round.

Q4 with prefetch 0: code 133, Hermes 22K 106, 40K 94, 90K 108 tok/s (+21-48% over the default).

Live traffic (2026-10-26 morning, Hermes tool calls): 152-180 tok/s, expert cache 94-96% hit, against 97-99% on
the coding benchmarks. Agent tool calls use experts the profile (learned overnight from benchmark traffic) ranks
lower; the adaptive tier and the saved profile move toward them as traffic comes in. Turning prefetch off also
lowers the counted hit rate by ~0.5-1 point (experts the second card used to stream in now count as CPU misses)
while making decode faster.

## Where the time goes

### Generation

One verify window (~4 tokens checked at once, short context) is ~15.7 ms
(`--stats --window-profile`, direct mode):

| main-card stage | ms |
| --- | --- |
| GDN/QSA mixer | 3.9 |
| hyper-connection reads | 2.2 |
| router + doorbell | 1.1 |
| experts in VRAM | 3.2 |
| waiting for the CPU's missed experts | 4.0 |
| waiting for the second card | 0.3 (3.9 with prefetch on) |
| other | ~1 |

- The CPU's misses: ~1.6 missed experts per layer at 98-99% hit, computed by 6 pool workers at ~25 GB/s. That
  rate is the pool's own limit, not the RAM's: per layer it reads only ~3 MB in two phases with a barrier each
  (~130 us), too little work to reach the RAM's speed. Evidence: with every CPU expert moved into dual-channel
  RAM (below) the pool's rate and the window time did not change (direct mode, ABAB: 24.5-25.2 GB/s, 15.5-15.6 ms
  per window either way), and pool workers 5/7 were within noise. At 94-95% hit (agent traffic) there are ~3x as
  many misses, so this share grows.
- RAM layout (measured with `bench/membw`, root, by physical address): with 3 sticks, physical 0-32 GiB is
  single-channel (~22 GB/s on 8 cores) and 32-96 GiB dual-channel (~36 GB/s). `--arena-place` is meant to put
  the CPU's experts in the fast places, but it timed them on one core (the pool's host thread was already
  pinned), where both regions read ~25 GB/s. Fixed on this branch (it times on all cores,
  `src/core/arena_place.cpp`): it now finds ~10K slow places (~20 GiB) and moves the ~2,900 CPU experts that were
  in them (42%) out. Measured effect on decode: none in direct mode; +3% in server ABAB runs
  (`bench/results/ram-*`), likely noise. A 4th stick is therefore unlikely to speed up IQ3_S decode; it adds room.
- The main card's own work (~11 ms) is neither compute- nor clearly bandwidth-bound. Estimate from the weight
  sizes: each window reads ~2.9 GB of dense weights (projections 2.0 GB, hyper-connections 0.64 GB, its part of
  the head ~0.3 GB) plus ~1.5-2 GB of resident experts, i.e. ~400-450 GB/s, about half of the 3090's 936 GB/s.
  At 4 rows the matrix math is tiny (compute is far from the limit; 400 W gave only ~5%). The gap to full
  bandwidth is latency: 48 layers of small, dependent kernels (mixer, hyper-connections, router with its host
  doorbell), each too short to fill the card. The expert kernels come closest to bandwidth-bound.
- "99% GPU busy" in nvidia-smi is misleading: the verify graph spins on host flags while it waits for the CPU
  and the second card, and that counts as busy.
- Long context adds attention cost: ~8.4 ms fixed + 2.4 ms per row at 40K vs 8.3 + 1.75 short.

### Prefill

Compute-bound on both cards (GDN projections, attention, expert GEMMs, ~1-1.3 s per section over a 64K prompt).
With the prompt split the second card runs the last layers and the main card's missing experts; it is the slower
stage (12.4 s vs 11.0 s at split 26), hence split 28. It streams ~49K expert copies (~100 GB at ~6.5 GB/s over
its x8 link), mostly ahead of use, so PCIe is not the limit; missed experts go to the second card, not the CPU,
so the hit rate barely matters for prefill. The PLE "blocked" counter is not on the critical path. Faster
prefill needs faster kernels or a third card.

## What was tried

Kept: prefetch 0, split 28 (IQ3_S), VRAM reserve 512 MiB, CPU vision, `--arena-place` timed on all cores (correct, but no measurable speedup).

Not kept (repeated A/B, within noise or worse): `--spec 5`, `--spec-min-p 0.4`, fused pool phases
(`STRATA_POOL_FUSED=1`, +1.6%), pool workers 5/7, adaptive decay 0.5/0.9, 192 swaps, a static second tier,
reserve 300 MiB, `--pcie-frac` 0.15/0.3, prefetch 4 MB, split 30/32 through the server, hot experts on the
second card (`STRATA_HOT_SPLIT`, -8..-17%), pre-armed second-card graphs (`STRATA_GPU2_ARM`, neutral),
400 W power limit (~5%).

Upstream (niko1221) on this box loses at every concurrency: single stream ~45% slower per row, and with
`--batch-mtp` C=2 136-162 / C=4 151-170 tok/s total (one MTP proposal per slot is hard-coded).

## Open directions

| idea | expected | cost |
| --- | --- | --- |
| 4th RAM stick | 32 GB more room (Q4, more at once); no decode gain expected for IQ3_S (the pool is not RAM-bound) | hardware |
| batching in the fork (C>=2) | C=2 ~1.35x total short / ~1.2x at 40K; C=4 ~1.65x / ~1.35x | several days of engine work |
| TP=2 | C=1 +25-35% (estimate; P2P works, 11-21 us one-way pushes) | weeks |
| 3rd card for Q4 | Q4 is CPU-miss-bound on 2 cards | hardware |

Designs and the measured cost model: `docs/OVERNIGHT-2026-10-26.md`. MoE batching scales worse than the dense 27B
on vLLM because rows rarely share experts (~1.4 rows per distinct expert per window).

## Files

- `configs/`: server configs (key redacted).
- `strata.sh`: start/stop/status; `smoke.py`: first smoke test.
- `bench/bench.py`: real recorded requests through the server (`corpus.json`); `bench/load.py`: C concurrent
  clients; `bench/concurrency.sh`: fork vs upstream matrix; `bench/trace_run.py` + `bench/route_analysis.py`:
  expert-routing traces and the batching simulation; `bench/membw.c`: RAM read speed per physical GiB (root);
  `bench/ram_offline.sh`: takes a physical range out of use (memory hotplug); `bench/extract_requests.py`: builds the request index from
  the vLLM logs (only request bodies).
- `bench/results/<run>/`: `results.json` / `load.json` of every run (server logs and config copies left out).
- `docs/`: the overnight report, the running plan (`PLAN.md`), the first 2x3090 analysis.
- Experiments (commit 1): `STRATA_HOT_SPLIT=f`, `STRATA_GPU2_PROF=1`, `STRATA_GPU2_ARM=1`; build with
  `cmake --build build --target strata`.

Not in git: the API key, model files and packs, `deps/`, the upstream clone (`~/git/strata/upstream`), server
logs and recorded request bodies.
