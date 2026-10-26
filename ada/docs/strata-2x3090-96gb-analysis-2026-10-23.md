# Strata on 2x3090 with 96 GB DDR4: engine, quant and expert-miss analysis (2026-10-23)

Follow-up to the Strata / Flash-Next report of 2026-10-19
(`hardware/strata-flash-next-report-2026-10-19.md`), now that the RAM kit is in: 3 of 4 sticks
(one defective, replacement requested), 96 GB at DDR4-3200, uneven channels (64 + 32 GB).
**Nothing in this note is a Strata measurement on this box yet**; the RAM bandwidth is measured
(hardware repo, `ram-install-checklist-2026-10-22.md`), everything else is from the projects'
docs and published numbers.

## Summary

- **Engine: try the eddoursul fork first, upstream as the A/B.** With 96 GB every IQ3_S expert
  fits in RAM, which removes the fork's one blocker (it has no low-RAM modes). On a 2-GPU box it
  measured 1.35-1.71x (IQ3_S) and 3.1-4.1x (UD-Q4_K_XL) upstream's layer split on prompts of
  16K-128K, and 199 tok/s IQ3_S decode. Upstream stays the engine for 3 cards (the fork has no
  layer split) and for Q4 until the RAM is 128 GB (it can keep a RAM budget of experts).
- **Upstream's drift is real, but at the app level.** Of 523 commits 2026-10-19..23, most touch
  the desktop host app, the terminal UI and the launcher. The engine got 4 releases in 4 days
  (0.1.45.3 -> 0.1.48), mostly opt-in work (DFlash2 drafter, cost-aware MTP window); an sm_86
  start-up tweak was made opt-in after measuring 0.995x. Run the engine + `serve/server.py`
  only, pinned to one engine version.
- **Quants: IQ3_S for speed on 2 cards, UD-Q4_K_XL for the quality question.** Q4 on 2 cards
  will be slow in decode (CPU/RAM-bound misses) but its quality is what decides whether a third
  3090 is worth buying.
- **Streaming misses over PCIe helps, but does not remove the RAM bottleneck in decode**: a
  PCIe 4.0 x8 link (~13 GB/s) is slower than even this RAM (~29 GB/s). Strata splits the misses
  between the two (`--pcie-frac`), worth ~30% on Q4's miss time. The real lever is fewer misses:
  more VRAM.
- **Files on the NVMe** (the vLLM KV disk tier was emptied): `/d/nvme_512/strata-data/models/`
  IQ3_S and UD-Q4_K_XL, SHA-256-checked. The n-gram table needs random reads that a SATA SSD
  cannot sustain during prefill.

## 1. Fork or upstream

| | eddoursul fork (`custom`) | upstream (Niko1221 `main`) |
| --- | --- | --- |
| base | merged upstream at e88ee39d (2026-10-19), +650 own commits, outside PRs | 0.2.3.1 / engine 0.1.48 |
| focus | engine and API speed only | engine + desktop app, terminal, launcher, many GPU vendors |
| 2 GPUs | 2nd GPU as an extra expert tier (`--second-gpu`) | layer split (`--gpus 0,1`, `--layer-split`) |
| 3+ GPUs | no | yes |
| experts | every expert in RAM (IQ3_S ~53 GB, Q4 ~81 GB taken) | also low-RAM, mapped and RAM-budget modes (`--resident-budget-gib`) |
| concurrency | one request (no `--batch`) | up to 8 slots |
| KV streaming | no (KV in VRAM) | yes |
| Linux | not documented (Windows build instructions); untested here | ready-made CUDA 13 engine |
| measured | 9700X, 160 GB DDR5-5600, Windows, 3090 (x4) + 5070 Ti (x16) | various; 2x3090 130-137 tok/s, 1,800-2,100 prefill at 16K |

Fork measurements on its own PC (tok/s, its `docs/COMPARISON.md`, 2026-10-18/19):

| | IQ3_S, both GPUs | IQ3_S, 3090 only | UD-Q4_K_XL, both | UD-Q4_K_XL, 3090 only |
| --- | --- | --- | --- | --- |
| prompt 16K | 3,119 | 2,160 | 3,381 | 1,491 |
| prompt 128K | 5,781 | 2,246 | 5,735 | 1,467 |
| decode, code | 198.7-208.7 | 132.6 | 130.2-133.3 | 87.5 |
| decode, thinking | 168.1-171.6 | 137.2 | 121.3-123.8 | 83.3 |

Caveats: different CPU (AVX-512), RAM (~2.5x the bandwidth) and second GPU; some settings are
calibrated for that PC (the two-stage prompt split layer "needs measuring on another pair").

**Plan**: fork with IQ3_S first; upstream engine (pinned) + `serve/server.py` with the same model
files as the A/B; upstream for Q4 until 128 GB.

## 2. Quants on 96 GB

| | IQ3_S (ISTA GSQ-RCO) | UD-Q4_K_XL (Unsloth) |
| --- | --- | --- |
| download | 54.8 GB shard 1 (+ the shared 28.8 GB n-gram shard) | 111.3 GB (4 shards, own n-gram table) |
| experts | 50 GB | 77 GB (71.7 GiB) |
| top-1 = Q8_0 / mean KL / KL p99 | 87.7% / 0.144 / 1.98 | 93.1% / 0.056 / 0.83 |
| 96 GB, fork | all in RAM, ~53 GB taken: fits | ~81 GB taken: ~10 GB left for the OS, tight |
| 96 GB, upstream | all in RAM | setup's rule picks all-in-RAM (71.7 + 16 GiB <= 92 GiB): ~20 GiB left; better `--resident-budget-gib ~48` on the split (GPUs hold ~30 GiB, budget the rest, no file reads) |
| share of expert reads from VRAM, 2 cards | ~93-95% | ~80-85% |

The n-gram shard is byte-identical across ISTA's Q2_0 / IQ2_XS / IQ3_XXS / IQ3_S / Coder files
(same SHA-256), so it was copied from the earlier Q2_0 download.

## 3. Expert misses: stream to the GPU or compute on the CPU?

Each token routes 10 experts per layer. A hit is computed from VRAM. A miss is either computed
by the CPU from RAM, or read by the GPU from pinned RAM over PCIe; `--pcie-frac` sets the share
of misses the GPU takes (`./setup.sh --calibrate` measures it per PC, together with
`--pool-workers`, `--spec-min-p` and the cache-adaptation settings).

| path for a miss | bandwidth |
| --- | --- |
| VRAM (a hit) | ~936 GB/s |
| CPU from this RAM (DDR4-3200, 3 sticks, measured) | 25-36 GB/s, median ~29-30 |
| GPU over PCIe 4.0 x8 | ~13 GB/s per card |

**Decode**: a missed expert is used for only the few rows of one verify pass, so its bytes cross
a bus every pass. Q4 on 2 cards misses ~0.8 GB per pass:

- CPU alone: 0.8 / 29 = ~28 ms
- PCIe alone: 0.8 / 13 = ~62 ms (pipeline stages take turns, so effectively one link at a time)
- best split, ~30% to PCIe: ~19 ms, ~30% less than the CPU alone

against ~8 ms for the GPU work of a whole pass. So for Q4 on 2 cards the misses dominate,
streaming softens it, and the slower single-channel region of the RAM shifts the best split
further towards PCIe. IQ3_S misses ~0.2 GB per pass (~5-7 ms CPU, mostly hidden).

**Prefill**: a 16K-token chunk uses each expert thousands of times, so copying the missing
experts to the GPU once per chunk is cheap against the compute: prompts stream experts over
PCIe. This is where the fork's tuning wins most (its Q4 prompts 3.1-4.1x upstream's).

**Adaptive caching** (`--adapt-every`, `--adapt-swaps`): the VRAM cache follows the
conversation's hot experts. Upstream's docs: a PC "whose CPU reads the missed experts slowly
(DDR3 or DDR4 in few channels) gains from swapping more" (+7.9% on a DDR3 Xeon).

**Consequence for the third 3090**: Q4's decode is limited by misses, not by GPU speed. A third
card raises Q4's expert residency from ~43% to ~61% and the VRAM read share from ~80-85% to
~92-94%, roughly halving the misses. Whether that is worth buying depends on Q4 being clearly
better than IQ3_S on our traffic, so measure quality first.

## 4. Disks and the n-gram table

- The 28.8 GB n-gram (PLE) table is read with direct I/O past the OS cache (`--ple-io direct`,
  default) unless setup keeps it in RAM (`--ple-io ram`, when RAM >= the model's need + table +
  16 GB: 107 GB for IQ3_S, so not at 96 GB, possibly at 128 GB).
- ~16 rows per token (the as-built note): ~2.4K rows/s at 150 tok/s decode, trivial for any SSD.
  Prefill at 3,000+ tok/s would want ~48K rows/s if every token's rows were read; upstream
  measured a SATA SSD at 10-14K rows/s and calibrates its warning at 15K for NVMe. Hence the
  NVMe. (The as-built note's "~3-4 GB/s" is MB/s: 23 KB x 150 tok/s = ~3.5 MB/s.)
- Space: models ~195 GB on the 469 GB NVMe. `vllm-qwen.service` has `DISK_TIER_GIB=400` on the
  same disk; its free-space guard copes but cleans every ~6 s at that point, so lower the cap to
  ~200-230 before production restarts.
- Q2_0 (superseded) stays on `/d/ssd_128` and the HDD archive.

## 5. State and next steps

Ready: upstream and fork checkouts (`~/git/strata/`), IQ3_S and UD-Q4_K_XL on the NVMe, the
pinned llama.cpp source for the fork build (`~/git/strata/deps/`). Plan with commands:
`~/git/strata/PLAN.md`.

When the box is idle (a CUDA build skews benchmarks):

1. Build the fork for sm_86 (CUDA 13.4, `-DSTRATA_GGML_DIR=` the pinned llama.cpp).
2. With production down: upstream `./setup.sh --setup --model IQ3_S --data-dir
   /d/nvme_512/strata-data --gpus 0,1 --no-start` (config, MTP draft layer), then
   `./setup.sh --calibrate` (15-30 min) for `--pcie-frac` and the adaptation settings.
3. IQ3_S on the fork and on upstream: decode (code/thinking, greedy and 0.7), prefill 16K/128K,
   `--layer-split` sweep (upstream), server log's expert tiers and `pcie_share`.
4. UD-Q4_K_XL on upstream with `--resident-budget-gib ~48`: quality against IQ3_S on
   `bench/quality_battery.py` and replayed Hermes tool-call / JSON requests (502 rate); decode
   for the record.
5. Decide the third 3090 from Q4's quality margin; revisit with 128 GB when the stick arrives.

## Sources

- [eddoursul/Strata `custom`](https://github.com/eddoursul/Strata/tree/custom): `docs/COMPARISON.md`,
  `examples/README.md`, `docs/DETAILS.md`
- [Niko1221/Strata](https://github.com/Niko1221/Strata): `docs/DETAILS.md` (low-RAM modes,
  `--pcie-frac`, `--calibrate`, `--ple-io`), `docs/MULTI_GPU.md`, `docs/UNSLOTH_Q4.md`,
  `docs/QUANTS.md`, `setup.py` (pinned files, `unsloth_budget`), git log 2026-10-19..23
- hardware repo: `ram-install-checklist-2026-10-22.md` (bandwidth), `as-built-c8e-2x3090-2026-09-28.md`,
  `strata-flash-next-report-2026-10-19.md`
