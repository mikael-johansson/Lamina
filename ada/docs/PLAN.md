# Strata on this box: preparation (2026-10-19)

Background and speed estimates: `~/git/vllm/qwen38-27b-rtx3090/docs/strata-flash-next-report-2026-10-19.md`
(this file is only about what is set up, and how to run the first test).

## Checkouts

| dir | what | notes |
| --- | --- | --- |
| `upstream/` | Niko1221/Strata `main` (e88ee39d, engine 0.2.0-beta / 0.1.45.3) | **use this** |
| `eddoursul/` | eddoursul/Strata `custom` (b0e551e6) | 621 ahead / 789 behind upstream, base 2026-10-16 |

Why upstream, not the fork:
- The fork drops the **layer split**, the **low-RAM / resident modes**, KV streaming and `--batch`
  (examples/README.md). Its 2nd GPU is only an expert-cache tier (`--second-gpu`), so no 3-card setup and no
  32 GB-RAM test.
- The fork is built and measured on Windows (MSVC, `.bat` configs); upstream ships a ready-made Linux CUDA 13
  engine (`strata-engine-0.1.45.3-linux-x86_64-cuda-cuda13.tar.gz`), and driver 615.71 / CUDA 13.4 is fine.
- What the fork has that matters (UD-Q4_K_XL, two-stage prompts): upstream runs UD-Q4_K_XL too
  (docs/UNSLOTH_Q4.md, with a layer split since 0.1.40). Revisit the fork's prompt-split only if upstream's
  prefill measures far below its numbers.

## Test now with 32 GB RAM: Q2_0 on 2x3090

- Upstream's **low-RAM resident mode** (`--low-ram resident`, layer split since 0.1.40): the experts the GPUs
  do not hold are copied into locked RAM, the rest of RAM is free. Q2_0 has 34 GB of experts; two dedicated
  3090s hold ~32-36 GB of them, so nearly everything runs from VRAM and RAM holds ~0-4 GB.
- IQ3_XXS (43 GB experts, ~5-10 GB in RAM) would probably fit too; IQ3_S (50 GB) is a stretch on 32 GB.
- The 28.8 GB n-gram shard (`-00002-of-00002`) is **byte-identical** for Q2_0 / IQ2_XS / IQ3_XXS / IQ3_S /
  Coder (same SHA-256 in setup.py), so moving to IQ3_S later adds only its 54.8 GB shard 1.
- Files: `/d/hdd_800/strata-data/models/Q2_0/` (pinned revision ed59f920, 66.4 GB; setup checks the SHA-256).
  Copied to the 128 GB SATA SSD (`sdc`, mounted at `/d/ssd_128` via fstab) for testing: the n-gram table gets random reads
  per token and 32 GB of RAM cannot cache all of it. The disk layout setup expects is
  `<data-dir>/models/Q2_0/<file>`; the MTP draft layer (~6 GB) lands in `<data-dir>` on first start.
- Disk: `/` has 6 GB free (venv + engine ~1-3 GB is OK in `upstream/`); NVMe is 96% full (401 GB is the vLLM
  KV disk tier).

### Test day (production down)

The test needs both GPUs free: stop `vllm-qwen.service`, and move the voice stack (voxcpm ~5.9 GB, whisper
~0.7 GB) off them, or leave it on and expect fewer experts in VRAM.

```sh
cd ~/git/strata/upstream
# STRATA_UV=1: a uv Python 3.12 venv instead of the system 3.14 (safer for the pip wheels)
STRATA_UV=1 ./setup.sh --setup --model Q2_0 --data-dir /d/ssd_128/strata-data \
  --gpus 0,1 --low-ram resident --context 131072 --no-start --yes
./setup.sh                      # starts the server, 127.0.0.1:8080, OpenAI + Anthropic APIs
```

Measure (as in the report's section 10): decode on code/thinking prompts (greedy and 0.7), prefill at 16K and
the longest context that fits, a `--layer-split` sweep (auto, then e.g. 22/24/26), the server log's
`resident RAM: ... blob reads from the file` (should be 0), free RAM during long prompts, and replayed Hermes
requests (tool calls, JSON validity, 502 rate). Then start production again.

## After the 128 GB kit

- 2 cards: IQ3_S (`--model IQ3_S`, every expert in RAM, no low-RAM mode). Download only its shard 1 into
  `models/IQ3_S/` and link shard 2 from Q2_0's folder.
- 3 cards (3rd on the M.2 x4 riser): `--gpus 0,1,2`, UD-Q4_K_XL (`--family unsloth`, 111 GB download, does not
  fit on the 128 GB SSD beside anything else: HDD or NVMe) or IQ3_S. Sweep `--layer-split` (auto misplaced an
  x16+x4 pair); give the x4 card fewer layers (report section 4.3).
- `/dev/shm` and the vLLM RAM tier only matter if the hybrid (27B on vLLM + Strata) is chosen.

## Update 2026-10-23: 96 GB RAM (3 sticks, one defective; replacement expected -> 128 GB)

- Models now on the NVMe (`/d/nvme_512/strata-data/models/`; vLLM's KV disk tier was emptied):
  `IQ3_S/` (shard 1 downloaded, shard 2 = the verified Q2_0 n-gram table) and `UD-Q4_K_XL/` (4 shards, 111 GB).
  Script: fetch_quants.sh in the session scratchpad; every file checked against setup.py's SHA-256.
  Before production restarts: `DISK_TIER_GIB=400` in vllm-qwen.service no longer fits beside ~195 GB of models;
  the free-space guard copes, but noisily (~6 s cleanups), so lower it to ~200-230.
- Q2_0 stays on `/d/ssd_128` + the HDD archive (superseded).
- **Engine choice: try the eddoursul fork first** (perf-focused, measured 1.35-4.1x upstream's layer split on a 2-GPU
  box; needs every expert in RAM, which 96 GB allows for IQ3_S ~53 GB; Q4 ~81 GB is tight), upstream engine +
  `serve/server.py` pinned to one engine version as the A/B. Upstream's recent churn is mostly the desktop app /
  terminal; engine work there is mostly opt-in (DFlash2, STRATA_SPEC_COST). For 3 cards later: upstream (layer split).
- Fork build (do it when the box is idle, it is a long CUDA compile):

      cd ~/git/strata/eddoursul && PATH=/usr/local/cuda-13.4/bin:$PATH cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DSTRATA_GGML_DIR=$HOME/git/strata/deps/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d
      cmake --build build --target strata -j 12

  UD-Q4_K_XL's pack needs `--compat-bf16` with upstream's `tools/iq_pack.py` (fork examples/README).
- Q4 on 2 cards is for **quality** (is a 3rd 3090 worth it?); speed on 48 GB VRAM will be low.

## Fork smoke test: ready (2026-10-23)

- Engine: `eddoursul/build/strata` (a85301ca, sm_86, CUDA 13.4, GCC 15; built in 3 min with
  `-DSTRATA_GGML_DIR=deps/llama.cpp-3cf03257...`). `--pcie-frac` is parsed but missing from `--help`.
- Python: `eddoursul/.venv` (system 3.14). Tools that read GGUF need
  `STRATA_GGUF_PY=~/git/strata/deps/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d/gguf-py`.
- Packs: `/d/nvme_512/strata-data/packs/{iq3_s,ud-q4_k_xl}` (Q4 with `--compat-bf16`).
- Draft layer: `/d/nvme_512/strata-data/mtp/rt` (ours: experts.bin and dense.bin byte-identical to basalt's
  published ones) and `mtp/rt-tuned` (basalt's tuned dense.bin, sha dfe18647...). Draft vocab: cjk (default).
- Configs: `configs/fork-iq3_s.json` (port 8091), `configs/fork-ud-q4_k_xl.json` (8092): the fork's examples with
  Linux paths, no vision, main GPU 1 (FTW3), second GPU 0 (FE). Left out for later A/B: `--prompt-split 26`,
  `--arena-place` (uneven RAM!), `--no-file-cache`; tune `--pcie-frac` (0 / 0.15 / 0.3) and
  `--second-gpu-prefetch-mb` for the x8 links.
- Run (GPUs free): `cd ~/git/strata && eddoursul/.venv/bin/python smoke.py configs/fork-iq3_s.json`

### Smoke result, fork IQ3_S (2026-10-23, 21:16)

Experts 46.84 GiB into RAM at 1.80 GiB/s; main-GPU cache 9,452 experts / 17.93 GiB; ~39 GB RAM still available.
Greedy decode: code 600 tokens at 190.3 tok/s (cache hit 97.9%), thinking 131 tokens at 168.6 tok/s (97.6%),
sanity 4 tokens correct. In line with the fork's own 199-209 / 162-172. Next: fork-ud-q4_k_xl.json.

### Smoke result, fork UD-Q4_K_XL (2026-10-23)

Ready after 221 s: experts 71.73 GiB into RAM at only 0.34 GiB/s (IQ3_S load did 1.80; probably page-cache
eviction / memory pressure at 96 GB). Main-GPU cache 5,761 experts / 16.81 GiB.
Greedy decode: code 600 tokens at 85.5-86.6 tok/s (cache hit 94.2%), thinking 145 tokens at 70.5-74.6 (91.2%).
Answers correct (391; 201 min). Below my 105-140 estimate and the fork's 130-133: misses dominate on 2 cards.
Tuning to try: --pcie-frac 0.15/0.3, --arena-place, --adapt-swaps, --pool-workers.

### First tuning pass (2026-10-23), bench/ (bench.py replays real logged requests, results in bench/runs/<tag>/)

IQ3_S: `--arena-place --no-file-cache --prompt-split 26` = +21-27% prefill at 90-146K (3,350 -> 4,070-4,250 tok/s),
decode same or better (code 230, thinking 192, 90K 169), start 46 -> 26 s. `--pcie-frac 0.15` cost decode (-5..-17%): keep 0.
-> configs/fork-iq3_s-tuned.json
Q4 with the same flags: start 221 -> 72 s, code 85 -> 118 tok/s. pcie-frac 0 / 0.2 / 0.35: code 118 / 107 / 103,
thinking 81 / 84 / 85, 40K agent turn 70 / 76 / 68 (prefill ~2,080 throughout). Picked 0.2 (agent turns) ->
configs/fork-ud-q4_k_xl-tuned.json. Single runs; differences under ~10% are within noise.

### Abliterated IQ3_S (SC117, 2026-10-23)

SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF rev eec4e10a: shard 1 (55.1 GB, SHA-256 OK) on
/d/ssd_128/strata-data/models/IQ3_S-abliterated-sc117/; shard 2 = our NVMe n-gram table (same SHA; symlinked, and
--ple-gguf points at the NVMe copy). 144 tensors (attn_output, ffn_down_shexp, ffn_down_exps; donor orcarouter
Uncensored) are read from the GGUF, so the pack's dense.bin is byte-identical to the original. Pack:
/d/ssd_128/strata-data/packs/iq3_s-abliterated-sc117. Config: configs/fork-iq3_s-abliterated.json (port 8093).
Draft layer: ours (measure acceptance). Start: ./strata.sh start configs/fork-iq3_s-abliterated.json

## Vision + second-GPU reserve (2026-10-25)

- CPU picture reader: `eddoursul/build-vision/bin/strata-vision` (tools/vision, native, no CUDA); encoder
  `/d/ssd_128/strata-data/vision/mmproj-Qwen3.8-Flash-Next-BF16.gguf` (ISTA rev ed59f920, size/sha OK).
  `"vision"` section (gpu false, max_tokens 1024) + `--vision` in the three tuned configs. 1080p screenshot:
  ~14 s first time (~13 s encode at ~1,000 image tokens), 1.0 s when the same picture is sent again; text read correctly.
- `--second-gpu-reserve-mib 512` (default 2048): second card 427 MiB free at idle, ~340 MiB at the 150K-prompt peak.
  IQ3_S 11,259 -> 11,933 second-tier experts; Q4 7,283 -> 7,797. Same-day A/B (512 vs 2048, 262K ctx):
  IQ3_S prefill +1-3.5%, decode within noise; Q4 prefill +2-3%, decode +6-13% on 5 of 8 requests, -11% on one.
- Backups of the configs before this: configs/*.bak-prevision.

## Upstream A/B prepared (2026-10-25)

- Upstream (`upstream/`, engine 0.1.50 from the signed feed) set up into `/d/ssd_128/strata-upstream` (packs + MTP copy;
  GGUFs read from `/d/nvme_512/strata-data/models`). Layer split auto on GPUs 0,1, KV streaming, all experts in RAM.
- Configs: `configs/up-iq3_s{,-p4}.json` (8094), `configs/up-ud-q4_k_xl{,-p4}.json` (8095); -p4 = `"parallel": 4`.
- `bench/load.py` (C concurrent streaming clients; short coding prompt and the 49K opencode request with per-client
  nonce) and `bench/concurrency.sh` (upstream p1/p4 and the fork, both models). Not run yet.

## Where the time goes, and batching headroom (2026-10-25)

Data: `bench/runs/trace-iq3`, `trace-q4` (18 real requests, `--dump-routing`), `bench/route_analysis.py`, x-* runs.
- IQ3_S decode: misses 0.37% of lookups (~6 distinct experts, 12 MB per step). Main card ~99% busy at its 300 W cap,
  second ~50% busy (~155 W). Step cost vs verified rows (spec 2/4/6/8, min-p 0): short ctx ~8.3 ms fixed + ~1.75 ms
  per row; 40K ctx ~8.4 + ~2.4 ms per row. Single GPU: 126 vs 217 tok/s (code).
- Batching (simulated from the trace): distinct misses per request fall with C (IQ3_S 5.4 -> 4.1 at C=4).
  Linear step model -> C=2 ~1.35x, C=4 ~1.7x total (more if drafts are batched); 40K ctx C=4 ~1.4x.
- Miss cliff: IQ3_S at 19,000 resident (vs 21,394) -> 4.6% misses (12x). Slots must not take expert-cache VRAM.
- Q4 decode: 6.6% misses (~90 experts, 280 MB per step from RAM, CPU-bound; main card ~45% busy). 3 cards
  (~20,850 resident) -> 0.41%, 4 cards -> 0.
- Prefill: GPU compute-bound, second card at 100% with --prompt-split (IQ3_S main ~70-90%; Q4 main ~23%, second
  also streams ~45% of the experts per chunk).
- Upstream batching with MTP (`--batch-mtp --batch-groups 1`, configs/up-iq3_s-p{2,4}-mtp.json): C=2 136-162 total,
  C=4 151-170 (C=1 147-168). Cause: one MTP proposal per slot (hard-coded `set_max_drafts(1)`), so a 2-slot window
  has the solo window's ~3.8 rows; and upstream's window is ~21.5 ms at 3.75 rows vs the fork's 14.8 ms at 4 rows
  (layer split: each stage waits ~7 ms). Fork 8-row window measured 22.1 ms (short) / 25.7 ms (40K): 2 slots x 4 rows
  -> est. C=2 ~1.35x short ctx, ~1.2x at 40K. Port target: upstream's run_slot_rows / commit_slot_prefixes into the
  fork's Verifier with k drafts per slot.

## Overnight 2026-10-25/26: see OVERNIGHT-2026-10-26.md

- `--second-gpu-prefetch-mb 0` in all three tuned configs: IQ3_S decode +4..+48% (more with context), Q4 +21..+48%.
- IQ3_S `--prompt-split 28` (was 26): prefill +1-2%. Q4 stays 26.
- Production `eddoursul/build/strata` is the original binary; experiments in `patches/fork-gpu2-experiments.patch`
  (binary `build/strata.hotsplit-exp`).
- Batching and TP=2: designs + measured cost models in the report; multi-day ports, not started.
