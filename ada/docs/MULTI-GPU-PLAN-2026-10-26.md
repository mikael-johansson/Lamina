# Multi-GPU findings and plan (2026-10-26)

IQ3_S on 2x RTX 3090 (PCIe 4.0 x8/x8), Ryzen 7 5700X, the eddoursul fork. Priorities: C=1 first, then C=2,
C=3/4; nothing above C=4 is needed. Numbers marked *estimate* are derived from stage times and weight sizes, not
measured end to end.

## Where a decode step's time goes today

A verify window (~3.3-4 tokens checked at once) is ~15.6 ms at short context. Per layer with CPU work (74% of
layers), medians from the timeline (`STRATA_POOL_TIMELINE`, `bench/results/cpu-pool/`):

| | us |
| --- | --- |
| main card before the router: HC read, mixer, HC read, router | ~150 |
| experts: the slower of the main card's own (~70) and the CPU's misses (~161) | ~161 |
| read the CPU's rows + combine | ~6 |
| **per layer** | **~317** |

- **The main card is latency-bound, not bandwidth-bound.** Its pre-router work reads ~65 us worth of weights at
  full VRAM speed but takes ~150 us: ~85 us per layer is per-kernel fixed cost. Achieved rates from the weight
  sizes (GGUF) and the stage times: GDN mixer ~550 GB/s, attention mixer ~400, each HC read ~140 (3 MB in
  ~23 us), the router ~120 (2.5 MB in ~22 us); the experts the main card holds ~690.
- **The CPU's misses are the longest piece** in most layers (~161 us vs the main card's ~70 us of expert work).
  Gate/up is compute-bound (codebook lookups, ~4.5 GB/s per core even from L2), down nearly RAM-bound at 6+
  cores (`bench/kbench.out`). There is no sync overhead in the pool. See `ada/README.md`, "Where the time goes".

## TP=2: revised verdict

Earlier estimate (overnight): C=1 +25-35%. Revised: **~+10-15% with today's misses** (*estimate*), for weeks of
work, because:

1. The main card's work does not halve when split: the bandwidth part (~65 us/layer) halves, the ~85 us fixed
   part does not, and TP adds two exchanges per layer (~10-20 us).
2. The CPU's misses are unchanged (TP holds the same experts, split by rows), and they are already the longest
   piece; a faster GPU side only waits longer for them.

| per layer with CPU work | today | TP=2 (*estimate*) |
| --- | --- | --- |
| pre-router | ~150 us | ~100 + ~15 exchange |
| experts (slower of GPU / CPU) | ~161 | ~161 |
| combine | ~6 | ~6 |
| total | ~317 | ~282 (about +12%) |

## With every expert in VRAM

IQ3_S's experts are ~50 GB (24,576 x ~2.05 MB). Two cards hold ~21,400 today; all of them need either ~20%
smaller experts on 2 cards (~2.8 instead of ~3.5 bits per weight: a quality cost) or a 3rd card.

| design, everything resident | per layer (*estimate*) | C=1 vs today | cost |
| --- | --- | --- | --- |
| today's main + second-card tier | ~185 us; ~246 with the pre-armed second-card graph | none to +29% | small; the second card's host round trip (~100 us per share) limits it |
| **layer split for decode** (each card runs its own layers, all their experts local) | ~236 us | **~+34%** | ~1 week in the fork |
| TP=2 | ~189 us | ~+68% (+25% over the layer split) | weeks |
| TP=3 | about TP=2 | no gain over TP=2 | weeks+ |

- **Layer split:** no CPU path, no host-mediated second-card share; one hand-off per window per card boundary
  (~40 KB per row: 4 HC streams x 2,560). The 3rd card's x4 link does not matter for it.
- **TP=3 is not worth it:** 2 KV heads, the expert width 640 (10 quant blocks) and 2,560 (40 blocks) do not split
  three ways; the fixed per-layer cost stays; the x4 card would be the slowest member of every exchange.
- Biggest uncertainty: that every layer's experts run at ~80 us when all are local (today's main card reaches
  ~690 GB/s on its share, so plausible).

## Upstream's layer split, measured today

Through the server, same requests as the fork benchmarks (`bench/results/up-ls-*`, `up-pipe2`):

| | code tok/s | Claude Code 40K tok/s | hit rate |
| --- | --- | --- | --- |
| fork, two-tier (today) | 232-240 | 176-185 | 98-99% |
| upstream layer split, default | 177 | 148 | 98.2-98.8% |
| + `STRATA_SPLIT_OVERLAP=1` | 178 | 146 | |
| + `--pipeline-windows 2` | 175 | 155 | |

- **Not the misses:** the hit rates match and both engines send misses to the CPU the same way.
- **Not mainly the split:** upstream's two knobs for its stage overheads gain at most 4% on NVIDIA here (its docs
  measured +21-32% on AMD).
- **Its per-layer GPU work is ~30% slower:** in direct mode on one card its host waits 10.8 ms per window for the
  GPU's non-expert work, the fork's 8.4 ms (~226 vs ~175 us per layer; caveat: a one-card run, upstream profiles
  only one card outside server mode, `bench/results/cpu-pool/upstream-one-card.out`). One concrete cause: the fork
  keeps the hyper-connection matrices in 8 bits (`--hc-q8`, 644 MiB), upstream in BF16 (1,288 MiB): double the
  bytes in the most latency-bound stage. The fork has other kernel work upstream lacks.
- **So it does not go away without misses;** it becomes a larger share. Upstream on 3 cards with everything in
  VRAM: ~226 + ~80 = ~306 us per layer (*estimate*), about today's fork. Useful as a test that the CPU path
  disappears, not as the measure of what a 3rd card gives.
- **Pipelined concurrency** (`--batch 2 --batch-groups 2`, short prompts): C=1 137-155 total, C=2 118-146 total
  (65-79 per request): no gain; the two requests take turns (most likely both stages queue on the one CPU pool
  and host loop; not profiled). MTP is off in that mode (`--batch-mtp` needs `--batch-groups 1`), and each slot
  took 0.47 GiB per card from the expert cache.

## Plan

The fork's speed comes from per-layer kernels, and a layer split runs the same layers, so they carry over. Build
the layer split in the fork (porting the fork's kernels into upstream is harder: the engines are 621 / 789
commits apart).

### Phase 1: decode layer split in the fork, C=1 (~1 week)

- One verify graph per card for its layer range, with that range's GDN state, QSA K/V and PLE history resident on
  it for good (today's `--prompt-split` copies layers' weights and state per prompt, on the prompt kernels).
- The window's hidden state (4 HC streams) handed card to card once per window, over P2P.
- One host loop serving each layer's routing on whichever card runs it; the CPU pool for any misses left.
- The output head and the MTP drafter on the last card (no head split).
- Each card's expert cache holds its own layers' experts (profile-ranked per layer range, adaptive tier per card).
- Carries over unchanged: `--hc-q8` and the fused HC kernels, native projections, the GDN/QSA kernels, the R4 hit
  path decided on the device, the captured windows, MTP, k8v4 K/V and its growth, the CPU pool, `--arena-place`.
- Dropped in this mode: the second-GPU tier and its host round trip, `--head-split`, `--prompt-split` as it is.
- Test: greedy tokens identical to today's engine under `--deterministic-adapt`; decode and prefill through the
  server against `bench/results/ram-*`.
- Expected on 2 cards: C=1 about today's (+-5%; same VRAM, same misses). On 3 cards with all of IQ3_S resident:
  ~+30-35%.

### Phase 2: two requests in flight (~1 more week)

- While card A runs request 2's layers, card B runs request 1's: each request keeps its own verify window.
- MTP per request (shared weights, own draft K/V ring), up to 4 drafts each as now: unlike upstream's batching,
  which merges requests into one window with one draft per request.
- A host thread per card (upstream's pipelined mode gains nothing here, most likely because the stages share one
  host loop and pool).
- **Request state borrows expert-cache slots only while a second request is active** and gives them back after
  (as the fork's K/V growth does): permanent slot sessions (~0.5 GiB per card per request) would cost ~1,000
  cached experts per card and C=1 speed (the miss cliff: ~2,400 fewer experts gave ~10x the misses).
- Expected on 2 cards: C=2 up to ~1.5-1.7x total (CPU-bound: the pool is busy ~40% of a window at C=1, ~80% with
  two in flight), each request ~80-85% of solo. On 3 cards the CPU mostly drops out and the headroom grows.
- C=3/C=4: 2 cards give two lanes, so requests 3-4 share them (total about C=2's, each request slower); 3 cards
  give three. Merging requests inside a window (the overnight batching design) would be a later step if needed.

### Before the code

- When the riser arrives: upstream's layer split on 3 cards (all of IQ3_S resident) to confirm the CPU path
  disappears and to measure the per-layer expert cost, knowing upstream itself is ~25-30% slower per layer.
- Hardware: the riser goes in the CPU-attached M.2 slot (today the Samsung PM981, the box's only NVMe, which moves
  to a chipset M.2 slot at no loss: it is a PCIe 3.0 drive). The 3rd card is then on the CPU's lanes like the other
  two, so P2P should work (`nvidia-smi topo -m` shows PHB between the current two).
- The fork drives exactly one extra card today (`--second-gpu N`); 3 cards need Phase 1.

## Other levers, for scale

| lever | C=1 | note |
| --- | --- | --- |
| 3rd card + layer split | ~+30-35% (*estimate*) | removes the CPU path for IQ3_S |
| 5900X / 5950X | ~+8..+20% (*estimate*) | gate/up is compute-bound and scales with cores; less the two-chiplet sync cost |
| fewer, fused kernels on the main card | unknown | attacks the same ~85 us/layer fixed cost TP cannot; needs a per-kernel profile (no nsys on the box) |
| CPU overclock (PBO/CO) | +1.5-3% | cores already at ~4.5 of 4.65 GHz under this load |
| VRAM headroom | small | ~1 GB unused: ~500 more experts |

## Status at the end of 2026-10-26 (branch `layer-split`)

Built: `--decode-split L` (balanced cards: each GPU its own layers' weights, experts and state; the head on the
last, MTP on the first; prompts in two stages on the resident state; the prompt cache split-aware) and `--batch N`
(the server's `"parallel"`): batch slots with BGEN / BT / BDONE / BSTOP, slot sessions in lent expert-cache slots,
admissions read beside decoding slots. Config: `ada/configs/fork-iq3_s-split-batch.json` (port 8096).

| | production | split + 2 batch slots |
| --- | --- | --- |
| short tasks, C=1 total | 223.5 tok/s | 244.8 |
| short tasks, C=2 total | 231.2 (queued) | 316.4 |
| 18 real requests, C=1 (shipped profile) | 171.1 | 167.6 |
| prefill 22-146K tokens | 3,015-4,452 | 3,131-4,583 (-4..+4%) |
| prefill <= 13K tokens | | -15..-30% (each stage streams its card's missing experts) |
| two 49K prompts at once | ~26-28 s queued | 24.2 s; the other request decodes at ~47 tok/s during a read |

Bit-exact against the unsplit engine (`STRATA_DSPLIT_NOHIT=1`, equal tiers). Limits: a slot holds 128K tokens
(`STRATA_SLOT_CELLS`, 1.5 GB lent per slot); penalties and images on the solo path; the adaptive tier off in split
mode; prompts are read one at a time. Next: slot state that grows with its conversation.

### Fix: prompt-cache restore with batch slots (later on 2026-10-26)

With `--decode-split` and `--batch`, every request that went on from a cached prefix crashed the engine (SIGSEGV,
the server restarted it and answered 503): `PromptCache::restore` loaded all 48 layers of a checkpoint into the main
session, while a checkpoint holds the main GPU's layers only up to the split (the second GPU saves its own).  The
main session's QSA pools of the later layers are frozen, and with batch slots unmapped and lent, so the copy went to
an unmapped address (the driver takes it for host memory).  Restore now loads the layers `checkpoint()` saved.
Without `--batch` it only copied unused state.  The `pf-split-batch` prefill run had three of these crashes
(opencode-30k, cc-40k, cc-80k-next): its numbers for those requests are a restart's, not the split's.

### `--spec 5` on the split (C=1, the corpus through the server, after the fix)

Geometric mean over the 8 requests, three interleaved rounds each (results `c4-r*`, `c5-r*`; an earlier set overlapped with a second session's runs on the same GPUs and port and was dropped): spec 4 225.3 / 225.0 / 224.5 tok/s, spec 5 225.7 / 225.2 / 224.8: +0.1%, noise.  No engine restarts in any of the six runs (the checkpoint fix).
Per request -6% to +8%, partly because the outputs differ (a different window size verifies other drafts).  The
direct code prompt (same tokens): spec 4 225.1 / 224.3, spec 5 219.5 / 220.4 (-2%); spec 5 accepts all 4 drafts in
83 of 159 windows, but a 5-token window costs more.  Kept spec 4.
