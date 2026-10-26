#!/bin/sh
# WARNING: deadlocks. ncu serializes the verify graph, whose kernels spin on host / other-GPU flags (see
# ada/docs/KERNEL-PROFILE-2026-10-26.md). Kept as a record; profile bench/gk with ncu_gk.sh instead.
# Nsight Compute on the decode's hot kernels (needs root while NVreg_RestrictProfilingToAdminUsers=1):
#   sudo sh bench/ncu_decode.sh
# A direct decode of the code prompt (200 tokens, the split config); after the prompt, 150 launches of the matching
# kernels profiled with the full section set, real clocks and caches.  The report: bench/runs/ncu-decode.ncu-rep
ARGS=$(python3 -c "
import json; c=json.load(open('/home/mikaeljo/git/strata/configs/fork-iq3_s-split-batch.json')); print(' '.join(a for a in c['args'] if a!='--vision'))")
cd /home/mikaeljo/git/strata/eddoursul
/usr/local/cuda/bin/ncu --target-processes all --graph-profiling node --set full --clock-control none --cache-control none \
  -k 'regex:gr_down_q8_split|gr_up_q8_multi|verify_router|native_gus80|native_down_sub|native_mmvq_il|native_mma_kernel|gdn_step_split|native_mmvq_multi|gdn_ab_multi|bf16_f32_mmvf_multi|gdn_conv_l2' \
  --launch-skip 4000 --launch-count 150 -f -o /home/mikaeljo/git/strata/bench/runs/ncu-decode \
  ./build/strata.dsplit $ARGS --tokens-file /tmp/claude-1000/-home-mikaeljo-git-strata/fe895871-8bbc-4761-b845-da895fd312d2/scratchpad/code_tokens.txt --max-new 200 --stats
chown mikaeljo:mikaeljo /home/mikaeljo/git/strata/bench/runs/ncu-decode.ncu-rep
