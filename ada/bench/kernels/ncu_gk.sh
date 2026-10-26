#!/bin/sh
# Nsight Compute on bench/gk (standalone kernels: safe under ncu's serialized replay, unlike the engine's decode):
#   sudo sh ada/bench/kernels/ncu_gk.sh   -> ada/bench/results/kernels/ncu-gk-<kernel>.ncu-rep, a minute or two (not in
#   git: a report holds the host's addresses; its details page is, as .txt: ncu --import <rep> --page details)
cd "$(dirname "$0")"
for k in gr_down_q8_split gr_up_q8_multi verify_router gdn_ab_multi; do
  timeout 600 /usr/local/cuda/bin/ncu --set full --cache-control all --clock-control none \
    -k "regex:$k" --launch-skip 50 --launch-count 3 -f -o ../results/kernels/ncu-gk-$k ./gk 4 2 > ../results/kernels/ncu-gk-$k.log 2>&1
  chown mikaeljo:mikaeljo ../results/kernels/ncu-gk-$k.*
  echo "$k: $(tail -1 ../results/kernels/ncu-gk-$k.log)"
done
