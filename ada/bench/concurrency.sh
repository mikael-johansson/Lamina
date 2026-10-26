#!/bin/bash
# Upstream vs fork at C=1/2/4, IQ3_S and UD-Q4_K_XL (~45-70 min). Needs both GPUs and no Strata server running.
#   bench/concurrency.sh            # everything
#   bench/concurrency.sh up-iq3     # only the runs whose tag starts with this
cd "$(dirname "$0")/.."
if pgrep -f '[s]erve/server.py --engine strata' >/dev/null; then
  echo "a Strata server is running: stop it first (./strata.sh stop)"; exit 1
fi
out=bench/runs/concurrency-$(date +%Y%m%d-%H%M).txt
run() {   # tag config levels
  [[ -n "$only" && "$1" != $only* ]] && return
  python3 bench/load.py "$2" --tag "$1" --levels "$3" 2>&1 | tee -a "$out"
}
only=$1
run up-iq3-p1    configs/up-iq3_s.json             1
run up-iq3-p4    configs/up-iq3_s-p4.json          1,2,4
run fork-iq3     configs/fork-iq3_s-tuned.json     1,2,4
run up-q4-p1     configs/up-ud-q4_k_xl.json        1
# up-q4-p4 removed: 77 GB of experts + 4 slots overran the 92 GiB RAM (swap storm, GPU driver stalls, 2026-10-25)
run fork-q4      configs/fork-ud-q4_k_xl-tuned.json 1,2,4
echo "summary: $out"
