#!/bin/bash
# Take a physical address range out of use (or give it back) without a reboot, through memory hotplug.
# Usage: sudo bench/ram_offline.sh <from GiB> <to GiB> offline|online
# Blocks are 2 GiB here (/sys/devices/system/memory/block_size_bytes). A block with unmovable pages
# refuses to go offline; the script reports it and carries on. "online" undoes it.
set -u
from=$1 to=$2 action=$3
bs=$(( 0x$(cat /sys/devices/system/memory/block_size_bytes) ))
ok=0 fail=0
for (( b = from * 2**30 / bs; b < to * 2**30 / bs; b++ )); do
  f=/sys/devices/system/memory/memory$b/state
  [ -e "$f" ] || continue
  if echo "$action" > "$f" 2>/dev/null; then ok=$((ok + 1)); else fail=$((fail + 1)); echo "memory$b: $action failed"; fi
done
echo "$action: $ok blocks done, $fail failed"
grep MemTotal /proc/meminfo
