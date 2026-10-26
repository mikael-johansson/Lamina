#!/bin/sh
# CPU package power (RAPL) and clocks: idle, 8 cores scalar, 8 cores AVX2 integer SIMD (the expert kernels' ops),
# 1 core AVX2.   sudo sh bench/cpu_power.sh
E=/sys/class/powercap/intel-rapl:0/energy_uj
B=$(dirname "$0")
watts() { a=$(cat $E); sleep 4; b=$(cat $E); echo $(( (b - a) / 4000000 )); }
mhz() { awk -v c="$1" '{if (c == "" || NR-1 == c) {s+=$1; n++}} END {printf "%.0f", s/n/1000}' \
        /sys/devices/system/cpu/cpu[0-7]/cpufreq/scaling_cur_freq; }
echo "idle: $(watts) W"
for i in 0 1 2 3 4 5 6 7; do taskset -c $i "$B/avxburn" 8 >/dev/null & done
sleep 2; echo "8 cores AVX2: $(watts) W, $(mhz) MHz"; wait
for i in 0 1 2 3 4 5 6 7; do taskset -c $i "$B/avxburn" 8 >/dev/null & done
sleep 2; echo "8 cores AVX2 again: $(watts) W, $(mhz) MHz"; wait
taskset -c 3 "$B/avxburn" 8 >/dev/null &
sleep 2; echo "1 core AVX2: $(watts) W, $(mhz 3) MHz on that core"; wait
