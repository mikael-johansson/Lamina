#!/bin/bash
# Start/stop the Strata fork server in the background.
#   ./strata.sh start [config]   (default configs/fork-iq3_s-tuned.json)
#   ./strata.sh stop | status | log
cd "$(dirname "$0")"
cfg=${2:-configs/fork-iq3_s-tuned.json}
log=strata-$(basename "$cfg" .json).server.log
case "$1" in
  start)
    pgrep -f '[s]erve/server.py --engine strata' >/dev/null && { echo "already running"; exit 1; }
    (cd eddoursul && exec setsid nohup .venv/bin/python serve/server.py --engine strata --config "../$cfg") \
       > "$log" 2>&1 < /dev/null &
    echo "starting $cfg, log: $log"
    until grep -q "^ready:" "$log" 2>/dev/null; do
      pgrep -f '[s]erve/server.py --engine strata' >/dev/null || { echo "failed:"; tail -5 "$log"; exit 1; }
      sleep 2
    done
    grep -A1 "^ready:" "$log" | tail -1 ;;
  stop)
    pkill -f '[s]erve/server.py --engine strata' && echo stopping
    while pgrep -f '[b]uild/strata --serve' >/dev/null; do sleep 1; done; echo stopped ;;
  status) pgrep -af '[s]erve/server.py --engine strata' | cut -c1-160 || echo "not running" ;;
  log) tail -f strata-*.server.log ;;
  *) echo "usage: $0 start [config] | stop | status | log" ;;
esac
