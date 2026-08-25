#!/usr/bin/env bash
# Test Rick's hypothesis: does an explicit roadmap rescue gemma's short-horizon planning failures?
set -u; B=$(cd "$(dirname "$0")" && pwd); L="$B/chain6.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
while ! grep -q "CHAIN5_DONE" "$B/chain5.log" 2>/dev/null; do sleep 60; done
log "chain5 done — roadmap A/B on gemma (profile gemma-tp4 already up)"
for i in $(seq 1 60); do curl -s --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null 2>&1 && break; sleep 10; done
for pass in 1 2 3; do
  timeout 3600 python3 "$B/shorthorizon.py" http://127.0.0.1:8002 gemma-4-26B-A4B-it-FP8-Dynamic "gemma-roadmap-$pass" --roadmap > "$B/sh-gemma-roadmap-$pass.log" 2>&1
  log "roadmap pass $pass :: $(tail -1 "$B/sh-gemma-roadmap-$pass.log" | cut -c1-150)"
  timeout 3600 python3 "$B/shorthorizon.py" http://127.0.0.1:8002 gemma-4-26B-A4B-it-FP8-Dynamic "gemma-bare-$pass" > "$B/sh-gemma-bare-$pass.log" 2>&1
  log "bare    pass $pass :: $(tail -1 "$B/sh-gemma-bare-$pass.log" | cut -c1-150)"
done
log "CHAIN6_DONE"
