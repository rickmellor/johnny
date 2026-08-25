#!/usr/bin/env bash
# After chain3: PlanBench across the fleet — gemma TP4 first (Rick's ask), then the comparators.
set -u; B=$(cd "$(dirname "$0")" && pwd); L="$B/chain4.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
while ! grep -q "CHAIN3_DONE" "$B/chain3.log" 2>/dev/null; do sleep 120; done
log "chain3 done — PlanBench sweep"
run(){ MODEL="$1"; PID="$2"; PORT="$3"; TAG="$4"
  for c in $(docker ps --format '{{.Names}}' | grep -E 'Qwen3.8|gemma-4-26B|coder'); do johnny down "$c" >>"$L" 2>&1; done; sleep 5
  log "=== $TAG: up $PID"
  timeout 2400 johnny up "$MODEL" --placement "$PID" --port "$PORT" --wait >>"$L" 2>&1
  C=$(docker ps --format '{{.Names}}' | grep -E "$PORT\$" | head -1)
  for i in $(seq 1 110); do docker logs "$C" 2>&1 | grep -q "Application startup complete" && break; sleep 10; done
  timeout 7200 johnny bench "$PID" --suite planbench --limit 100 --concurrency 4 --yes > "$B/pb-$TAG.log" 2>&1
  log "$TAG rc=$? :: $(grep -oE '✓ planbench.*' "$B/pb-$TAG.log" | tail -1 | cut -c1-120)"
}
run gemma-4-26B-A4B-it-FP8-Dynamic gemma-tp4-c4-mml262144-v0202 8002 gemma-tp4
run qwen-27b-coder induct-tp2-gmu0.92-seqs64-bt16384-mml95417 8003 coder
run Qwen3.8-27B-FP8 effort-medium-tp4 8003 qwen38-med
run Qwen3.8-27B-FP8 effort-low-tp4 8003 qwen38-low
for c in $(docker ps --format '{{.Names}}' | grep -E 'Qwen3.8|gemma-4-26B|coder'); do johnny down "$c" >>"$L" 2>&1; done; sleep 5
timeout 2400 johnny profile up daily >>"$L" 2>&1
log "CHAIN4_DONE"
