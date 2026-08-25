#!/usr/bin/env bash
# Short-horizon delegate probe: gemma TP4 vs coder vs Qwen3.8-medium (8-step budget, real fs+pytest sandbox)
set -u; B=$(cd "$(dirname "$0")" && pwd); L="$B/chain5.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
run(){ MODEL="$1"; PID="$2"; PORT="$3"; TAG="$4"
  for c in $(docker ps --format '{{.Names}}' | grep -E 'Qwen3.8|gemma-4-26B|coder'); do johnny down "$c" >>"$L" 2>&1; done; sleep 5
  log "=== $TAG: up $PID"
  timeout 2400 johnny up "$MODEL" --placement "$PID" --port "$PORT" --wait >>"$L" 2>&1
  C=$(docker ps --format '{{.Names}}' | grep -E "$PORT\$" | head -1)
  for i in $(seq 1 110); do docker logs "$C" 2>&1 | grep -q "Application startup complete" && break; sleep 10; done
  timeout 3600 python3 "$B/shorthorizon.py" "http://127.0.0.1:$PORT" "$MODEL" "$TAG" > "$B/sh-$TAG.log" 2>&1
  log "$TAG :: $(tail -1 "$B/sh-$TAG.log" | cut -c1-160)"
}
run gemma-4-26B-A4B-it-FP8-Dynamic gemma-tp4-c4-mml262144-v0202 8002 gemma-tp4
run qwen-27b-coder induct-tp2-gmu0.92-seqs64-bt16384-mml95417 8003 coder
run Qwen3.8-27B-FP8 effort-medium-tp4 8003 qwen38-med
for c in $(docker ps --format '{{.Names}}' | grep -E 'Qwen3.8|gemma-4-26B|coder'); do johnny down "$c" >>"$L" 2>&1; done; sleep 5
timeout 2400 johnny profile up gemma-tp4 >>"$L" 2>&1
log "CHAIN5_DONE — fleet left on the new gemma-tp4 profile"
