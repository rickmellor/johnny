#!/usr/bin/env bash
# Queued after the effort matrix: validate gemma-4-26B TP4 at full 262K native context, 0.20.2 vs nightly.
set -u; B=$(cd "$(dirname "$0")" && pwd); L="$B/gemma-tp4.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
while ! grep -q "MATRIX_ALL_DONE" "$B/matrix-all.log" 2>/dev/null; do sleep 120; done
log "effort matrix done — starting gemma TP4 max-context validation"
try(){ PID="$1"; TAG="$2"
  for c in $(docker ps --format '{{.Names}}' | grep -E 'Qwen3.8|gemma-4-26B|coder'); do johnny down "$c" >>"$L" 2>&1; done; sleep 5
  log "=== $TAG: up $PID"
  MML=262144
  for attempt in 1 2; do
    timeout 2400 johnny up gemma-4-26B-A4B-it-FP8-Dynamic --placement "$PID" --port 8002 --wait >>"$L" 2>&1
    C=johnny-gemma-4-26B-A4B-it-FP8-Dynamic-8002
    for i in $(seq 1 100); do
      LG=$(docker logs "$C" 2>&1 | grep -v "not documented")
      echo "$LG" | grep -q "Application startup complete" && { log "$TAG READY"; break; }
      echo "$LG" | grep -qiE "Traceback|RuntimeError|hipError|Engine core initialization failed|ValueError|AssertionError" && { log "$TAG ERROR"; break; }
      docker ps --format '{{.Names}}' | grep -q "^$C$" || { log "$TAG GONE"; break; }
      sleep 10
    done
    KV=$(docker logs "$C" 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens" | tail -1)
    CONC=$(docker logs "$C" 2>&1 | grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" | tail -1)
    log "$TAG :: $KV | $CONC"
    if [ -n "$KV" ]; then break; fi
    N=$(docker logs "$C" 2>&1 | grep -oE "stored in KV cache \(([0-9]+)\)" | grep -oE "[0-9]+" | tail -1)
    if [ -n "$N" ] && [ "$attempt" = 1 ]; then MML=$(( (N-2048)/1024*1024 )); log "$TAG: KV too small -> retry mml=$MML"
      (cd ~/repos/johnny && uv run --quiet python "$B/../kvexp/mkplacement.py" gemma-4-26B-A4B-it-FP8-Dynamic "$PID" "$PID" "mml=$MML" >>"$L" 2>&1); else break; fi
  done
  docker ps --format '{{.Names}}' | grep -q gemma-4-26B || return 0
  log "$TAG: deep needle probe (single + 4-way at max depth)"
  timeout 7200 python3 ~/repos/johnny/scratch/kvexp/deepprobe.py http://127.0.0.1:8002 gemma-4-26B-A4B-it-FP8-Dynamic "$MML" 4 > "$B/gtp4-$TAG-deep.log" 2>&1
  log "$TAG deep :: $(tail -1 "$B/gtp4-$TAG-deep.log" | cut -c1-140)"
  log "$TAG: perf"; timeout 1800 python3 ~/repos/johnny/scratch/kvexp/perfprobe.py http://127.0.0.1:8002 gemma-4-26B-A4B-it-FP8-Dynamic 4 > "$B/gtp4-$TAG-perf.log" 2>&1
  log "$TAG perf :: $(cat "$B/gtp4-$TAG-perf.log" | tail -1)"
}
try gemma-tp4-c4-mml262144-v0202 v0202
try gemma-tp4-c4-mml262144-nightly nightly
log "restoring daily"; for c in $(docker ps --format '{{.Names}}' | grep -E 'Qwen3.8|gemma-4-26B|coder'); do johnny down "$c" >>"$L" 2>&1; done
timeout 2400 johnny profile up daily >>"$L" 2>&1
log "GEMMA_TP4_DONE"
