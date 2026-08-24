#!/usr/bin/env bash
# Complete effort/model matrix. TP4 work first (needs all 4 GPUs), then the TP2 pair concurrently.
set -u; B=$(cd "$(dirname "$0")" && pwd); L="$B/matrix-all.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
res(){ grep -oE "✓ [a-z]+ .*" "$1" 2>/dev/null | tail -1 | cut -c1-110; }

quality(){ # $1=placement $2=tag  — quick suites against a running seat
  for s in humaneval "arc --limit 200" needle icl; do
    t=$(echo "$s" | awk '{print $1}')
    log "  $2/$t"; timeout 7200 johnny bench "$1" --suite $s --yes > "$B/m-$2-$t.log" 2>&1
    log "  $2/$t rc=$? :: $(res "$B/m-$2-$t.log")"
  done
}
ab(){ # $1=placement $2=tag $3=conc
  log "  $2/automationbench (limit 30, conc $3)"
  timeout 14400 johnny bench "$1" --suite automationbench --limit 30 --concurrency "$3" --yes > "$B/m-$2-ab.log" 2>&1
  log "  $2/ab rc=$? :: $(res "$B/m-$2-ab.log")"
}
up4(){ johnny down "$(docker ps --format '{{.Names}}' | grep -E 'Qwen3.8|gemma-4-26B|coder' | head -1)" >>"$L" 2>&1 || true
       for c in $(docker ps --format '{{.Names}}' | grep -E 'Qwen3.8|gemma-4-26B|coder'); do johnny down "$c" >>"$L" 2>&1; done
       sleep 5; log "up $1"; timeout 1800 johnny up Qwen3.8-27B-FP8 --placement "$1" --port 8003 --wait >>"$L" 2>&1; }

# ---------- TP4: xhigh (full) ----------
log "=== TP4 xhigh"; up4 effort-xhigh-tp4; quality effort-xhigh-tp4 xhigh4; ab effort-xhigh-tp4 xhigh4 4
# ---------- TP4: medium (quality; AB already measured 30.0%) ----------
log "=== TP4 medium"; up4 effort-medium-tp4; quality effort-medium-tp4 med4
# ---------- ctxsafe once (effort-independent: capacity, not generation) ----------
log "=== ctxsafe (effort-low-tp4, applies to all efforts — same knobs)"
for c in $(docker ps --format '{{.Names}}' | grep -E 'Qwen3.8|gemma-4-26B|coder'); do johnny down "$c" >>"$L" 2>&1; done; sleep 5
timeout 14400 johnny bench effort-low-tp4 --suite ctxsafe --yes > "$B/m-ctxsafe.log" 2>&1
log "ctxsafe rc=$? :: $(res "$B/m-ctxsafe.log")"
# ---------- TP2 pair: gemma AB + coder AB/ICL, concurrently ----------
log "=== TP2 pair: gemma + coder"
timeout 1800 johnny up gemma-4-26B-A4B-it-FP8-Dynamic --placement induct-tp2-gmu0.92-seqs32-bt16384-mml110832 --port 8002 --wait >>"$L" 2>&1
timeout 1800 johnny up qwen-27b-coder --placement induct-tp2-gmu0.92-seqs64-bt16384-mml95417 --port 8003 --wait >>"$L" 2>&1
( timeout 14400 johnny bench induct-tp2-gmu0.92-seqs32-bt16384-mml110832 --suite automationbench --limit 30 --concurrency 4 --yes > "$B/m-gemma-ab.log" 2>&1; echo "gemma-ab rc=$?" >> "$L" ) &
( timeout 14400 johnny bench induct-tp2-gmu0.92-seqs64-bt16384-mml95417 --suite automationbench --limit 30 --concurrency 4 --yes > "$B/m-coder-ab.log" 2>&1; echo "coder-ab rc=$?" >> "$L";
  timeout 1200 johnny bench induct-tp2-gmu0.92-seqs64-bt16384-mml95417 --suite icl --yes > "$B/m-coder-icl.log" 2>&1 ) &
wait
log "gemma-ab :: $(res "$B/m-gemma-ab.log")"; log "coder-ab :: $(res "$B/m-coder-ab.log")"; log "coder-icl :: $(res "$B/m-coder-icl.log")"
log "MATRIX_ALL_DONE"
