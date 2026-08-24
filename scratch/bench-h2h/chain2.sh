#!/usr/bin/env bash
# Resequenced: 1 (finish xhigh AB) -> 5 (gemma TP4 262K) -> 3 (ctxsafe) -> 4 (TP2 pair) -> 2 (Qwen3.8 medium quality)
set -u; B=$(cd "$(dirname "$0")" && pwd); L="$B/chain2.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
res(){ grep -oE "✓ [a-z]+ .*" "$1" 2>/dev/null | tail -1 | cut -c1-115; }
downall(){ for c in $(docker ps --format '{{.Names}}' | grep -E 'Qwen3.8|gemma-4-26B|coder'); do johnny down "$c" >>"$L" 2>&1; done; sleep 5; }

# ---------- STAGE 1 (finish): wait out the running xhigh AutomationBench ----------
log "STAGE 1: waiting for xhigh AutomationBench to finish"
while pgrep -f "bin/johnny bench" >/dev/null; do sleep 60; done
log "STAGE 1 DONE :: $(res "$B/m-xhigh4-ab.log")"

# ---------- STAGE 5: gemma TP4 @ 262K, two images ----------
gtry(){ PID="$1"; TAG="$2"; MML=262144
  downall; log "STAGE 5/$TAG: up $PID"
  timeout 2400 johnny up gemma-4-26B-A4B-it-FP8-Dynamic --placement "$PID" --port 8002 --wait >>"$L" 2>&1
  C=johnny-gemma-4-26B-A4B-it-FP8-Dynamic-8002
  for i in $(seq 1 110); do
    LG=$(docker logs "$C" 2>&1 | grep -v "not documented")
    echo "$LG" | grep -q "Application startup complete" && { log "  $TAG READY"; break; }
    echo "$LG" | grep -qiE "Traceback|RuntimeError|hipError|Engine core initialization failed|ValueError|AssertionError" && { log "  $TAG ENGINE ERROR"; break; }
    docker ps --format '{{.Names}}' | grep -q "^$C$" || { log "  $TAG CONTAINER GONE"; break; }
    sleep 10; done
  KV=$(docker logs "$C" 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens" | tail -1)
  CONC=$(docker logs "$C" 2>&1 | grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" | tail -1)
  log "  $TAG KV :: ${KV:-none} | ${CONC:-—}"
  docker ps --format '{{.Names}}' | grep -q gemma-4-26B || { log "  $TAG failed to serve — skipping probes"; return 0; }
  log "  $TAG deep needle (single ladder + 4-way @ max depth)"
  timeout 9000 python3 ~/repos/johnny/scratch/kvexp/deepprobe.py http://127.0.0.1:8002 gemma-4-26B-A4B-it-FP8-Dynamic "$MML" 4 > "$B/gtp4-$TAG-deep.log" 2>&1
  log "  $TAG deep :: $(tail -1 "$B/gtp4-$TAG-deep.log" | cut -c1-140)"
  timeout 1800 python3 ~/repos/johnny/scratch/kvexp/perfprobe.py http://127.0.0.1:8002 gemma-4-26B-A4B-it-FP8-Dynamic 4 > "$B/gtp4-$TAG-perf.log" 2>&1
  log "  $TAG perf :: $(tail -1 "$B/gtp4-$TAG-perf.log")"; }
gtry gemma-tp4-c4-mml262144-v0202 v0202
gtry gemma-tp4-c4-mml262144-nightly nightly
log "STAGE 5 DONE"

# ---------- STAGE 3: ctxsafe (own disposable seat, all 4 GPUs) ----------
downall; log "STAGE 3: ctxsafe on effort-low-tp4 (effort-independent)"
timeout 14400 johnny bench effort-low-tp4 --suite ctxsafe --yes > "$B/m-ctxsafe.log" 2>&1
log "STAGE 3 DONE rc=$? :: $(res "$B/m-ctxsafe.log")"

# ---------- STAGE 4: TP2 pair concurrently ----------
downall; log "STAGE 4: gemma + coder (TP2, concurrent)"
timeout 2400 johnny up gemma-4-26B-A4B-it-FP8-Dynamic --placement induct-tp2-gmu0.92-seqs32-bt16384-mml110832 --port 8002 --wait >>"$L" 2>&1
timeout 2400 johnny up qwen-27b-coder --placement induct-tp2-gmu0.92-seqs64-bt16384-mml95417 --port 8003 --wait >>"$L" 2>&1
( timeout 14400 johnny bench induct-tp2-gmu0.92-seqs32-bt16384-mml110832 --suite automationbench --limit 30 --concurrency 4 --yes > "$B/m-gemma-ab.log" 2>&1 ) &
( timeout 14400 johnny bench induct-tp2-gmu0.92-seqs64-bt16384-mml95417 --suite automationbench --limit 30 --concurrency 4 --yes > "$B/m-coder-ab.log" 2>&1
  timeout 1200 johnny bench induct-tp2-gmu0.92-seqs64-bt16384-mml95417 --suite icl --yes > "$B/m-coder-icl.log" 2>&1 ) &
wait
log "STAGE 4 DONE :: gemma-ab $(res "$B/m-gemma-ab.log") | coder-ab $(res "$B/m-coder-ab.log") | coder-icl $(res "$B/m-coder-icl.log")"

# ---------- STAGE 2: Qwen3.8 TP4 medium quality ----------
downall; log "STAGE 2: Qwen3.8 TP4 medium quality suites"
timeout 2400 johnny up Qwen3.8-27B-FP8 --placement effort-medium-tp4 --port 8003 --wait >>"$L" 2>&1
for s in humaneval "arc --limit 200" needle icl; do
  t=$(echo "$s" | awk '{print $1}')
  timeout 7200 johnny bench effort-medium-tp4 --suite $s --yes > "$B/m-med4-$t.log" 2>&1
  log "  med4/$t rc=$? :: $(res "$B/m-med4-$t.log")"
done
log "STAGE 2 DONE"
downall; log "restoring daily"; timeout 2400 johnny profile up daily >>"$L" 2>&1
log "CHAIN2_ALL_DONE"
