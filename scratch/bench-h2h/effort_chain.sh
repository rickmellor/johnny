#!/usr/bin/env bash
# reasoning_effort matrix: {low,medium} x {TP2 conc2, TP4 conc4}, AutomationBench --limit 30, sequential.
set -u; B=$(cd "$(dirname "$0")" && pwd); L="$B/effort-chain.log"; log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
run_one(){ PID="$1"; CONC="$2"; TAG="$3"; PORT="$4"
  log "=== $TAG: up $PID (auto-warm)"
  timeout 1800 johnny up Qwen3.8-27B-FP8 --placement "$PID" --port "$PORT" --wait >>"$L" 2>&1
  grep -q "GDN warm-up" "$L" && log "warm-up ran"
  log "$TAG: automationbench --limit 30 --concurrency $CONC"
  timeout 10800 johnny bench "$PID" --suite automationbench --limit 30 --concurrency "$CONC" > "$B/ab-$TAG.log" 2>&1
  log "$TAG rc=$? $(tr '\r' '\n' < "$B/ab-$TAG.log" | grep -oE '[0-9]+/30 .*(reward|pass_rate)=[0-9.]+.*' | tail -1 | cut -c1-80)"
  johnny down "johnny-Qwen3.8-27B-FP8-$PORT" >>"$L" 2>&1
}
# --- TP2 runs on 2 GPUs: displace gemma (SAINT chat falls back to cloud meanwhile)
log "downing gemma seat for TP2 runs"; johnny down johnny-gemma-4-26B-A4B-it-FP8-Dynamic-8002 >>"$L" 2>&1
run_one effort-low-tp2 2 tp2-low 8004
run_one effort-medium-tp2 2 tp2-med 8004
log "restoring gemma"; timeout 1200 johnny up gemma-4-26B-A4B-it-FP8-Dynamic --placement induct-tp2-gmu0.92-seqs32-bt16384-mml110832 --port 8002 --wait >>"$L" 2>&1
# --- TP4 runs need all 4 GPUs
log "downing gemma+coder for TP4 runs"; johnny down johnny-gemma-4-26B-A4B-it-FP8-Dynamic-8002 >>"$L" 2>&1; johnny down johnny-qwen-27b-coder-8003 >>"$L" 2>&1
run_one effort-low-tp4 4 tp4-low 8004
run_one effort-medium-tp4 4 tp4-med 8004
log "restoring daily profile"; timeout 1800 johnny profile up daily >>"$L" 2>&1
log "EFFORT_CHAIN_DONE"
