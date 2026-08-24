#!/usr/bin/env bash
# After chain2: re-run gemma AutomationBench on the TP4 262K seat to test the context-truncation hypothesis.
set -u; B=$(cd "$(dirname "$0")" && pwd); L="$B/chain3.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
while ! grep -q "CHAIN2_ALL_DONE" "$B/chain2.log" 2>/dev/null; do sleep 120; done
log "chain2 done — gemma TP4 262K AutomationBench (truncation test)"
for c in $(docker ps --format '{{.Names}}' | grep -E 'Qwen3.8|gemma-4-26B|coder'); do johnny down "$c" >>"$L" 2>&1; done; sleep 5
timeout 2400 johnny up gemma-4-26B-A4B-it-FP8-Dynamic --placement gemma-tp4-c4-mml262144-v0202 --port 8002 --wait >>"$L" 2>&1
C=johnny-gemma-4-26B-A4B-it-FP8-Dynamic-8002
for i in $(seq 1 110); do docker logs "$C" 2>&1 | grep -q "Application startup complete" && break; sleep 10; done
log "seat: $(docker logs "$C" 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | tail -1)"
timeout 14400 johnny bench gemma-tp4-c4-mml262144-v0202 --suite automationbench --limit 30 --concurrency 4 --yes > "$B/m-gemma-tp4-ab.log" 2>&1
log "gemma TP4 AB rc=$? :: $(grep -oE '✓ automationbench.*' "$B/m-gemma-tp4-ab.log" | tail -1 | cut -c1-120)"
for c in $(docker ps --format '{{.Names}}' | grep -E 'Qwen3.8|gemma-4-26B|coder'); do johnny down "$c" >>"$L" 2>&1; done; sleep 5
timeout 2400 johnny profile up daily >>"$L" 2>&1
log "CHAIN3_DONE"
