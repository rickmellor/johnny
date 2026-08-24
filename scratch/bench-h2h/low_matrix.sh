#!/usr/bin/env bash
# Complete the quality matrix for effort-low-tp4 (seat already up + warmed).
set -u; B=$(cd "$(dirname "$0")" && pwd); L="$B/low-matrix.log"; P=effort-low-tp4
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
for suite in "humaneval" "arc --limit 200" "needle" "icl"; do
  tag=$(echo "$suite" | awk '{print $1}')
  log "=== $tag"
  timeout 7200 johnny bench "$P" --suite $suite > "$B/low-$tag.log" 2>&1
  log "$tag rc=$? :: $(grep -oE '✓ [a-z]+ .*' "$B/low-$tag.log" | tail -1 | cut -c1-110)"
done
log "=== ctxsafe (deep, capped 262144)"
timeout 10800 johnny bench "$P" --suite ctxsafe > "$B/low-ctxsafe.log" 2>&1
log "ctxsafe rc=$? :: $(grep -oE '✓ ctxsafe.*' "$B/low-ctxsafe.log" | tail -1 | cut -c1-110)"
log "LOW_MATRIX_DONE"
