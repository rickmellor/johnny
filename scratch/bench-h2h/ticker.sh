#!/usr/bin/env bash
# Persistent 10-min status ticker → appends a one-line snapshot to ticker.log
B=$(cd "$(dirname "$0")" && pwd)
while true; do
  ST=$(grep -oE "=== [A-Za-z0-9 ]+" "$B/matrix-all.log" 2>/dev/null | tail -1)
  LAST=$(grep -E "rc=" "$B/matrix-all.log" 2>/dev/null | tail -1 | sed 's/.*:: //' | cut -c1-70)
  CUR=""
  for f in "$B"/m-*.log "$B"/gtp4-*.log; do
    [ -f "$f" ] || continue
    [ "$(( $(date +%s) - $(stat -c %Y "$f") ))" -lt 600 ] || continue
    n=$(basename "$f" .log); p=$(tr '\r' '\n' < "$f" | grep -oE "[0-9]+/(30|164|200|16)" | tail -1)
    r=$(tr '\r' '\n' < "$f" | grep -oE "reward=[0-9.]+, pass_rate=[0-9.]+" | tail -1)
    [ -n "$p$r" ] && CUR="$CUR $n=${p} ${r}"
  done
  echo "[$(date +%H:%M)] stage:${ST:-?} last:${LAST:-—} active:${CUR:-—}" >> "$B/ticker.log"
  grep -q GEMMA_TP4_DONE "$B/gemma-tp4.log" 2>/dev/null && { echo "[$(date +%H:%M)] ALL_DONE" >> "$B/ticker.log"; break; }
  sleep 600
done
