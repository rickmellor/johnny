#!/usr/bin/bash
# wait.sh <name> <port> [timeout_s] — block until /health is 200 or the container dies
N=$1 P=$2 T=${3:-900}; s=$(date +%s)
while :; do
  curl -sf -o /dev/null localhost:$P/health && { echo "$N ready in $(( $(date +%s)-s ))s"; exit 0; }
  [ "$(docker inspect -f '{{.State.Running}}' $N 2>/dev/null)" = true ] || { echo "$N DIED"; docker logs $N 2>&1 | tail -25; exit 1; }
  [ $(( $(date +%s)-s )) -gt $T ] && { echo "$N timeout"; exit 2; }
  sleep 5
done
