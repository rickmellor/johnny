#!/usr/bin/bash
# Poll a vLLM server's /v1/models endpoint until it returns 200 OK.
# Usage: ./wait-ready.sh <port> [max_minutes]
#   port:         port to poll (default 9000)
#   max_minutes:  give up after this many minutes (default 10)
# Exits 0 on ready, 1 on timeout.

PORT=${1:-9000}
MAX_MIN=${2:-10}
MAX_ITERS=$(( MAX_MIN * 2 ))   # 30s per iter

for i in $(seq 1 $MAX_ITERS); do
  sleep 30
  STATUS=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/v1/models")
  ELAPSED=$(( i * 30 ))
  echo "t=${ELAPSED}s status=${STATUS}"
  if [ "$STATUS" = "200" ]; then
    echo "ready at ${ELAPSED}s"
    exit 0
  fi
done

echo "TIMEOUT after ${MAX_MIN}m — server did not become ready"
echo "---"
echo "Last 20 lines of vllm-tuning logs:"
docker logs --tail 20 vllm-tuning 2>&1 | tail -20
exit 1
