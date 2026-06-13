#!/usr/bin/bash
# vLLM tuning benchmark — runs against a specified port/model
# Usage: ./bench.sh <port> <model-name>
#
# Sweeps concurrency 16..1024 (high enough to find saturation knee on small models).
# Watch the curve: if TPS keeps climbing, you have batched-token headroom.

PORT=${1:-9000}
MODEL=${2:-tuning-model}
BASE_URL="http://localhost:${PORT}"

PROMPTS=(
  "The Ferrari F355 is"
  "The history of aviation began"
  "Quantum mechanics describes"
  "In machine learning, transformers"
  "The chemistry of catalysts"
  "Architectural design principles"
  "The geology of volcanic islands"
  "Embedded systems engineering"
)

echo "=== Warmup (4 requests) ==="
for i in {1..4}; do
  curl -s "${BASE_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPTS[0]}\",\"max_tokens\":100}" \
    -o /dev/null
done
echo "Done."

echo ""
echo "=== Concurrency sweep (repeated prompt, prefix cache friendly) ==="
for n in 16 32 64 128 256 512 1024; do
  echo -n "Concurrency ${n}: "
  T=$(date +%s.%N)
  for i in $(seq 1 $n); do
    curl -s "${BASE_URL}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPTS[0]}\",\"max_tokens\":100}" \
      -o /dev/null &
  done
  wait
  T2=$(date +%s.%N)
  ELAPSED=$(echo "$T2 - $T" | bc)
  TPS=$(echo "scale=1; $n * 100 / $ELAPSED" | bc)
  echo "${ELAPSED}s, ${TPS} tok/s"
done

echo ""
echo "=== Concurrency sweep (varied prompts, no prefix cache) ==="
for n in 16 32 64 128 256 512 1024; do
  echo -n "Concurrency ${n}: "
  T=$(date +%s.%N)
  for i in $(seq 1 $n); do
    P="${PROMPTS[$((i % 8))]} ${i}"
    curl -s "${BASE_URL}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL}\",\"prompt\":\"${P}\",\"max_tokens\":100}" \
      -o /dev/null &
  done
  wait
  T2=$(date +%s.%N)
  ELAPSED=$(echo "$T2 - $T" | bc)
  TPS=$(echo "scale=1; $n * 100 / $ELAPSED" | bc)
  echo "${ELAPSED}s, ${TPS} tok/s"
done

echo ""
echo "=== Single-request latency ==="
for i in {1..3}; do
  echo -n "Run ${i}: "
  T=$(date +%s.%N)
  RESULT=$(curl -s "${BASE_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPTS[$((i % 8))]}\",\"max_tokens\":100}")
  ACTUAL_TOK=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo "100")
  T2=$(date +%s.%N)
  ELAPSED=$(echo "$T2 - $T" | bc)
  TPS=$(echo "scale=1; ${ACTUAL_TOK} / $ELAPSED" | bc)
  echo "${ELAPSED}s (${ACTUAL_TOK} tokens), ${TPS} tok/s"
done

echo ""
echo "=== Long output (2048 tokens, concurrency 1) ==="
T=$(date +%s.%N)
RESULT=$(curl -s "${BASE_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPTS[0]}\",\"max_tokens\":2048}")
ACTUAL_TOK=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo "2048")
T2=$(date +%s.%N)
ELAPSED=$(echo "$T2 - $T" | bc)
TPS=$(echo "scale=1; ${ACTUAL_TOK} / $ELAPSED" | bc)
echo "${ELAPSED}s (${ACTUAL_TOK} tokens), ${TPS} tok/s"

echo ""
echo "=== GPU stats ==="
docker exec vllm-tuning amd-smi monitor -ptumv 2>/dev/null | head -10 || echo "(could not read GPU stats from container)"

echo ""
echo "=== Benchmark complete ==="
