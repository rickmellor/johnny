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

echo "=== Warmup (adaptive: sustained load until throughput plateaus) ==="
# A few quick requests measure COLD: on aggressive-idle GPUs (RDNA4 deep-idles between
# seats) the clock hasn't ramped to boost and CUDA graphs/kernels aren't warm, so a
# cold-launched seat benches far slower than one launched right after another — which made
# the *second* point of every (tp) group look ~3x faster purely from run order.
#
# Rather than a fixed, hardware-specific duration, we fire rounds of concurrent load and
# watch round-over-round throughput: while clocks ramp it keeps rising; once it stops
# improving by > TOL for STABLE_ROUNDS rounds we're at steady state and stop. Self-calibrates
# across GPUs. Bounded by MIN/MAX seconds. All knobs are env-overridable.
WARM_CONC=${WARMUP_CONCURRENCY:-24}
WARM_TOK=128
WARMUP_TOL=${WARMUP_TOL:-0.03}              # a round must beat best by >3% to count as "still ramping"
WARMUP_STABLE_ROUNDS=${WARMUP_STABLE_ROUNDS:-3}
WARMUP_MIN_SECONDS=${WARMUP_MIN_SECONDS:-4}
WARMUP_MAX_SECONDS=${WARMUP_MAX_SECONDS:-30}
warm_start=$(date +%s); best=0; stable=0; round=0
while : ; do
  round=$((round + 1))
  T=$(date +%s.%N)
  for i in $(seq 1 "$WARM_CONC"); do
    curl -s "${BASE_URL}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPTS[$((i % 8))]} warm ${round}\",\"max_tokens\":${WARM_TOK}}" \
      -o /dev/null &
  done
  wait
  T2=$(date +%s.%N)
  elapsed=$(echo "$T2 - $T" | bc)
  tps=$(echo "scale=3; $WARM_CONC * $WARM_TOK / $elapsed" | bc)
  total=$(( $(date +%s) - warm_start ))
  if [ "$(echo "$tps > $best * (1 + $WARMUP_TOL)" | bc -l)" -eq 1 ]; then best=$tps; stable=0; else stable=$((stable + 1)); fi
  printf "  round %d: %.0f tok/s (best %.0f · stable %d/%d · %ds)\n" "$round" "$tps" "$best" "$stable" "$WARMUP_STABLE_ROUNDS" "$total"
  if [ "$total" -ge "$WARMUP_MAX_SECONDS" ]; then echo "  (max ${WARMUP_MAX_SECONDS}s — using current)"; break; fi
  if [ "$stable" -ge "$WARMUP_STABLE_ROUNDS" ] && [ "$total" -ge "$WARMUP_MIN_SECONDS" ]; then echo "  (plateaued at steady state)"; break; fi
done
echo "Done (${total}s)."

echo ""
echo "=== Concurrency sweep (repeated prompt, prefix cache friendly) ==="
for n in ${BENCH_CONCURRENCY:-16 32 64 128 256 512 1024}; do
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
for n in ${BENCH_CONCURRENCY:-16 32 64 128 256 512 1024}; do
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
for i in {1..5}; do
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
