#!/usr/bin/env bash
# llama.cpp-appropriate throughput/latency bench for johnny induction.
# Unlike the vLLM bench.sh (concurrency 16..1024), local GGUF seats are far slower and
# have few parallel slots, so we sweep a modest range and short outputs. Output format
# mirrors bench.sh ("<n> tok/s" lines + a "Single-request" section) so _parse_bench works.
#
# Usage: bench_llamacpp.sh <port> <model>
set -uo pipefail
PORT="${1:?port}"; MODEL="${2:?model}"
URL="http://127.0.0.1:${PORT}/v1/completions"
MAXTOK="${JOHNNY_BENCH_MAXTOK:-96}"
LEVELS="${JOHNNY_BENCH_LEVELS:-1 4 8 16 32}"
PROMPT="Write a short paragraph about distributed systems and consensus."

# completion_tokens summed from response bodies in a temp dir (accurate tok/s).
_sum_tokens() { # dir -> total completion_tokens
  local d="$1" tot=0 n
  for f in "$d"/*.json; do
    [ -f "$f" ] || continue
    n=$(grep -oE '"completion_tokens"[: ]+[0-9]+' "$f" | grep -oE '[0-9]+' | tail -1)
    tot=$((tot + ${n:-0}))
  done
  echo "$tot"
}

echo "== warmup =="
for i in 1 2; do
  curl -s "$URL" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":16}" >/dev/null
done

echo "== concurrency sweep =="
for n in $LEVELS; do
  d=$(mktemp -d)
  t0=$(date +%s.%N)
  for i in $(seq 1 "$n"); do
    curl -s "$URL" -H 'Content-Type: application/json' \
      -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT} (${i})\",\"max_tokens\":${MAXTOK}}" \
      -o "$d/$i.json" &
  done
  wait
  t1=$(date +%s.%N)
  toks=$(_sum_tokens "$d")
  el=$(awk "BEGIN{print ($t1-$t0)}")
  ts=$(awk "BEGIN{if($el>0) printf \"%.1f\", $toks/$el; else print 0}")
  echo "concurrency=${n}: ${toks} tokens in ${el}s => ${ts} tok/s"
  rm -rf "$d"
done

echo "== Single-request latency =="
d=$(mktemp -d)
t0=$(date +%s.%N)
curl -s "$URL" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAXTOK}}" -o "$d/1.json"
t1=$(date +%s.%N)
toks=$(_sum_tokens "$d")
el=$(awk "BEGIN{print ($t1-$t0)}")
ts=$(awk "BEGIN{if($el>0) printf \"%.1f\", $toks/$el; else print 0}")
echo "Single-request: ${toks} tokens in ${el}s => ${ts} tok/s"
rm -rf "$d"
