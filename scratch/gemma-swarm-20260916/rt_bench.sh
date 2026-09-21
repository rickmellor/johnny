#!/usr/bin/bash
# rt_bench.sh <tag> <seat-index> <port> <served-model> <conc> [nprompts] — real-text 2789-in/250-out bench from a client container, with E2E latency
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
TAG=$1 I=$2 PORT=$3 SERVED=$4 C=$5 NP=${6:-288}; mkdir -p results/$TAG
docker run --rm --network host --entrypoint vllm -v /mnt/data/models:/models:ro -v $PWD/dataset:/data:ro vllm/vllm-openai-rocm:v0.28.0 \
  bench serve --backend openai --base-url http://127.0.0.1:$PORT --model $SERVED --tokenizer /models/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic \
  --dataset-name custom --dataset-path /data/realtext-$I.jsonl --skip-chat-template --custom-output-len 250 --ignore-eos \
  --num-prompts $NP --max-concurrency $C --seed $RANDOM --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 > results/$TAG/seat$I-c$C.txt 2>&1
python3 - results/$TAG/seat$I-c$C.txt "$TAG" $I $C <<'PY'
import re,sys
t=open(sys.argv[1]).read(); g=lambda k:(re.search(re.escape(k)+r"\s*:?\s+([\d.]+)",t) or [0,'nan'])[1]
print(f"{sys.argv[2]:34s} seat{sys.argv[3]} c={sys.argv[4]:>3} | {g('Request throughput (req/s)'):>5} req/s | total {g('Total token throughput (tok/s)'):>8} | out {g('Output token throughput (tok/s)'):>7} tok/s | TTFT p50 {g('P50 TTFT (ms)'):>7} | TPOT p50 {g('P50 TPOT (ms)'):>6} | E2E p50 {g('P50 E2EL (ms)'):>8} p90 {g('P90 E2EL (ms)'):>8} p99 {g('P99 E2EL (ms)'):>8} ms | fail {g('Failed requests')} | accept-len {g('Acceptance length')}")
PY
