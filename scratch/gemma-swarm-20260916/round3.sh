#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
for bt in 16384 4096; do
  BT=$bt ./launch.sh exp-p2 8104 4,5 vllm/vllm-openai-rocm:v0.20.2 2 --limit-mm-per-prompt '{"image":0}' --no-enable-prefix-caching >/dev/null
  ./wait.sh exp-p2 8104 | tail -2
  docker logs exp-p2 2>&1 | grep -E "GPU KV cache size" | cut -c60-
  NP=64 ./bench.sh tp2-v0202-bf16-bt$bt 32 2800 1 exp-p2 | tee -a results/summary.txt     # prefill-only ceiling
  for c in 24 32; do ./bench.sh tp2-v0202-bf16-bt$bt $c 2800 250 exp-p2 | tee -a results/summary.txt; done
  docker logs exp-p2 > logs/tp2-v0202-bf16-bt$bt.log 2>&1
done
echo ROUND3 DONE
