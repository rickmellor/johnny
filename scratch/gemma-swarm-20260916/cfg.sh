#!/usr/bin/bash
# cfg.sh <tag> <conc> <image> <tp> [vllm args...] — launch on HIP 4,5, bench prefill-only + mixed 2800/250, save log
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
TAG=$1 C=$2 IMG=$3 TP=$4; shift 4
echo "### $TAG"
./launch.sh exp-p2 8104 ${GPUS:-4,5} $IMG $TP --limit-mm-per-prompt '{"image":0}' --no-enable-prefix-caching "$@" >/dev/null
if ./wait.sh exp-p2 8104 1200 | tail -4; then
  docker logs exp-p2 2>&1 | grep -E "GPU KV cache size" | head -1 | sed 's/.*GPU KV/KV/'
  NP=$((C*2)) ./bench.sh $TAG $C 2800 1 exp-p2 | tee -a results/summary.txt
  ./bench.sh $TAG $C 2800 250 exp-p2 | tee -a results/summary.txt
fi
docker logs exp-p2 > logs/$TAG.log 2>&1; docker rm -f exp-p2 >/dev/null 2>&1
