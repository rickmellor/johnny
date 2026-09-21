#!/usr/bin/bash
# NAME=exp-x PORT=81xx GPUS=a,b CONCS="32 64" cfg2.sh <tag> <image> <tp> [vllm args...]
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
TAG=$1 IMG=$2 TP=$3; shift 3
echo "### $TAG"
./launch.sh $NAME $PORT $GPUS $IMG $TP --limit-mm-per-prompt '{"image":0}' --no-enable-prefix-caching "$@" >/dev/null
./wait.sh $NAME $PORT 1200 | tail -4
if curl -sf -o /dev/null localhost:$PORT/health; then
  docker logs $NAME 2>&1 | grep -E "GPU KV cache size" | head -1 | sed 's/.*GPU KV/KV/'
  for C in $CONCS; do ./bench.sh $TAG $C 2800 250 $NAME | tee -a results/summary.txt; done
fi
docker logs $NAME > logs/$TAG.log 2>&1; docker rm -f $NAME >/dev/null 2>&1
