#!/usr/bin/bash
# box3.sh <tag> "<BT blocks conc>" x3 — three DP2EP2 seats with per-seat settings, benched simultaneously (per-seat concurrency)
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
TAG=$1; shift; V=vllm/vllm-openai-rocm:v0.20.2
export EXTRA_ENV="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple"
echo "### $TAG"; G=(0,1 2,3 4,5); i=0
for spec in "$@"; do set -- $spec
  BT=$1 ./launch.sh box-$i $((8110+i)) ${G[$i]} $V 1 --limit-mm-per-prompt '{"image":0}' --no-enable-prefix-caching --data-parallel-size 2 --enable-expert-parallel --num-gpu-blocks-override $2 >/dev/null
  SPECS[$i]="$spec"; i=$((i+1)); sleep 20
done
for i in 0 1 2; do ./wait.sh box-$i $((8110+i)) 1500 | tail -1; done
for rep in 1 2; do
  for i in 0 1 2; do set -- ${SPECS[$i]}; (MULT=10 ./bench.sh $TAG-seat$i-bt$1-blk$2 $3 2800 250 box-$i > results/$TAG-seat$i.tmp) & done; wait
  for i in 0 1 2; do cat results/$TAG-seat$i.tmp | tee -a results/summary.txt | cut -c1-200; done
done
for i in 0 1 2; do docker logs box-$i > logs/$TAG-seat$i.log 2>&1; docker rm -f box-$i >/dev/null 2>&1; done
echo ALLDONE
