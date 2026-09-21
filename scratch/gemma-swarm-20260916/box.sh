#!/usr/bin/bash
# CONCS="16 24" box.sh <tag> <image> <tp> [vllm args...] — three seats on HIP pairs 0,1 / 2,3 / 4,5, benched simultaneously
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
TAG=$1 IMG=$2 TP=$3; shift 3
echo "### $TAG"
i=0; for g in ${GROUPS_:-0,1 2,3 4,5}; do
  ./launch.sh box-$i $((8110+i)) $g $IMG $TP --limit-mm-per-prompt '{"image":0}' --no-enable-prefix-caching "$@" >/dev/null; i=$((i+1)); sleep ${STAGGER:-45}
done
N=$i; SEATS=""; for j in $(seq 0 $((N-1))); do SEATS="$SEATS box-$j"; done
ok=1; for i in $(seq 0 $((N-1))); do ./wait.sh box-$i $((8110+i)) 1500 | tail -2; curl -sf -o /dev/null localhost:$((8110+i))/health || ok=0; done
if [ $ok = 1 ]; then
  docker logs box-0 2>&1 | grep -E "GPU KV cache size" | head -6 | sed 's/.*GPU KV/KV/'
  for C in $CONCS; do ./bench.sh $TAG $C 2800 250 $SEATS | tee -a results/summary.txt
    for j in $(seq 0 $((N-1))); do echo -n "  seat$j preemptions: "; curl -s localhost:$((8110+j))/metrics | grep -E '^vllm:num_preemptions_total' | awk '{s+=$2; printf "%s ", $2} END {print "(sum " s ")"}'; done
  done
fi
for i in $(seq 0 $((N-1))); do docker logs box-$i > logs/$TAG-seat$i.log 2>&1; docker rm -f box-$i >/dev/null 2>&1; done
echo BOXDONE
