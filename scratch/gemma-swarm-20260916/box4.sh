#!/usr/bin/bash
# box4.sh <round> "<tag image tp conc gmu seqs K dp blk>" x3 — per-seat configs, real-text bench w/ E2E. K=MTP tokens (0=off), dp=0|2 (DP+EP), blk=block override (0=auto)
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
R=$1; shift; G=(0,1 2,3 4,5); i=0
for spec in "$@"; do set -- $spec; TAGS[$i]=$1; CC[$i]=$4
  A=(--limit-mm-per-prompt '{"image":0}' --no-enable-prefix-caching)
  [ "$7" != 0 ] && A+=(--speculative-config "{\"model\":\"/models/google/gemma-4-26B-A4B-it-assistant\",\"num_speculative_tokens\":$7,\"moe_backend\":\"triton\"}")
  [ "$8" != 0 ] && A+=(--data-parallel-size $8 --enable-expert-parallel)
  [ "$9" != 0 ] && A+=(--num-gpu-blocks-override $9)
  EXTRA_ENV="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple" GMU=$5 SEQS=$6 ./launch.sh box-$i $((8110+i)) ${G[$i]} vllm/vllm-openai-rocm:$2 $3 "${A[@]}" >/dev/null
  i=$((i+1)); sleep 30
done
for i in 0 1 2; do ./wait.sh box-$i $((8110+i)) 1500 | tail -3; done
for i in 0 1 2; do docker logs box-$i 2>&1 | grep -E "GPU KV cache size" | head -1 | sed "s/.*GPU KV/seat$i ${TAGS[$i]} KV/" | cut -c1-90; done
for rep in 1 2 3; do for i in 0 1 2; do c=$(echo ${CC[$i]} | cut -d, -f$rep); [ -n "$c" ] && curl -sf -o /dev/null localhost:$((8110+i))/health && ./rt_bench.sh $R-${TAGS[$i]} $i $((8110+i)) gemma $c & done; wait; done | tee -a results/summary.txt
for i in 0 1 2; do docker logs box-$i > logs/$R-${TAGS[$i]}.log 2>&1; docker rm -f box-$i >/dev/null; done
echo ROUNDDONE
