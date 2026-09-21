#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
P2P="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple"; V28=vllm/vllm-openai-rocm:v0.28.0; V20=vllm/vllm-openai-rocm:v0.20.2
SPEC='{"model":"/models/google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4,"moe_backend":"triton"}'
COMMON=(--limit-mm-per-prompt '{"image":0}' --no-enable-prefix-caching)
./launch.sh box-0 8110 0,1 $V28 2 "${COMMON[@]}" --speculative-config "$SPEC" >/dev/null; sleep 30
EXTRA_ENV="$P2P" GMU=0.93 ./launch.sh box-1 8111 2,3 $V28 1 "${COMMON[@]}" --data-parallel-size 2 --enable-expert-parallel --speculative-config "$SPEC" >/dev/null; sleep 30
EXTRA_ENV="$P2P" ./launch.sh box-2 8112 4,5 $V20 1 "${COMMON[@]}" --data-parallel-size 2 --enable-expert-parallel --num-gpu-blocks-override 15200 >/dev/null
for i in 0 1 2; do ./wait.sh box-$i $((8110+i)) 1500 | tail -3; done
for i in 0 1 2; do docker logs box-$i 2>&1 | grep -E "GPU KV cache size" | head -1 | sed "s/.*GPU KV/seat$i KV/"; done
T=(tp2-mtp4-v0280 dp2ep2-mtp4-v0280 dp2ep2-nomtp-v0202-BASELINE); CC=(32 24 30)
for rep in 1 2; do for i in 0 1 2; do curl -sf -o /dev/null localhost:$((8110+i))/health && ./rt_bench.sh MTP1-${T[$i]} $i $((8110+i)) gemma ${CC[$i]} & done; wait; done | tee -a results/summary.txt
for i in 0 1 2; do docker logs box-$i > logs/MTP1-${T[$i]}.log 2>&1; done
echo MTP1DONE
