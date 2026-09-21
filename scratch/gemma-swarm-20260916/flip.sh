#!/usr/bin/bash
# flip.sh — does a burst of pure prefill / pure decode flip a DP2+EP2 seat into the slow mode? three variants, same sequence
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
V=vllm/vllm-openai-rocm:v0.20.2; G=(0,1 2,3 4,5); P2P="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple"
BLK=(15200 13000 15200); XENV=("" "" "PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
for i in 0 1 2; do
  EXTRA_ENV="$P2P ${XENV[$i]}" ./launch.sh box-$i $((8110+i)) ${G[$i]} $V 1 --limit-mm-per-prompt '{"image":0}' --enable-prefix-caching --data-parallel-size 2 --enable-expert-parallel --num-gpu-blocks-override ${BLK[$i]} >/dev/null; sleep 20
done
for i in 0 1 2; do ./wait.sh box-$i $((8110+i)) 1500 | tail -1; done
step() { echo "--- $1"; NP=$4 ./bench.sh flip-$1 30 $2 $3 box-0 box-1 box-2 | cut -c1-40,95-400; }
step 1-mixed 2800 250 150
step 2-prefill-burst 2800 1 300
step 3-mixed 2800 250 150
step 4-decode-burst 64 250 200
step 5-mixed 2800 250 150
echo "seat0=blk15200 control | seat1=blk13000 | seat2=blk15200+expandable_segments"
echo FLIPDONE
