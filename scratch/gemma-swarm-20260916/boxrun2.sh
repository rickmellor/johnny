#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
export EXTRA_ENV="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple"
DP="--data-parallel-size 2 --enable-expert-parallel"
CONCS="24 28 32" ./box.sh box-dp2ep2x3-nightly-bt8192 vllm/vllm-openai-rocm:nightly-27a94d1ce4e3fc100c4732439ccec10f8246a804 1 $DP
CONCS="28 32 40" GMU=0.97 BT=4096 ./box.sh box-dp2ep2x3-v0202-bt4096-gmu97 vllm/vllm-openai-rocm:v0.20.2 1 $DP
echo ALLDONE
