#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
V=vllm/vllm-openai-rocm:v0.20.2
CONCS="24 32 36" ./box.sh box-tp2x3-v0202-bt8192 $V 2
CONCS="16 24 28" EXTRA_ENV="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple" ./box.sh box-dp2ep2x3-v0202-bt8192 $V 1 --data-parallel-size 2 --enable-expert-parallel
echo ALLDONE
