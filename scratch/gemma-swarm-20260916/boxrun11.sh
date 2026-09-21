#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
export MULT=10 STAGGER=20 EXTRA_ENV="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple"
CONCS="28 30 28" ./box.sh FINAL-dp2ep2x3-v0202-bt8192-blk15200 vllm/vllm-openai-rocm:v0.20.2 1 --data-parallel-size 2 --enable-expert-parallel --num-gpu-blocks-override 15200
echo ALLDONE
