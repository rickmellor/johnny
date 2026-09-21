#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
export EXTRA_ENV="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple"
DP="--data-parallel-size 2 --enable-expert-parallel"; V=vllm/vllm-openai-rocm:v0.20.2
export MULT=10
CONCS="24 24 28" STAGGER=20 BT=16384 ./box.sh box-dp2ep2x3-v0202-bt16384-long $V 1 $DP
CONCS="28 28 30" STAGGER=20 BT=8192 ./box.sh box-dp2ep2x3-v0202-bt8192-long $V 1 $DP
echo ALLDONE
