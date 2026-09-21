#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
export MULT=10 STAGGER=20 EXTRA_ENV="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple"
DP="--data-parallel-size 2 --enable-expert-parallel"; V=vllm/vllm-openai-rocm:v0.20.2
CONCS="30 30" GMU=0.95 ./box.sh box-dp2ep2x3-v0202-gmu95-preemptcheck $V 1 $DP
CONCS="32 40" GMU=0.95 SEQS=14 ./box.sh box-dp2ep2x3-v0202-gmu95-seqs14 $V 1 $DP
echo ALLDONE
