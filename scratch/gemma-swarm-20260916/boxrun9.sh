#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
export MULT=10 STAGGER=20 EXTRA_ENV="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple"
DP="--data-parallel-size 2 --enable-expert-parallel"
CONCS="24 26" GMU=0.93 ./box.sh box-dp2ep2x3-nightly-gmu93-long vllm/vllm-openai-rocm:nightly-27a94d1ce4e3fc100c4732439ccec10f8246a804 1 $DP
./tune_run.sh
export EXTRA_ENV="$EXTRA_ENV VLLM_TUNED_CONFIG_FOLDER=/tuned" EXTRA_VOL="$PWD/tuned-moe-ep:/tuned:ro"
CONCS="26 26 28" GMU=0.92 ./box.sh box-dp2ep2x3-v0202-gmu92-tunedmoe-long vllm/vllm-openai-rocm:v0.20.2 1 $DP
echo ALLDONE
