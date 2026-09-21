#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
until grep -q ALLDONE logs/boxrun5.out; do sleep 10; done
export MULT=10 STAGGER=20 EXTRA_ENV="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple"; V=vllm/vllm-openai-rocm:v0.20.2
CONCS="28 28" ./box.sh box-dp2noep-x3-v0202-long $V 1 --data-parallel-size 2
CONCS="32 36" ./box.sh box-tp2ep-x3-v0202-long $V 2 --enable-expert-parallel
CONCS="30 32" GMU=0.97 SEQS=32 ./box.sh box-dp2ep2x3-v0202-seqs32-gmu97-long $V 1 --data-parallel-size 2 --enable-expert-parallel
echo ALLDONE
