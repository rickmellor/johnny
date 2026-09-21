#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
export MULT=10 STAGGER=20; V=vllm/vllm-openai-rocm:v0.20.2
CONCS="32 32" ./box.sh box-tp2x3-v0202-bt8192-long $V 2
export EXTRA_ENV="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple"
GROUPS_="0,1,2 3,4,5" CONCS="48 56" ./box.sh box-dp3ep3x2-v0202-bt8192-long $V 1 --data-parallel-size 3 --enable-expert-parallel
GROUPS_="0,1,2,3,4,5" CONCS="96 144" ./box.sh box-dp6ep6x1-v0202-bt8192-long $V 1 --data-parallel-size 6 --enable-expert-parallel
echo ALLDONE
