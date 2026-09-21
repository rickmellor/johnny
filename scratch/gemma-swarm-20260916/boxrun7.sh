#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
until grep -q ALLDONE logs/boxrun6.out; do sleep 10; done
export MULT=10 STAGGER=20 EXTRA_ENV="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple" GROUPS_="1,2 3,4 0,5"; V=vllm/vllm-openai-rocm:v0.20.2
CONCS="28 30" ./box.sh box-dp2ep2x3-v0202-topo-long $V 1 --data-parallel-size 2 --enable-expert-parallel
CONCS="32" ./box.sh box-tp2x3-v0202-topo-long $V 2
echo ALLDONE
