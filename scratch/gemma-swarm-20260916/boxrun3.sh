#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
until grep -q ALLDONE logs/boxrun2.out; do sleep 10; done
export EXTRA_ENV="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple"
DP="--data-parallel-size 2 --enable-expert-parallel"; V=vllm/vllm-openai-rocm:v0.20.2
CONCS="28 28" STAGGER=20 ./box.sh box-dp2ep2x3-v0202-bt8192-rerun $V 1 $DP
CONCS="20 24" STAGGER=20 BT=16384 ./box.sh box-dp2ep2x3-v0202-bt16384 $V 1 $DP
echo ALLDONE
