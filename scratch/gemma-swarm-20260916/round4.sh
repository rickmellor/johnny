#!/usr/bin/bash
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916
V=vllm/vllm-openai-rocm
P2P="HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple"
EXTRA_ENV="$P2P" ./cfg.sh pp2-v0202 32 $V:v0.20.2 1 --pipeline-parallel-size 2
EXTRA_ENV="VLLM_TUNED_CONFIG_FOLDER=/tuned" EXTRA_VOL="$PWD/tuned-moe:/tuned:ro" ./cfg.sh tp2-v0202-tunedmoe 32 $V:v0.20.2 2
./cfg.sh tp2-nightly-bf16 32 $V:nightly-27a94d1ce4e3fc100c4732439ccec10f8246a804 2
./cfg.sh tp2-v0280-bf16-noaiter 32 $V:v0.28.0 2
EXTRA_ENV="$P2P" ./cfg.sh dp2ep-v0202 16 $V:v0.20.2 1 --data-parallel-size 2 --enable-expert-parallel
GPUS=4 GMU=0.95 BT=2048 SEQS=8 ./cfg.sh tp1-v0202-diag 4 $V:v0.20.2 1
echo ROUND4 DONE
