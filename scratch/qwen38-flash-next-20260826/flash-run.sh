#!/bin/bash
# usage: flash-run.sh <image> <ctx> <parallel> [extra llama-server args...]
IMG=${1:-johnny-llamacpp-qwen4exp:gfx1201}; CTX=${2:-65536}; NP=${3:-1}; shift 3 2>/dev/null
W=/home/rick/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL
VG=$(getent group video | cut -d: -f3); RG=$(getent group render | cut -d: -f3)
docker rm -f flash-test >/dev/null 2>&1
docker run -d --name flash-test --device=/dev/kfd --device=/dev/dri --group-add=$VG --group-add=$RG \
  --ipc=host --shm-size 16g -v $W:/weights:ro -p 0.0.0.0:8003:8080 -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  $IMG -m /weights/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --host 0.0.0.0 --port 8080 \
  --alias Qwen3.8-Flash-Next -ngl 999 -fa on -c $CTX --parallel $NP --metrics --jinja \
  --load-mode none --reasoning on --reasoning-format deepseek \
  --chat-template-kwargs '{"reasoning_effort":"low"}' \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 "$@"
