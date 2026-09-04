#!/bin/bash
# llama-bench matrix on the cached HIP build layer. usage: lb.sh <label> <HIP_VISIBLE_DEVICES> [extra llama-bench args...]
L=$1; V=$2; shift 2
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add=44 --group-add=992 --ipc=host \
  -v /home/rick/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL:/w:ro -e HIP_VISIBLE_DEVICES=$V -e LD_LIBRARY_PATH=/app/lib \
  --entrypoint /app/full/llama-bench 27cb20db598a -m /w/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -ngl 999 -p 512 -n 64 -r 2 -o md "$@" 2>&1 | grep -E "^\|" | grep -v -- "---" | sed "s/^/[$L] /"
