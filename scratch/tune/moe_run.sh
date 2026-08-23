#!/usr/bin/env bash
# Run the Ray-free MoE tuner on all 4 GPUs for one image, merge, report. usage: moe_run.sh <image> <tag>
set -u; S=$(cd "$(dirname "$0")" && pwd); IMG="$1"; TAG="$2"; O="$S/moe-$TAG"; mkdir -p "$O" "$S/tcache-moe-$TAG"
for g in 0 1 2 3; do
  nohup docker run --rm --name tune-moe-$TAG-gpu$g --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=$g -e TRITON_CACHE_DIR=/tcache \
    -v "$S":/tune -v "$O":/out -v "$S/tcache-moe-$TAG":/tcache --entrypoint bash vllm/vllm-openai-rocm:$IMG \
    -c "pip install -q ray >/dev/null 2>&1; python3 /tune/tune_moe_gemma.py --gpu $g --ngpu 4 --tp 2 --out /out" > "$S/moe-$TAG-gpu$g.log" 2>&1 &
done
while [ "$(grep -l '\[gpu[0-9]\] DONE' $S/moe-$TAG-gpu*.log 2>/dev/null | wc -l)" -lt 4 ]; do sleep 20; done
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0 -v "$S":/tune -v "$O":/out --entrypoint bash vllm/vllm-openai-rocm:$IMG -c "pip install -q ray >/dev/null 2>&1; python3 /tune/tune_moe_gemma.py --merge --tp 2 --out /out" 2>&1 | grep -E "merged|Error|Traceback" | tail -3
ls "$O"
