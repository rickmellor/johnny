#!/usr/bin/bash
# Tune fused-MoE for the EP2 shape on all six GPUs (v0.20.2), merge to tuned-moe-ep/
cd /home/rick/repos/johnny/scratch/gemma-swarm-20260916; S=$PWD; O=$S/tuned-moe-ep; mkdir -p $O $S/tcache-moe-ep
for g in 0 1 2 3 4 5; do
  docker run -d --rm --name tune-ep-gpu$g --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 992 -e HIP_VISIBLE_DEVICES=$g -e TRITON_CACHE_DIR=/tcache \
    -v $S:/tune -v $O:/out -v $S/tcache-moe-ep:/tcache --entrypoint bash vllm/vllm-openai-rocm:v0.20.2 \
    -c "pip install -q ray >/dev/null 2>&1; python3 /tune/tune_moe_ep.py --gpu $g --ngpu 6 --out /out > /out/gpu$g.log 2>&1" >/dev/null
done
while [ "$(docker ps --format '{{.Names}}' | grep -c tune-ep-gpu)" -gt 0 ]; do sleep 20; done
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 992 -e HIP_VISIBLE_DEVICES=0 -v $S:/tune -v $O:/out --entrypoint bash vllm/vllm-openai-rocm:v0.20.2 \
  -c "pip install -q ray >/dev/null 2>&1; python3 /tune/tune_moe_ep.py --merge --out /out" 2>&1 | grep -E "merged|Error|Traceback" | tail -3
ls $O; echo TUNEDONE
