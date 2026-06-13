#!/usr/bin/bash
# One-shot: push coder context to native 256K at gmu=0.95 (bf16 KV, MTP on).
# If KV preflight rejects the target, read vLLM's reported ceiling and relaunch
# at that exact max. Lands on the largest mml that fits.
set -u
PORT=8002
launch() {  # $1=mml $2=gmu
  docker rm -f vllm-coder >/dev/null 2>&1
  docker run -d --name vllm-coder \
    --device=/dev/kfd --device=/dev/dri --group-add=video --group-add=render \
    --ipc=host --shm-size=32g \
    -v ~/models:/models -v ~/vllm/vllm-cache:/root/.cache/vllm \
    -p ${PORT}:8000 \
    -e HIP_VISIBLE_DEVICES=2,3 -e NCCL_PROTO=Simple \
    -e HIP_FORCE_DEV_KERNARG=1 -e SAFETENSORS_FAST_GPU=1 \
    vllm/vllm-openai-rocm:v0.20.2 \
    /models/Qwen/Qwen3.6-27B-FP8 \
    --served-model-name qwen-27b-coder \
    --tensor-parallel-size 2 \
    --max-model-len "$1" \
    --gpu-memory-utilization "$2" \
    --enable-prefix-caching \
    --max-num-seqs 64 --max-num-batched-tokens 16384 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    >/dev/null 2>&1
}
wait_outcome() {  # echoes READY | CEILING:<n> | FAIL | TIMEOUT
  local deadline=$((SECONDS+600))
  while [ $SECONDS -lt $deadline ]; do
    curl -s --max-time 4 http://localhost:${PORT}/v1/models 2>/dev/null | grep -q '"id"' && { echo READY; return; }
    local ceil
    ceil=$(docker logs vllm-coder 2>&1 | grep -oiE "estimated maximum model length is [0-9]+" | grep -oE "[0-9]+" | tail -1)
    [ -n "$ceil" ] && { echo "CEILING:$ceil"; return; }
    local st; st=$(docker inspect -f '{{.State.Status}}' vllm-coder 2>/dev/null)
    if [ "$st" != "running" ]; then
      ceil=$(docker logs vllm-coder 2>&1 | grep -oiE "maximum model length is [0-9]+" | grep -oE "[0-9]+" | tail -1)
      [ -n "$ceil" ] && { echo "CEILING:$ceil"; return; }
      echo FAIL; return
    fi
    sleep 10
  done
  echo TIMEOUT
}
floor1024() { echo $(( ($1/1024)*1024 )); }
report() {
  echo "--- final KV report ---"
  docker logs vllm-coder 2>&1 | grep -iE "Available KV cache memory|GPU KV cache size|Maximum concurrency" | tail -4
}

echo "### Attempt 1: mml=262144 gmu=0.95"
launch 262144 0.95
r=$(wait_outcome); echo "outcome: $r"
if [ "$r" = READY ]; then echo "SUCCESS: coder now at 262144 (full native 256K)"; report; exit 0; fi
if [[ "$r" == CEILING:* ]]; then
  c=$(floor1024 ${r#CEILING:})
  echo "### 262144 rejected; vLLM ceiling=${r#CEILING:} -> retry at $c"
  launch "$c" 0.95
  r2=$(wait_outcome); echo "outcome: $r2"
  if [ "$r2" = READY ]; then echo "SUCCESS: coder now at $c"; report; exit 0; fi
  if [[ "$r2" == CEILING:* ]]; then
    c2=$(floor1024 ${r2#CEILING:})
    echo "### ceiling shifted to ${r2#CEILING:} -> retry at $c2"
    launch "$c2" 0.95
    r3=$(wait_outcome); echo "outcome: $r3"
    [ "$r3" = READY ] && { echo "SUCCESS: coder now at $c2"; report; exit 0; }
    echo "FELL THROUGH at $c2: $r3"; report; exit 1
  fi
  echo "retry failed: $r2"; docker logs vllm-coder 2>&1 | tail -6; exit 1
fi
echo "attempt 1 failed: $r"; docker logs vllm-coder 2>&1 | tail -8; exit 1
