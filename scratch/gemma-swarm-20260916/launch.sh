#!/usr/bin/bash
# launch.sh <name> <port> <gpus csv> <image> <tp> [extra vllm args...]   (env: EXTRA_ENV="K=V K=V", GMU, SEQS, BT, MML)
# Experiment seat mirroring johnny's vLLM container recipe; NOT johnny-managed (ports 81xx).
set -euo pipefail
NAME=$1 PORT=$2 GPUS=$3 IMAGE=$4 TP=$5; shift 5
MODEL=${MODEL:-RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic}
ENVS=(-e HIP_VISIBLE_DEVICES=$GPUS)
for kv in ${EXTRA_ENV:-}; do ENVS+=(-e "$kv"); done
VOLS=(); for v in ${EXTRA_VOL:-}; do VOLS+=(-v "$v"); done
if [ "$TP" -gt 1 ]; then ENVS+=(-e HSA_ENABLE_IPC_MODE_LEGACY=0 -e NCCL_PROTO=Simple); fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --device /dev/kfd --device /dev/dri --group-add 44 --group-add 992 \
  --ipc host --shm-size 16g --security-opt label=disable \
  -v /mnt/data/models:/models -v /mnt/data/vllm-cache:/root/.cache/vllm "${VOLS[@]}" \
  -p 127.0.0.1:${PORT}:8000 "${ENVS[@]}" "$IMAGE" \
  "/models/$MODEL" --served-model-name gemma \
  --tensor-parallel-size "$TP" --max-model-len "${MML:-4096}" \
  --gpu-memory-utilization "${GMU:-0.95}" --max-num-seqs "${SEQS:-256}" \
  --max-num-batched-tokens "${BT:-8192}" "$@"
