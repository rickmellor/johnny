#!/usr/bin/env bash
# llama-bench speed bench (single-stream prefill/decode) for a GGUF via the llamacpp
# docker image. GGUF/llamacpp only — llama-bench loads weights directly (not a client),
# so it can't target vLLM. Prints llama.cpp's pp/tg table (pp512 = prefill t/s,
# tg128 = decode t/s).
#
# Usage: bench_llama_speed.sh <gguf (abs, or rel to models_dir)> [ngl] [n_cpu_moe] [override_tensor]
set -uo pipefail
GGUF="${1:?gguf path}"; NGL="${2:-999}"; NCMOE="${3:-0}"; OT="${4:-}"
IMAGE="${JOHNNY_LLAMACPP_IMAGE:-johnny-llamacpp-dsv4:gfx1201}"
MODELS_DIR="${JOHNNY_MODELS_DIR:-$HOME/models}"
case "$GGUF" in
  /models/*) CPATH="$GGUF" ;;
  "$MODELS_DIR"/*) CPATH="/models/${GGUF#"$MODELS_DIR"/}" ;;
  /*) CPATH="$GGUF" ;;
  *) CPATH="/models/$GGUF" ;;
esac
ARGS=(-m "$CPATH" -ngl "$NGL" -fa 0 -p 512 -n 128 -r 2)
[ "$NCMOE" != "0" ] && ARGS+=(-ncmoe "$NCMOE")
[ -n "$OT" ] && ARGS+=(-ot "$OT")
exec docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined --ipc=host -v "$MODELS_DIR":/models:ro \
  --entrypoint /opt/llamacpp/bin/llama-bench "$IMAGE" "${ARGS[@]}"
