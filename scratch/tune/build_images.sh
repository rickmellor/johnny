#!/usr/bin/env bash
# Bake the tuned JSONs into derived images: <base> + configs → johnny-vllm-rocm:<tag>-gfx1201
set -eu; S=$(cd "$(dirname "$0")" && pwd)
build(){ BASE="$1"; TAG="$2"; D="$S/img-$TAG"; rm -rf "$D"; mkdir -p "$D/fp8" "$D/moe"
  cp "$S"/out/*.json "$D/fp8/" 2>/dev/null || true
  ls "$S"/moe-"$TAG"/*.json >/dev/null 2>&1 && cp "$S"/moe-"$TAG"/*.json "$D/moe/" || ls "$S"/moe-any/*.json >/dev/null 2>&1 && cp "$S"/moe-any/*.json "$D/moe/" || true
  cat > "$D/Dockerfile" <<EOF
FROM $BASE
# RDNA4 (gfx1201 / AMD_Radeon_R9700) tuned Triton configs — w8a8 block-FP8 GEMM + fused-MoE (tuned on specul8-o-matic 2026-08-23)
COPY fp8/ /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/configs/
COPY moe/ /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/
EOF
  touch "$D/fp8/.keep" "$D/moe/.keep"
  docker build -q -t "johnny-vllm-rocm:$TAG-gfx1201" "$D" && echo "built johnny-vllm-rocm:$TAG-gfx1201 (fp8: $(ls "$D/fp8" | grep -c json), moe: $(ls "$D/moe" | grep -c json))"; }
build vllm/vllm-openai-rocm:v0.27.1 v0.27.1
build vllm/vllm-openai-rocm:nightly-e9d1398d9edfd90fcc1cf783805240e3effec013 nightly0822
build vllm/vllm-openai-rocm:v0.20.2 v0.20.2
