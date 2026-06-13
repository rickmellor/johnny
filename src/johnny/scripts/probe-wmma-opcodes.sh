#!/usr/bin/bash
# Probe which WMMA / SWMMAC opcodes the gfx1201 (RDNA4 / R9700) assembler accepts.
#
# Strategy: feed each candidate mnemonic to llvm-mc -mcpu=gfx1201. If the
# assembler emits "invalid instruction" / "unknown" / "not a recognized
# mnemonic", the hardware does not implement it. Operand-shape errors are
# treated as ACCEPTED because they prove the mnemonic itself is known.
#
# Run on specul8-o-matic (or any host with docker access to the vllm-rocm image).
# No GPU access needed — this is pure assembler introspection.
#
# Usage:
#   bash scripts/probe-wmma-opcodes.sh
#   bash scripts/probe-wmma-opcodes.sh vllm/vllm-openai-rocm:v0.20.2   # explicit image
#   bash scripts/probe-wmma-opcodes.sh '' gfx1201                      # explicit arch

set -u

IMAGE="${1:-vllm/vllm-openai-rocm:v0.20.2}"
ARCH="${2:-gfx1201}"

# All known RDNA4-relevant WMMA / SWMMAC candidates plus the suspected-missing
# FP4 variants. Add to this list as new precisions appear in AMD's ISA notes.
OPS=(
  # Dense WMMA 16x16x16 — confirmed present on gfx1201
  v_wmma_f32_16x16x16_f16
  v_wmma_f32_16x16x16_bf16
  v_wmma_f16_16x16x16_f16
  v_wmma_bf16_16x16x16_bf16
  v_wmma_i32_16x16x16_iu8
  v_wmma_i32_16x16x16_iu4
  v_wmma_f32_16x16x16_fp8_fp8
  v_wmma_f32_16x16x16_fp8_bf8
  v_wmma_f32_16x16x16_bf8_fp8
  v_wmma_f32_16x16x16_bf8_bf8

  # Dense WMMA 16x16x32 (wider-K)
  v_wmma_f32_16x16x32_f16
  v_wmma_f32_16x16x32_bf16
  v_wmma_f32_16x16x32_fp8_fp8

  # Speculative FP4 dense variants (expected REJECTED on gfx1201)
  v_wmma_f32_16x16x16_fp4_fp4
  v_wmma_f32_16x16x32_fp4
  v_wmma_f32_16x16x64_fp4
  v_wmma_f16_16x16x16_fp4_fp4

  # Sparse SWMMAC 16x16x32 (2:4 structured sparsity)
  v_swmmac_f32_16x16x32_f16
  v_swmmac_f32_16x16x32_bf16
  v_swmmac_f32_16x16x32_fp8_fp8
  v_swmmac_i32_16x16x32_iu8
  v_swmmac_i32_16x16x32_iu4
  v_swmmac_f32_16x16x32_fp4_fp4
)

echo "=== WMMA opcode probe ==="
echo "image: $IMAGE"
echo "arch:  $ARCH"
echo

docker run --rm --entrypoint bash \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render \
  "$IMAGE" -c "
MC=/opt/rocm/llvm/bin/llvm-mc
ARGS=\"-arch=amdgcn -mcpu=$ARCH -filetype=null\"

for OP in ${OPS[*]}; do
  # Synthesize a plausible operand list. Operand-shape errors still prove the
  # mnemonic is recognized, so we treat them as ACCEPTED below.
  ASM=\"\${OP} v[0:7], v[8:11], v[12:15], v[0:7]\"
  OUT=\$(echo \"\$ASM\" | \"\$MC\" \$ARGS 2>&1)
  if [ -z \"\$OUT\" ]; then
    printf '  ACCEPTED  %s\n' \"\$OP\"
  else
    if echo \"\$OUT\" | grep -qiE 'invalid instruction|unknown|not a recognized|invalid mnemonic'; then
      printf '  REJECTED  %s\n' \"\$OP\"
    else
      printf '  ACCEPTED  %s   (operand-shape err only)\n' \"\$OP\"
    fi
  fi
done
"

echo
echo "Interpretation:"
echo "  ACCEPTED  = mnemonic known to gfx1201 → hardware implements it"
echo "  REJECTED  = mnemonic unknown to assembler → hardware does NOT implement it"
echo "See references/rdna4-wmma-precision-support.md for the cached results table."
