#!/usr/bin/bash
# probe-mtp-availability.sh — does this model have an MTP head?
#
# Usage:   ./probe-mtp-availability.sh <model-dir>
# Example: ./probe-mtp-availability.sh ~/models/cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit
#
# Reports:
#   - mtp_num_hidden_layers from text_config (Qwen family) or top-level
#   - Number of mtp.* / nextn.* weight tensors in the safetensors index
#   - Whether MTP tensors are in the quantization 'ignore' list (unquantized)
#
# Exit 0 = MTP present and usable. Exit 1 = no MTP, do not pass
# --speculative-config '{"method":"mtp",...}' to vLLM.
#
# Observed acceptance rates by family (as of 2026-06-08):
#   Qwen3.5-MoE (122B-A10B):  pos[0] ~85%, pos[1] ~66%, mean 2.49 → +33% single-stream
#   Qwen3.6 dense (27B):       pos[0] ~58%, pos[1] ~35%, mean 1.95 → +63% single, -45% peak
#   Gemma-4 family:            NO MTP head (none of the gemma-4 variants tested have it)

set -e

MODEL_DIR="${1:?usage: $0 <model-dir>}"
[ -f "$MODEL_DIR/config.json" ] || { echo "ERR: no config.json at $MODEL_DIR"; exit 1; }
[ -f "$MODEL_DIR/model.safetensors.index.json" ] || { echo "ERR: no safetensors index at $MODEL_DIR"; exit 1; }

python3 << PY
import json, sys

model_dir = "$MODEL_DIR"
c = json.load(open(f"{model_dir}/config.json"))
tc = c.get('text_config', c)  # some models put it top-level, others under text_config

mtp_layers = tc.get('mtp_num_hidden_layers') or c.get('mtp_num_hidden_layers') or c.get('num_nextn_predict_layers')
print(f"mtp_num_hidden_layers:    {mtp_layers}")
print(f"mtp_use_dedicated_embeds: {tc.get('mtp_use_dedicated_embeddings')}")

idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))
keys = list(idx.get('weight_map', {}).keys())
mtp_keys = [k for k in keys if k.startswith('mtp.') or k.startswith('model.mtp.') or 'nextn' in k.lower()]
print(f"mtp.* / nextn weights:    {len(mtp_keys)}")

if mtp_keys:
    top_parts = sorted(set(k.split('.', 3)[1] for k in mtp_keys if len(k.split('.')) >= 2))
    print(f"mtp.<X> top-level parts:  {top_parts}")
    # Confirm MTP is unquantized (otherwise vLLM may fail to find drafter weights)
    ignore = c.get('quantization_config', {}).get('ignore', [])
    mtp_in_ignore = [x for x in ignore if 'mtp' in x.lower()]
    print(f"mtp in quantization.ignore: {len(mtp_in_ignore)} entries (unquantized: {'YES' if mtp_in_ignore else 'NO'})")
    print()
    print("VERDICT: MTP available. Use:")
    print("  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'")
    print()
    print("Verify acceptance rate from runtime logs:")
    print("  docker logs <container> 2>&1 | grep 'SpecDecoding metrics'")
    print("Healthy: pos[0] > 70%, mean acceptance length > 2.0")
    sys.exit(0)
else:
    print()
    print("VERDICT: No MTP head. Do NOT pass --speculative-config; vLLM will fail")
    print("to find drafter weights and refuse to start.")
    sys.exit(1)
PY
