#!/usr/bin/env python3
"""
Audit ~/models for tuning coverage gaps and unused MTP capabilities.

Usage: python3 audit-models.py [/path/to/models]
Default models dir: ~/models

Output: one row per model with: path, size, quantization, multimodal flag,
MTP head presence (from text_config.mtp_num_hidden_layers and safetensors index),
and whether a TUNING_REPORT.md exists next to the model.

Use this BEFORE starting a tuning session to identify:
  - Models present on disk but never benched
  - Models with MTP heads that no current launcher exercises
  - Empty placeholder directories (size 0)

The "MTP" column shows Y/<tensor_count> when the model ships a multi-token
prediction head. Even if mtp_num_hidden_layers is None, a non-zero tensor
count means the head exists in the safetensors index — try enabling MTP
via vLLM's --speculative-config.
"""
import json
import os
import sys
import glob


def audit(models_dir):
    rows = []
    for cfg_path in sorted(glob.glob(f'{models_dir}/*/*/config.json')):
        model_dir = os.path.dirname(cfg_path)
        rel = os.path.relpath(model_dir, models_dir)
        has_tuning = os.path.exists(f'{model_dir}/TUNING_REPORT.md')
        safetensors = glob.glob(f'{model_dir}/*.safetensors')
        total_bytes = sum(os.path.getsize(f) for f in safetensors)
        size_gb = total_bytes / (1024**3)

        try:
            c = json.load(open(cfg_path))
        except Exception as e:
            rows.append({'path': rel, 'error': str(e)})
            continue

        arch = c.get('architectures', ['?'])[0]
        tc = c.get('text_config', c)
        mtp_layers = (
            tc.get('mtp_num_hidden_layers')
            or c.get('num_nextn_predict_layers')
            or tc.get('num_nextn_predict_layers')
        )

        mtp_tensor_count = 0
        idx_path = f'{model_dir}/model.safetensors.index.json'
        if os.path.exists(idx_path):
            try:
                idx = json.load(open(idx_path))
                keys = idx.get('weight_map', {}).keys()
                mtp_tensor_count = sum(
                    1 for k in keys if k.startswith('mtp.') or 'nextn' in k.lower()
                )
            except Exception:
                pass

        qc = c.get('quantization_config', {})
        quant = qc.get('quant_method') or c.get('torch_dtype', 'bf16?')
        if isinstance(qc, dict) and qc.get('format') == 'pack-quantized':
            bits = None
            for g in qc.get('config_groups', {}).values():
                bits = g.get('weights', {}).get('num_bits')
                break
            if bits:
                quant = f'{quant}-INT{bits}'

        rows.append({
            'path': rel,
            'size_gb': round(size_gb, 1),
            'arch': arch,
            'quant': str(quant),
            'multimodal': 'vision_config' in c or 'ForConditionalGeneration' in arch,
            'has_mtp': (mtp_layers and mtp_layers > 0) or mtp_tensor_count > 0,
            'mtp_tensors': mtp_tensor_count,
            'tuned': has_tuning,
        })

    print(f"{'PATH':<50} {'SIZE':>6}  {'QUANT':<22}  {'MM':<3} {'MTP':<8} {'TUNED':<5}")
    print('-' * 110)
    for r in rows:
        if 'error' in r:
            print(f"{r['path']:<50}  PARSE_ERROR: {r['error']}")
            continue
        mm = 'Y' if r['multimodal'] else '-'
        mtp = f"Y/{r['mtp_tensors']}" if r['has_mtp'] else '-'
        tuned = 'Y' if r['tuned'] else '-'
        print(f"{r['path']:<50} {r['size_gb']:>5}G  {r['quant']:<22}  {mm:<3} {mtp:<8} {tuned:<5}")

    print()
    print("GAPS TO INVESTIGATE:")
    untuned_serving = [r for r in rows if isinstance(r, dict) and not r.get('tuned')
                        and r.get('size_gb', 0) > 5 and not r['path'].startswith(('BAAI', 'nomic'))]
    for r in untuned_serving:
        flags = []
        if r['has_mtp']:
            flags.append(f"has MTP head ({r['mtp_tensors']} tensors)")
        print(f"  - {r['path']}: untuned ({', '.join(flags) or 'no MTP'})")

    empty = [r for r in rows if isinstance(r, dict) and r.get('size_gb') == 0]
    for r in empty:
        print(f"  - {r['path']}: EMPTY DIRECTORY (placeholder, consider rmdir)")


if __name__ == '__main__':
    audit(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/models'))
