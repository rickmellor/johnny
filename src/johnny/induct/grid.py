"""Seeded search + hardware-fit + KV-preflight (§3.6).

The novel, launch-free parts of induction:
- **hardware-fit**: which TP options even fit (weights/GPU vs VRAM budget, native
  dtype), pruning the rest *with reasons* — no launch.
- **KV-preflight**: a coarse max-context estimate from config dims, so impossible
  contexts are pruned for free before spending a multi-minute launch.
- **seeded points**: candidate configs seeded by priors (imported/shared placements)
  + arch defaults, kept small (coordinate-descent-ish) rather than a brute grid.
On-disk weight size is used as the weight-bytes truth (robust + quant-agnostic).
"""

from __future__ import annotations

import json
from pathlib import Path

_KV_MIN_BYTES = 2.0e9       # reserve for a minimal KV pool
_OVERHEAD_BYTES = 1.5e9     # CUDA/HIP context, activations, fragmentation
_GMU_CAP = 0.92             # don't assume you can use 100% of VRAM
_QUANT_DTYPE = {"fp8": "fp8", "awq": "int4", "gptq": "int4", "int4": "int4", "bf16": "bf16", "fp4": "fp4"}


def read_config(path: str) -> dict:
    p = Path(path) / "config.json"
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


# Defaults for model families whose config.json is stripped (values come from the
# transformers config class at load time, so they're absent on disk). Keyed by
# model_type. ctx = native context; head_dim where it's fixed across sizes.
_MODEL_TYPE_DEFAULTS = {
    "gemma3": {"ctx": 131072, "head_dim": 256},
    "gemma3_text": {"ctx": 131072, "head_dim": 256},
    "gemma2": {"ctx": 8192, "head_dim": 256},
}

_CTX_KEYS = ("max_position_embeddings", "max_sequence_length", "n_positions", "seq_length")


def dims(config: dict) -> dict:
    tc = config.get("text_config") or {}

    def g(k):
        return config.get(k) if config.get(k) is not None else tc.get(k)

    mt = str(g("model_type") or "").lower()
    defaults = _MODEL_TYPE_DEFAULTS.get(mt, {})

    n_heads = g("num_attention_heads")
    hidden = g("hidden_size")
    head_dim = g("head_dim") or defaults.get("head_dim") or ((hidden // n_heads) if (hidden and n_heads) else None)

    ctx = None
    for k in _CTX_KEYS:
        if g(k):
            ctx = g(k)
            break
    if not ctx:
        ctx = defaults.get("ctx")  # stripped config → family default

    return {
        "layers": g("num_hidden_layers"),
        "kv_heads": g("num_key_value_heads") or n_heads,
        "head_dim": head_dim,
        "ctx": ctx,
    }


def model_size_bytes(path: str) -> int:
    p = Path(path)
    total = 0
    for f in p.glob("*"):
        if f.is_file() and f.suffix in (".safetensors", ".bin", ".gguf", ".pt"):
            total += f.stat().st_size
    if total == 0:  # fallback: everything
        for f in p.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def kv_bytes_per_token(d: dict, kv_dtype_bytes: int = 2) -> float | None:
    if not (d.get("layers") and d.get("kv_heads") and d.get("head_dim")):
        return None
    return 2.0 * d["layers"] * d["kv_heads"] * d["head_dim"] * kv_dtype_bytes


def viable_placements(size_bytes: int, d: dict, quant: str | None, hardware, free_count: int) -> tuple[list, list]:
    """Return (viable, pruned). Each viable: {tp, quant, per_gpu_gb, kv_ceiling_ctx}."""
    nd = set(hardware.native_dtypes)
    need_dtype = _QUANT_DTYPE.get((quant or "").lower())
    vram = min((g.vram_gb for g in hardware.groups), default=0.0)
    viable, pruned = [], []
    kv_pt = kv_bytes_per_token(d, 2)
    tps = sorted({1, 2, 4, 8, free_count} & set(range(1, max(free_count, 1) + 1))) or [1]
    for tp in tps:
        if tp > free_count:
            pruned.append({"tp": tp, "reason": f"needs {tp} GPUs, {free_count} free"})
            continue
        if need_dtype and nd and need_dtype not in nd:
            pruned.append({"tp": tp, "reason": f"quant {quant} -> {need_dtype} not natively accelerated"})
            continue
        per_gpu = size_bytes / tp
        budget = vram * 1e9 * _GMU_CAP
        if per_gpu + _KV_MIN_BYTES + _OVERHEAD_BYTES > budget:
            pruned.append({"tp": tp, "reason": f"weights {per_gpu/1e9:.1f}GB/GPU + KV/overhead exceed {budget/1e9:.1f}GB"})
            continue
        free_for_kv = budget - per_gpu - _OVERHEAD_BYTES
        if kv_pt:
            max_ctx = int(free_for_kv * tp / kv_pt)
            if d.get("ctx"):
                max_ctx = min(max_ctx, d["ctx"])
        else:
            max_ctx = d.get("ctx")
        viable.append({"tp": tp, "quant": quant, "per_gpu_gb": round(per_gpu / 1e9, 1), "kv_ceiling_ctx": max_ctx})
    return viable, pruned


def seed_priors(registry: dict, model_id: str) -> list[dict]:
    """Knob dicts from existing placements (imported or shared) to center the search."""
    m = (registry.get("models") or {}).get(model_id) or {}
    return [p.get("knobs", {}) for p in m.get("placements", []) if p.get("knobs")]


def cpu_viable(size_bytes: int, host_ram_gb: float, headroom: float = 0.7) -> dict:
    """Can this model run on CPU given host RAM? (weights vs ~70% of RAM)."""
    budget = host_ram_gb * 1e9 * headroom
    fits = bool(size_bytes) and size_bytes < budget
    return {
        "device": "cpu",
        "fits": fits,
        "per_host_gb": round(size_bytes / 1e9, 1),
        "reason": None if fits else f"{size_bytes / 1e9:.0f} GB exceeds ~{budget / 1e9:.0f} GB CPU budget ({host_ram_gb:.0f} GB RAM)",
    }


def cpu_candidate_points(embeddings: bool, ncpu: int, native_ctx: int | None,
                         priors: list, max_points: int | None = None) -> list[dict]:
    """CPU sweep: a small grid over cpuset (threads) × context/batch.

    Embeddings get a batch×threads grid at native (small) context; generative CPU
    LLMs sweep threads × a couple of seq settings.
    """
    half = max(1, ncpu // 2)
    cpusets = list(dict.fromkeys([f"0-{ncpu - 1}", f"0-{half - 1}"]))
    prior = priors[0] if priors else {}
    pts: list[dict] = []
    if embeddings:
        mml = prior.get("max_model_len") or native_ctx or 2048
        for cs in cpusets:
            for batched in dict.fromkeys([prior.get("max_num_batched_tokens") or 8192, 16384]):
                pts.append({"device": "cpu", "embeddings": True, "cpuset": cs, "max_model_len": mml,
                            "max_num_batched_tokens": batched, "max_num_seqs": prior.get("max_num_seqs") or 256})
    else:
        mml = prior.get("max_model_len") or (min(native_ctx, 16384) if native_ctx else 8192)
        for cs in cpusets:
            for seqs in dict.fromkeys([prior.get("max_num_seqs") or 8, 16]):
                pts.append({"device": "cpu", "embeddings": False, "cpuset": cs, "max_model_len": mml,
                            "max_num_seqs": seqs, "max_num_batched_tokens": prior.get("max_num_batched_tokens") or 4096})
    if max_points:
        pts = pts[:max_points]
    return pts


def candidate_points(viable: list, priors: list, max_points: int | None = None) -> list[dict]:
    prior = priors[0] if priors else {}
    base_batched = prior.get("max_num_batched_tokens") or 16384
    base_gmu = prior.get("gpu_memory_util") or 0.90
    base_seqs = prior.get("max_num_seqs") or 64
    pts: list[dict] = []
    for sk in viable:
        ceil = sk.get("kv_ceiling_ctx")
        mml_opts = [x for x in (ceil, prior.get("max_model_len")) if x]
        mml = min(mml_opts) if mml_opts else ceil
        for gmu in dict.fromkeys([base_gmu, 0.92]):
            for seqs in dict.fromkeys([base_seqs, 32]):
                pts.append({
                    "tp": sk["tp"], "quant": sk["quant"], "max_model_len": mml,
                    "gpu_memory_util": gmu, "max_num_seqs": seqs,
                    "max_num_batched_tokens": base_batched, "kv_cache_dtype": "auto",
                    "mtp": {"enabled": False},
                })
    if max_points:
        pts = pts[:max_points]
    return pts
