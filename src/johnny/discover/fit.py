"""Fast pre-download fit verdict (the LM-Studio "traffic light").

Pre-download we know the weight size (HF metadata) but not the config dims, so this
is a weights-vs-VRAM check across TP options: fits / tight / won't-fit + the limiting
factor. The precise KV/context math (induct/grid.py) runs post-download with config.
"""

from __future__ import annotations

_QUANT_DTYPE = {"fp8": "fp8", "awq": "int4", "gptq": "int4", "int4": "int4", "bf16": "bf16", "fp4": "fp4"}
_GMU_CAP = 0.92
_OVERHEAD = 1.5e9
_KV_MIN = 2.0e9


def fit_verdict(size_bytes: int, hardware, quant: str | None = None) -> dict:
    nd = set(hardware.native_dtypes)
    need = _QUANT_DTYPE.get((quant or "").lower())
    vram = min((g.vram_gb for g in hardware.groups), default=0.0)
    ngpu = len(hardware.gpus)
    if not size_bytes:
        return {"verdict": "unknown", "detail": "size unavailable"}
    if need and nd and need not in nd:
        return {"verdict": "wont-fit", "best_tp": None,
                "detail": f"quant {quant} -> {need} not natively accelerated (have {sorted(nd)})"}
    if not ngpu or not vram:
        return {"verdict": "unknown", "detail": "no GPUs detected (CPU/LM Studio/Ollama backends only)"}
    budget = vram * 1e9 * _GMU_CAP - _OVERHEAD
    for tp in sorted({1, 2, 4, 8, ngpu} & set(range(1, ngpu + 1))):
        per = size_bytes / tp
        if per + _KV_MIN <= budget:
            frac = per / (vram * 1e9)
            return {"verdict": "tight" if frac > 0.75 else "fits", "best_tp": tp,
                    "per_gpu_gb": round(per / 1e9, 1),
                    "detail": f"TP{tp}: {per / 1e9:.1f} GB/GPU of {vram:.0f} GB"}
    return {"verdict": "wont-fit", "best_tp": None,
            "detail": f"{size_bytes / 1e9:.0f} GB won't fit across {ngpu}×{vram:.0f} GB even at TP{ngpu}"}
