"""Fast pre-download fit verdict (the LM-Studio "traffic light").

Pre-download we know the weight size (HF metadata) but not the config dims, so this
is a weights-vs-VRAM check across TP options: fits / tight / won't-fit + the limiting
factor. The precise KV/context math (induct/grid.py) runs post-download with config.
"""

from __future__ import annotations

# quant label -> the compute dtype it needs natively accelerated.
# Covers the modern llm-compressor / community labels (NVFP4, W4A16, FP8-Dynamic…).
_QUANT_DTYPE = {
    "fp8": "fp8", "w8a8": "fp8",
    "nvfp4": "fp4", "mxfp4": "fp4", "fp4": "fp4",
    "awq": "int4", "gptq": "int4", "w4a16": "int4", "int4": "int4", "4bit": "int4",
    "int8": "int8", "8bit": "int8",
    "bf16": "bf16", "fp16": "fp16",
}
_GMU_CAP = 0.92
_OVERHEAD = 1.5e9
_KV_MIN = 2.0e9


def quant_native_dtype(quant: str | None) -> str | None:
    """The compute dtype a quant needs accelerated, or None if unquantized/unknown."""
    return _QUANT_DTYPE.get((quant or "").lower())


def dtype_fit(quant: str | None, hardware) -> dict:
    """Does this quant's compute dtype run natively on the detected hardware?

    The 'fits my dtypes' check — distinct from the VRAM fit. Returns
    {ok: True|False|None, need: dtype|None, detail}. ok=None = unquantized / can't tell.
    """
    q = (quant or "").lower()
    if q == "gguf":
        return {"ok": False, "need": None, "detail": "GGUF format → llama.cpp/Ollama, not vLLM"}
    need = quant_native_dtype(quant)
    nd = set(hardware.native_dtypes)
    if not quant or need is None:
        return {"ok": None, "need": None, "detail": "unquantized / unknown quant"}
    if not nd:
        return {"ok": None, "need": need, "detail": f"need {need}; native dtypes undetected"}
    if need in nd:
        return {"ok": True, "need": need, "detail": f"{need} natively accelerated"}
    return {"ok": False, "need": need, "detail": f"{need} NOT native here (have {sorted(nd)})"}


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
