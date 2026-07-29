"""Fast pre-download fit verdict (the LM-Studio "traffic light").

Pre-download we know the weight size (HF metadata) but not the config dims, so this
is a weights-vs-memory check: fits / tight / offload / won't-fit + the limiting
factor. The precise KV/context math (induct/grid.py) runs post-download with config.

Backend-aware: safetensors repos fit against per-GPU VRAM across vLLM TP options;
GGUF repos fit against total VRAM + host RAM, because the llamacpp backend layer-splits
across all GPUs and can push MoE experts to RAM (--override-tensor / --n-cpu-moe) —
"bigger than VRAM" is a speed tier there, not a wall.
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

# llamacpp path: per-GPU reserve for KV cache + compute buffers (matches the validated
# 27-of-32 GB weight target in induct/llamacpp.py) and the RAM fraction experts may
# spill into (matches induct/grid.cpu_viable headroom).
_LCPP_RESERVE_GB = 5.0
_LCPP_RAM_HEADROOM = 0.7


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
        # llama.cpp dequantizes in-kernel, so GGUF has no native-dtype requirement.
        return {"ok": None, "need": None, "detail": "GGUF → llamacpp backend (dtype-agnostic), not vLLM"}
    need = quant_native_dtype(quant)
    nd = set(hardware.native_dtypes)
    if not quant or need is None:
        return {"ok": None, "need": None, "detail": "unquantized / unknown quant"}
    if not nd:
        return {"ok": None, "need": need, "detail": f"need {need}; native dtypes undetected"}
    if need in nd:
        return {"ok": True, "need": need, "detail": f"{need} natively accelerated"}
    return {"ok": False, "need": need, "detail": f"{need} NOT native here (have {sorted(nd)})"}


def gguf_fit_verdict(size_bytes: int, hardware) -> dict:
    """llamacpp-backend fit: weights layer-split over ALL GPUs (no TP constraint),
    overflow expert-offloaded to host RAM. Only past VRAM+RAM is it a wont-fit."""
    gb = size_bytes / 1e9
    ngpu = len(hardware.gpus)
    vram_total = sum(g.vram_gb for g in hardware.gpus)
    ram_budget = (hardware.host_ram_gb or 0.0) * _LCPP_RAM_HEADROOM
    if not ngpu or not vram_total:
        if gb <= ram_budget:
            return {"verdict": "fits", "backend": "llamacpp", "device": "cpu",
                    "detail": f"llama.cpp CPU: {gb:.0f} GB in {hardware.host_ram_gb:.0f} GB RAM"}
        return {"verdict": "wont-fit", "backend": "llamacpp",
                "detail": f"{gb:.0f} GB exceeds ~{ram_budget:.0f} GB RAM budget (no GPUs)"}
    gpu_budget = max(0.0, vram_total - _LCPP_RESERVE_GB * ngpu)
    if gb <= gpu_budget:
        frac = gb / vram_total
        return {"verdict": "tight" if frac > 0.75 else "fits", "backend": "llamacpp",
                "gpu_count": ngpu, "detail": f"llama.cpp: {gb:.0f} GB across {ngpu} GPU ({vram_total:.0f} GB VRAM)"}
    if gb <= gpu_budget + ram_budget:
        off = gb - gpu_budget
        return {"verdict": "offload", "backend": "llamacpp", "gpu_count": ngpu,
                "offload_gb": round(off, 1),
                "detail": f"llama.cpp expert-offload: ~{off:.0f} GB → RAM, {gpu_budget:.0f} GB on {ngpu} GPU (slower decode)"}
    return {"verdict": "wont-fit", "backend": "llamacpp",
            "detail": f"{gb:.0f} GB exceeds {vram_total:.0f} GB VRAM + ~{ram_budget:.0f} GB RAM budget"}


def fit_verdict(size_bytes: int, hardware, quant: str | None = None, gguf: bool = False) -> dict:
    nd = set(hardware.native_dtypes)
    need = _QUANT_DTYPE.get((quant or "").lower())
    vram = min((g.vram_gb for g in hardware.groups), default=0.0)
    ngpu = len(hardware.gpus)
    if not size_bytes:
        return {"verdict": "unknown", "detail": "size unavailable"}
    if gguf or (quant or "").lower() == "gguf":
        return gguf_fit_verdict(size_bytes, hardware)
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
