"""Backend drivers — the pluggability seam (§3.3).

A driver abstracts one inference engine (vLLM, LM Studio, Ollama) behind a common
interface. Semantics are shared; mechanics/knobs vary, so a driver *declares its
capabilities* and the engine/UI adapt. vLLM is driver #1; LM Studio is a read-only
spike here (full driver at P7); Ollama later.
"""

from __future__ import annotations

from .base import Capabilities, Driver, ModelInfo, SeatInfo

__all__ = ["Driver", "Capabilities", "SeatInfo", "ModelInfo", "get_driver", "available_drivers"]


def get_driver(name: str, **kw) -> Driver:
    if name == "vllm":
        from .vllm import VllmDriver

        return VllmDriver(**kw)
    if name == "lmstudio":
        from .lmstudio import LmStudioDriver

        return LmStudioDriver(**kw)
    if name == "llamacpp":
        from .llamacpp import LlamaCppDriver

        return LlamaCppDriver(**kw)
    raise ValueError(f"unknown backend: {name}")


def available_drivers() -> list[str]:
    """Backends whose CLI/runtime is present on this box."""
    out = []
    for name in ("vllm", "lmstudio", "llamacpp"):
        try:
            if get_driver(name).available():
                out.append(name)
        except Exception:
            pass
    return out
