"""Resolve reusable scripts: bundled-by-default, config-overridable.

The mlops scripts (bench, probes, eval harnesses) ship *inside* the package
(src/johnny/scripts/) so a clean `pipx install` is self-contained — no dependency
on a particular machine's ~/.hermes or ~/vllm-tuning. A user may still point
`config.scripts.<key>` at their own copy to override the bundled one.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

# logical key -> bundled filename
_BUNDLED = {
    "bench": "bench.sh",
    "wait_ready": "wait-ready.sh",
    "push_ctx": "push-coder-ctx.sh",
    "probe_dtypes": "probe-wmma-opcodes.sh",
    "probe_mtp": "probe-mtp-availability.sh",
    "audit_models": "audit-models.py",
    "arc_eval": "arc_eval.py",
    "humaneval_score": "humaneval_chat_score.py",
    "code_needle": "code_needle.py",
    "bench_llamacpp": "bench_llamacpp.sh",     # llama.cpp client throughput (concurrency 1..32)
    "llama_bench": "bench_llama_speed.sh",     # llama.cpp single-stream prefill/decode (llama-bench)
}


def bundled_path(key: str) -> str | None:
    fn = _BUNDLED.get(key)
    if not fn:
        return None
    try:
        p = importlib.resources.files("johnny").joinpath("scripts", fn)
        return str(p) if p.is_file() else None
    except (ModuleNotFoundError, FileNotFoundError, AttributeError):
        return None


def resolve_script(key: str, cfg: dict | None = None) -> str | None:
    """An explicit `config.scripts.<key>` override (if it exists on disk) wins;
    otherwise the package-bundled copy. None if neither is available."""
    cfg = cfg or {}
    override = (cfg.get("scripts") or {}).get(key)
    if override:
        p = Path(override).expanduser()
        if p.exists():
            return str(p)
    return bundled_path(key)


def all_status(cfg: dict | None = None) -> dict[str, dict]:
    """For `doctor`: which scripts resolve, and from where."""
    out: dict[str, dict] = {}
    cfg = cfg or {}
    overrides = cfg.get("scripts") or {}
    for key in _BUNDLED:
        ov = overrides.get(key)
        if ov and Path(ov).expanduser().exists():
            out[key] = {"path": str(Path(ov).expanduser()), "source": "override"}
        elif bundled_path(key):
            out[key] = {"path": bundled_path(key), "source": "bundled"}
        else:
            out[key] = {"path": None, "source": "missing"}
    return out
