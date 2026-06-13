"""Lifecycle & cleanup (§3.8 / P8).

Surface removal candidates by combining three signals — on-disk models not in the
registry, registry models with no placement validated on *this* hardware, and stale
(not recently used) models. Read-only by default (`cleanup` dry-runs); deletion is
explicit + confirmed.
"""

from __future__ import annotations

from pathlib import Path

from . import config as C
from .backends.vllm import VllmDriver
from .hardware import detect as hwd
from .induct import grid
from .registry import store
from .telemetry import collect


def cleanup_candidates(cfg: dict | None = None, stale_days: int = 30) -> dict:
    cfg = cfg if cfg is not None else (C.load_yaml(C.get_paths().config_file) or {})
    models_dir = (cfg.get("roots") or {}).get("models_dir")
    hw = hwd.detect()
    reg = store.load()
    models = reg.get("models") or {}

    reg_paths = {}
    for mid, m in models.items():
        lp = (m.get("identity") or {}).get("local_path")
        if lp:
            reg_paths[lp] = mid

    cands: list[dict] = []

    # 1. On-disk models the registry doesn't track (register them, or reclaim disk).
    drv = VllmDriver(models_dir=models_dir)
    for mi in drv.list_local():
        if mi.id not in reg_paths:
            size = grid.model_size_bytes(mi.path)
            cands.append({
                "target": mi.id, "kind": "untracked", "path": mi.path,
                "size_gb": round(size / 1e9, 1), "reason": "on disk, not in registry",
            })

    # 2. Registry models with NO placement validated on this hardware fingerprint.
    for mid, m in models.items():
        pls = m.get("placements") or []
        if pls and all((p.get("validation_key") or {}).get("hardware_fingerprint") != hw.fingerprint for p in pls):
            cands.append({
                "target": mid, "kind": "unvalidated", "size_gb": None,
                "reason": f"no placement validated on {hw.fingerprint} (re-induct or remove)",
            })

    # 3. Stale: a tracked model not loaded/served within stale_days (weak signal).
    cutoff = collect.now() - stale_days * 86400
    for mid, m in models.items():
        life = m.get("lifecycle") or {}
        last = life.get("last_served") or life.get("last_loaded")
        if last and last < cutoff:
            cands.append({"target": mid, "kind": "stale", "size_gb": None,
                          "reason": f"not used since {last} (> {stale_days}d)"})

    return {"fingerprint": hw.fingerprint, "candidates": cands}


def delete_untracked(candidate: dict) -> bool:
    """Delete an 'untracked' on-disk model dir. Returns True on success."""
    if candidate.get("kind") != "untracked" or not candidate.get("path"):
        return False
    import shutil

    p = Path(candidate["path"])
    models_dir = Path((C.load_yaml(C.get_paths().config_file) or {}).get("roots", {}).get("models_dir", "")).expanduser()
    # safety: only delete inside the configured models dir
    try:
        p.resolve().relative_to(models_dir.resolve())
    except (ValueError, OSError):
        return False
    shutil.rmtree(p, ignore_errors=True)
    return not p.exists()
