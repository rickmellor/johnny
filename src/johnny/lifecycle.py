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


def _models_dir(cfg: dict | None = None) -> str | None:
    cfg = cfg if cfg is not None else (C.load_yaml(C.get_paths().config_file) or {})
    return (cfg.get("roots") or {}).get("models_dir")


def delete_path(path: str, cfg: dict | None = None) -> bool:
    """Delete a model dir — but only if it lives inside the configured models dir."""
    import shutil

    md = _models_dir(cfg)
    if not md:
        return False
    p = Path(path)
    try:
        p.resolve().relative_to(Path(md).expanduser().resolve())
    except (ValueError, OSError):
        return False
    shutil.rmtree(p, ignore_errors=True)
    return not p.exists()


def delete_untracked(candidate: dict) -> bool:
    """Delete an 'untracked' on-disk model dir. Returns True on success."""
    if candidate.get("kind") != "untracked" or not candidate.get("path"):
        return False
    return delete_path(candidate["path"])


def resolve_target(target: str, cfg: dict | None = None) -> dict | None:
    """Resolve a removal target (registry id, local_path, or on-disk path) to
    {model_id, local_path, path, size_gb}. None if it matches nothing."""
    cfg = cfg if cfg is not None else (C.load_yaml(C.get_paths().config_file) or {})
    md = _models_dir(cfg)
    reg = store.load()
    models = reg.get("models") or {}

    def _entry(model_id, local_path):
        path = None
        if md and local_path:
            cand = Path(md).expanduser() / local_path
            if cand.exists():
                path = str(cand)
        return {"model_id": model_id, "local_path": local_path, "path": path,
                "size_gb": round(grid.model_size_bytes(path) / 1e9, 1) if path else None}

    if target in models:
        return _entry(target, (models[target].get("identity") or {}).get("local_path"))
    for mid, m in models.items():
        if (m.get("identity") or {}).get("local_path") == target:
            return _entry(mid, target)
    p = Path(target).expanduser()
    if not p.is_absolute() and md:
        p = Path(md).expanduser() / target
    if p.exists():
        return {"model_id": None, "local_path": target, "path": str(p),
                "size_gb": round(grid.model_size_bytes(str(p)) / 1e9, 1)}
    return None


def running_seat_for(info: dict, cfg: dict | None = None) -> str | None:
    """If the target model is currently serving, return its seat name (so we refuse
    to delete a live model)."""
    from .engine import all_seats

    mid = info.get("model_id")
    if not mid:
        return None
    for s in all_seats(cfg):
        labels = (s.extra or {}).get("labels", {})
        if s.model == mid or labels.get("johnny.model") == mid:
            return s.name
    return None


def remove(info: dict, registry_only: bool = False, cfg: dict | None = None) -> dict:
    """Remove a single model: its on-disk weights (unless --registry-only) and/or its
    registry entry."""
    result = {"model_id": info.get("model_id"), "deleted_path": None, "deregistered": False}
    if not registry_only and info.get("path"):
        if delete_path(info["path"], cfg):
            result["deleted_path"] = info["path"]
    if info.get("model_id"):
        reg = store.load()
        if info["model_id"] in (reg.get("models") or {}):
            del reg["models"][info["model_id"]]
            store.save(reg)
            result["deregistered"] = True
    return result
