"""Registry load/save/merge — thin wrapper over the YAML file."""

from __future__ import annotations

from pathlib import Path

from .. import config as C


def load(path: Path | None = None) -> dict:
    paths = C.get_paths()
    p = Path(path) if path else paths.registry_file
    return C.load_yaml(p) or C.registry_stub()


def save(reg: dict, path: Path | None = None) -> None:
    paths = C.get_paths()
    p = Path(path) if path else paths.registry_file
    reg.setdefault("schema_version", C.REGISTRY_SCHEMA_VERSION)
    C.write_yaml(
        p,
        reg,
        header=f"# johnny registry — schema v{C.REGISTRY_SCHEMA_VERSION} (machine-written; `registry import`/induction)",
    )


def models(reg: dict) -> dict:
    return reg.get("models") or {}


def get(reg: dict, model_id: str) -> dict | None:
    return models(reg).get(model_id)


def delete_placement(reg: dict, model_id: str, placement_id: str) -> dict | None:
    """Remove one placement (by exact id) from a model, mutating reg. Returns the
    removed placement, or None if the model/placement wasn't found."""
    m = get(reg, model_id)
    if not m:
        return None
    pls = m.get("placements") or []
    for i, p in enumerate(pls):
        if p.get("id") == placement_id:
            return pls.pop(i)
    return None


def merge_imported(existing: dict, imported: dict) -> dict:
    """Overlay imported models onto an existing registry (imported entries win;
    other models are preserved). Fingerprints are unioned."""
    out = dict(existing or C.registry_stub())
    out.setdefault("models", {})
    out["models"].update(imported.get("models", {}))
    fps = set(out.get("fingerprints") or []) | set(imported.get("fingerprints") or [])
    out["fingerprints"] = sorted(fps)
    out["schema_version"] = C.REGISTRY_SCHEMA_VERSION
    return out
