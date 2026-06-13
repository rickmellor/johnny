"""Schema versioning + forward migrations for the files johnny owns.

Every owned YAML file carries a `schema_version`. When the tool's current
version is newer than a file's, `johnny migrate` makes a timestamped backup,
applies the registered forward migrations in order, and stamps the new version.
A file that's *newer* than the tool is left untouched (with a warning) — never
silently downgraded.

This ships from P0 so the first adopters aren't stranded by a v0.2 format change.
v1 is the baseline; the only registered migration is v0->v1 (adopt a file that
predates schema versioning).
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import yaml

from . import config as C


def _stamp_v1(data: dict) -> dict:
    """v0 -> v1: a file that predates schema versioning becomes a v1 file."""
    data = dict(data or {})
    data["schema_version"] = 1
    return data


# kind -> ordered [(target_version, migrate_fn)]; applied for cur < target_version <= goal.
MIGRATIONS: dict[str, list[tuple[int, callable]]] = {
    "config": [(1, _stamp_v1)],
    "registry": [(1, _stamp_v1)],
    "profiles": [(1, _stamp_v1)],
}

CURRENT: dict[str, int] = {
    "config": C.CONFIG_SCHEMA_VERSION,
    "registry": C.REGISTRY_SCHEMA_VERSION,
    "profiles": C.PROFILES_SCHEMA_VERSION,
}


def owned_files(paths: C.Paths) -> list[tuple[str, Path]]:
    return [
        ("config", paths.config_file),
        ("registry", paths.registry_file),
        ("profiles", paths.profiles_file),
    ]


def _file_status(kind: str, path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"kind": kind, "path": str(p), "exists": False}
    data = yaml.safe_load(p.read_text()) or {}
    return {
        "kind": kind,
        "path": str(p),
        "exists": True,
        "version": int(data.get("schema_version", 0)),
        "target": CURRENT[kind],
        "_data": data,
    }


def migrate_file(kind: str, path: Path, dry_run: bool = False) -> dict:
    """Migrate one owned file toward the current schema. Returns a result dict
    with an `action`: absent | up-to-date | newer-than-tool | would-migrate | migrated.
    """
    st = _file_status(kind, path)
    if not st["exists"]:
        return {**st, "action": "absent"}

    cur, target = st["version"], st["target"]
    if cur == target:
        return {**st, "action": "up-to-date"}
    if cur > target:
        return {**st, "action": "newer-than-tool"}
    if dry_run:
        return {**st, "action": "would-migrate"}

    p = Path(path)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = p.with_name(f"{p.name}.bak-v{cur}-{ts}")
    shutil.copy2(p, backup)

    data = st["_data"]
    for tv, fn in MIGRATIONS[kind]:
        if cur < tv <= target:
            data = fn(data)
    data["schema_version"] = target
    p.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
    return {**st, "action": "migrated", "backup": str(backup), "new_version": target}


def migrate_all(paths: C.Paths, dry_run: bool = False) -> list[dict]:
    return [migrate_file(kind, path, dry_run=dry_run) for kind, path in owned_files(paths)]
