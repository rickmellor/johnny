"""Sync a chat tool's provider config from the registry (§3.9).

Compute the provider's `base_url` (primary seat) + `models:` catalog (served-name →
context_length) from the registry, and patch *only* that provider block in the chat
tool's config — never a blind rewrite. Default is a preview; `--write` patches in
place with a timestamped backup.
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path

import yaml

from ..engine import load_config
from ..registry import store


def hermes_config_path() -> Path:
    return Path.home() / ".hermes" / "config.yaml"


def compute_block(provider_name: str, cfg: dict | None = None) -> dict:
    cfg = cfg if cfg is not None else load_config()
    reg = store.load()
    net = cfg.get("network") or {}
    host = net.get("advertise_host") or "127.0.0.1"
    if host == "auto":
        host = "127.0.0.1"
    base_port = (net.get("ports") or {}).get("base", 8000)
    models: dict = {}
    for mid, m in (reg.get("models") or {}).items():
        ctx = (m.get("capabilities") or {}).get("native_context")
        if not ctx:
            mmls = [(p.get("knobs") or {}).get("max_model_len") for p in m.get("placements", [])]
            mmls = [x for x in mmls if x]
            ctx = max(mmls) if mmls else None
        if ctx:
            models[mid] = {"context_length": ctx}
    return {"name": provider_name, "base_url": f"http://{host}:{base_port}/v1", "models": models}


def _patch(data, name: str, block: dict) -> bool:
    """Find a list item dict whose name==name anywhere in the structure; patch it in place."""
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("name") == name:
                item["base_url"] = block["base_url"]
                item["models"] = block["models"]
                return True
            if _patch(item, name, block):
                return True
    elif isinstance(data, dict):
        for v in data.values():
            if _patch(v, name, block):
                return True
    return False


def sync(provider_name: str | None = None, write: bool = False, cfg: dict | None = None) -> dict:
    cfg = cfg if cfg is not None else load_config()
    ext = cfg.get("external") or {}
    provider_name = provider_name or ext.get("provider") or "johnny"
    block = compute_block(provider_name, cfg)
    path = Path(ext["config_path"]).expanduser() if ext.get("config_path") else hermes_config_path()
    if not write:
        return {"path": str(path), "block": block, "written": False}
    if not path.exists():
        return {"error": f"no chat-tool config at {path}"}
    data = yaml.safe_load(path.read_text())
    if not _patch(data, provider_name, block):
        return {"error": f"provider '{provider_name}' not found in {path}"}
    bak = path.with_name(path.name + f".bak-{datetime.datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(path, bak)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return {"path": str(path), "backup": str(bak), "written": True, "block": block}
