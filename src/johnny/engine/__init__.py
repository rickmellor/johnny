"""Engine — the shared library all front-ends call (§3.5).

Aggregates backend drivers, derives the live fleet view, and exposes placement,
launch/down/swap, resolve, and the reaper. No front-end logic lives here; the CLI,
daemon, TUI, and router are thin clients over this.
"""

from __future__ import annotations

from .. import config as C
from ..backends import get_driver


def load_config() -> dict:
    return C.load_yaml(C.get_paths().config_file) or {}


def drivers_from_config(cfg: dict | None = None) -> dict:
    cfg = cfg if cfg is not None else load_config()
    roots = cfg.get("roots") or {}
    docker = cfg.get("docker") or {}
    enabled = (cfg.get("backends") or {}).get("enabled") or ["vllm"]
    drivers: dict = {}
    for name in enabled:
        try:
            if name == "vllm":
                drivers[name] = get_driver("vllm", models_dir=roots.get("models_dir"), image=docker.get("vllm_image"))
            elif name == "llamacpp":
                drivers[name] = get_driver("llamacpp", models_dir=roots.get("models_dir"), image=docker.get("llamacpp_image"))
            else:
                drivers[name] = get_driver(name)
        except Exception:
            continue
    return drivers


def all_seats(cfg: dict | None = None) -> list:
    """Live fleet view across all available backends."""
    seats = []
    for drv in drivers_from_config(cfg).values():
        try:
            if drv.available():
                seats.extend(drv.runtime_state())
        except Exception:
            continue
    return seats


def driver_for(seat, cfg: dict | None = None):
    return drivers_from_config(cfg).get(getattr(seat, "backend", None))
