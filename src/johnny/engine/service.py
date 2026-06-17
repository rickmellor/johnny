"""resolve + reap — the read-side request-plane primitive and the idle reaper.

`resolve` is the focused hot-path projection SAINT calls per dispatch (§3.13):
where is this role/seat, what model, is it ready. `reap` is the stateless one-shot
the cron/timer runs: down idle-past-TTL unpinned seats so the cards reach deep idle.
"""

from __future__ import annotations

import datetime

from ..telemetry import collect, sources
from ..util import run
from . import all_seats, driver_for, load_config
from . import profiles


def _endpoint(cfg: dict, seat) -> str | None:
    bind = (cfg.get("network") or {}).get("bind_address", "127.0.0.1")
    return f"http://{bind}:{seat.port}/v1" if seat.port else None


def _find(seats, target):
    for s in seats:
        if s.name == target:
            return s, None
    model_hint = profiles.role_to_model(target) or target
    for s in seats:
        labels = (s.extra or {}).get("labels", {})
        if s.model == model_hint or labels.get("johnny.model") == model_hint:
            return s, model_hint
    return None, model_hint


def resolve(target: str, cfg: dict | None = None) -> dict:
    cfg = cfg if cfg is not None else load_config()
    seats = all_seats(cfg)
    seat, model_hint = _find(seats, target)
    if not seat:
        return {
            "seat": None,
            "endpoint": None,
            "model": model_hint or target,
            "state": "absent",
            "eta_s": collect.cold_start_estimate(model_hint or target),
            "queue_depth": None,
        }
    state = "ready" if seat.state == "ready" else ("loading" if seat.state in ("loading", "running") else "failed")
    queue_depth = None
    eta = None
    if state == "ready" and seat.port:
        queue_depth = sources.metrics_for_port(seat.port).get("waiting")
    elif state == "loading":
        eta = collect.cold_start_estimate(seat.model or model_hint or target)
    return {
        "seat": seat.name,
        "endpoint": _endpoint(cfg, seat),
        "model": seat.model,
        "state": state,
        "eta_s": eta,
        "queue_depth": queue_depth,
    }


def ready_chat_seats(cfg: dict | None = None) -> list:
    """Ready, non-embeddings seats — what `alive` falls back to when a role doesn't
    resolve (no profile). A seat is embeddings if its served model has a pooling
    placement in the registry; an unknown model is treated as chat-capable."""
    cfg = cfg if cfg is not None else load_config()
    from ..registry import store

    reg = store.load()
    out = []
    for s in all_seats(cfg):
        if s.state != "ready" or not s.port:
            continue
        m = store.get(reg, s.model) if s.model else None
        is_emb = bool(m) and any(
            (p.get("extra") or {}).get("runner") == "pooling" for p in (m.get("placements") or [])
        )
        if not is_emb:
            out.append(s)
    return out


def _container_started_epoch(name: str) -> int | None:
    rc, out, _ = run(["docker", "inspect", "-f", "{{.State.StartedAt}}", name], timeout=8)
    if rc != 0 or not out.strip():
        return None
    try:
        base = out.strip().split(".")[0].rstrip("Z")
        dt = datetime.datetime.fromisoformat(base).replace(tzinfo=datetime.timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return None


def reap(idle_ttl: int | None = None, dry_run: bool = False, cfg: dict | None = None) -> list[dict]:
    cfg = cfg if cfg is not None else load_config()
    try:
        collect.ingest_spool()  # pick up pushed activity before judging idleness
    except Exception:
        pass
    ttl = int(idle_ttl if idle_ttl is not None else (cfg.get("reaper") or {}).get("idle_ttl_s", 1800))
    n = collect.now()
    actions: list[dict] = []
    for s in all_seats(cfg):
        if collect.is_pinned(s.name, n):
            actions.append({"seat": s.name, "action": "skip", "reason": "pinned"})
            continue
        if s.port and s.state == "ready":
            if (sources.metrics_for_port(s.port).get("running") or 0) > 0:
                actions.append({"seat": s.name, "action": "skip", "reason": "busy (running>0)"})
                continue
        la = collect.last_activity(s.name)
        if la is None:
            la = _container_started_epoch(s.name)
        idle = n - (la or n)
        if idle > ttl:
            if not dry_run:
                drv = driver_for(s, cfg)
                if drv:
                    drv.stop(s.name)
            actions.append({"seat": s.name, "action": "would-reap" if dry_run else "reap", "idle_s": idle, "ttl_s": ttl})
        else:
            actions.append({"seat": s.name, "action": "keep", "idle_s": idle, "ttl_s": ttl})
    return actions
