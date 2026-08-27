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
    # A role may map to different models in different profiles; the running fleet
    # decides which one wins. Profile file order only breaks the no-seat tie (the
    # absent estimate names the first-profile model, as before).
    hints = profiles.role_to_models(target) or [target]
    for hint in hints:
        for s in seats:
            labels = (s.extra or {}).get("labels", {})
            if s.model == hint or labels.get("johnny.model") == hint:
                return s, hint
    return None, hints[0]


def seat_guidance(target: str, model: str | None = None) -> str | None:
    """Optional agent-guidance hint declared on a profile seat, e.g. `guidance: roadmap`.

    Some seats execute delegated work well but plan it poorly — gemma-4-26B measured
    PlanBench 34% / AutomationBench 0-6.7% (long loops) yet 5/5 on short delegate tasks,
    where an explicit ordered brief cut it from 6.4 to 5.1 tool steps and kept the hardest
    task off the step ceiling (8/8 bare vs 5-6 roadmapped, 2026-08-24). Clients (input's
    spawn_agent) read this to decide whether to hand the sub-agent a step-by-step plan
    instead of an open-ended question. Declared per seat in profiles.yaml so it travels
    with the fleet config rather than being hard-coded per model in every client.
    """
    from . import profiles as _profiles

    # Guidance is a property of the MODEL a profile seat runs, not of the role name: the same
    # role ('chat') is served by different models across profiles, and matching on the role
    # leaked gemma-tp4's `roadmap` onto the Qwen3.5-122B seat (2026-08-27). When the caller
    # knows which model is live, only that model's seat entries count; the role/alias match is
    # kept solely for the absent-seat case (no live model to key on).
    for prof in (_profiles.all_profiles() or {}).values():
        for seat in prof.get("seats") or []:
            g = seat.get("guidance")
            if not g:
                continue
            if model:
                if model == seat.get("model"):
                    return g
                continue
            role = seat.get("role")
            aliased = (prof.get("role_aliases") or {}).get(target)
            if target in (role, seat.get("model")) or (aliased and aliased == role):
                return g
    return None


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
            "guidance": seat_guidance(target, model_hint),
        }
    state = "ready" if seat.state == "ready" else ("loading" if seat.state in ("loading", "running") else "failed")
    queue_depth = None
    eta = None
    if state == "ready" and seat.port:
        # queue depth is advisory; never let a slow /metrics stall the hot path
        queue_depth = sources.metrics_for_port(seat.port, timeout=0.3).get("waiting")
    elif state == "loading":
        eta = collect.cold_start_estimate(seat.model or model_hint or target)
    return {
        "seat": seat.name,
        "endpoint": _endpoint(cfg, seat),
        "model": seat.model,
        "state": state,
        "eta_s": eta,
        "queue_depth": queue_depth,
        "guidance": seat_guidance(target, seat.model),
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
