"""Launch engine: build_spec + up / down / swap (§3.5).

Mutations take the lock so two `up`s can't double-claim GPUs/ports. The engine
computes *what* and *where*; the driver renders *how* (compose) and runs it. Teardown
is always a single named seat — never "all" — with a guard that errors rather than
evicting a sibling.
"""

from __future__ import annotations

from .. import config as C
from ..backends import get_driver
from ..backends.base import SeatInfo
from ..runtime import probe
from ..runtime.lock import mutation_lock
from ..telemetry import collect
from . import all_seats, driver_for, load_config
from .placement import allocate_port, assign_gpus, free_gpus, pick_placement, role_for, viable
from ..hardware import detect as hwdetect
from ..registry import store


class PlacementError(RuntimeError):
    pass


def build_spec(model_id: str, model: dict, placement: dict, gpus: list[int], port: int, cfg: dict, hardware) -> dict:
    roots = cfg.get("roots") or {}
    docker = cfg.get("docker") or {}
    ident = model.get("identity") or {}
    local_path = ident.get("local_path")
    image = placement.get("image") or docker.get("vllm_image")
    visible_env = "HIP_VISIBLE_DEVICES" if (hardware and hardware.vendor == "amd") else "CUDA_VISIBLE_DEVICES"
    return {
        "container_name": f"johnny-{model_id}-{port}",
        "image": image,
        "served_model_name": model_id,
        "model_path": f"/models/{local_path}" if local_path else None,
        "models_dir": roots.get("models_dir"),
        "vllm_cache": roots.get("vllm_cache"),
        "port": port,
        "bind_address": (cfg.get("network") or {}).get("bind_address", "127.0.0.1"),
        "gpus": gpus,
        "visible_env": visible_env,
        "shm_size": docker.get("shm_size", "16g"),
        "knobs": placement.get("knobs") or {},
        "extra": placement.get("extra") or {},
        "env": placement.get("env") or {},
        "labels": {
            "johnny.managed": "1",
            "johnny.model": model_id,
            "johnny.placement": placement.get("id", ""),
            "johnny.port": str(port),
            "johnny.gpus": " ".join(str(g) for g in gpus),
            "johnny.use_case": str(placement.get("use_case") or ""),
        },
    }


def _up_lmstudio(model_id: str, model: dict, placement: dict, cfg: dict) -> dict:
    """LM Studio launch: it owns GPU placement + a shared server port, so we skip
    GPU assignment/port allocation and delegate to `lms load` via the driver."""
    drv = get_driver("lmstudio")
    if not drv.available():
        raise PlacementError("LM Studio (`lms`) is not available on this box")
    ident = model.get("identity") or {}
    knobs = placement.get("knobs") or {}
    extra = placement.get("extra") or {}
    spec = {
        "model_key": ident.get("local_path") or ident.get("repo_id") or model_id,
        "identifier": model_id,
        "context_length": knobs.get("max_model_len"),
        "gpu_offload": knobs.get("gpu_offload") or extra.get("gpu_offload") or "max",
        "ttl": extra.get("ttl"),
        "port": (cfg.get("lmstudio") or {}).get("port", 1234),
    }
    seat = drv.launch(spec)
    collect.record_activity(seat.name)
    return {"action": "launched", "seat": seat.name, "port": seat.port, "gpus": [],
            "model": model_id, "placement": placement.get("id"), "state": "loading", "backend": "lmstudio"}


def _find_seat(seats, name_or_model: str):
    for s in seats:
        if s.name == name_or_model:
            return s
    for s in seats:
        labels = (s.extra or {}).get("labels", {})
        if s.model == name_or_model or labels.get("johnny.model") == name_or_model:
            return s
    return None


def up(
    model_id: str,
    placement_id: str | None = None,
    port: int | None = None,
    swap: str | None = None,
    force: bool = False,
    wait: bool = False,
    wait_timeout: float = 600.0,
) -> dict:
    cfg = load_config()
    hardware = hwdetect.detect()
    reg = store.load()
    model = store.get(reg, model_id)
    if not model:
        raise PlacementError(f"no model '{model_id}' in the registry (try `johnny registry import`)")
    placement = pick_placement(model.get("placements") or [], placement_id, hardware)
    if not placement:
        raise PlacementError(f"no placement for '{model_id}'" + (f" with id {placement_id}" if placement_id else ""))

    with mutation_lock():
        seats = all_seats(cfg)
        existing = _find_seat(seats, model_id)
        if existing and not swap and not force:
            return {"action": "exists", "seat": existing.name, "port": existing.port, "state": existing.state, "model": model_id}

        if (placement.get("backend") or "vllm") == "lmstudio":
            return _up_lmstudio(model_id, model, placement, cfg)

        if swap:
            target = _find_seat(seats, swap)
            if not target:
                raise PlacementError(f"--swap target '{swap}' is not running")
            drv = driver_for(target, cfg)
            if drv:
                drv.stop(target.name)
            if port is None:
                port = target.port
            seats = [s for s in seats if s.name != target.name]

        knobs = placement.get("knobs") or {}
        gc = knobs.get("gpu_count") or 0
        free = free_gpus(hardware, seats)
        ok, reason = viable(knobs, hardware, free)
        if not ok and not force:
            raise PlacementError(
                f"cannot place '{model_id}': {reason}. Pass --swap <seat> to free GPUs, or --force."
            )
        gpus = assign_gpus(gc, hardware, free)
        if port is None:
            port = allocate_port(cfg, seats, role=role_for(placement))

        spec = build_spec(model_id, model, placement, gpus, port, cfg, hardware)
        drv = get_driver("vllm", models_dir=(cfg.get("roots") or {}).get("models_dir"), image=spec["image"])
        started = collect.now()
        seat = drv.launch(spec)
        collect.record_activity(seat.name, ts=started)
        collect.record_load_event(seat.name, model_id, placement.get("id", ""), started, None)

    result = {"action": "launched", "seat": seat.name, "port": port, "gpus": gpus,
              "model": model_id, "placement": placement.get("id"), "state": "loading"}
    if wait:
        if _wait_ready(port, wait_timeout):
            collect.record_load_event(seat.name, model_id, placement.get("id", ""), started, collect.now())
            result["state"] = "ready"
            result["endpoint"] = f"http://{spec['bind_address']}:{port}/v1"
        else:
            result["state"] = "loading"
            result["eta_s"] = collect.cold_start_estimate(model_id, placement.get("id"))
    return result


def down(seat_name: str, drain: bool = False) -> dict:
    cfg = load_config()
    with mutation_lock():
        seats = all_seats(cfg)
        target = _find_seat(seats, seat_name)
        if not target:
            raise PlacementError(f"no running seat '{seat_name}'")
        if drain:
            # vLLM has no drain mode; without a router to stop admission this no-ops.
            pass
        drv = driver_for(target, cfg)
        if not drv:
            raise PlacementError(f"no driver for backend '{target.backend}'")
        drv.stop(target.name)
        collect.remove_pin(target.name)
    return {"action": "down", "seat": target.name, "drain": drain}


def swap(seat_name: str, model_id: str, wait: bool = False) -> dict:
    return up(model_id, swap=seat_name, wait=wait)


def _wait_ready(port: int, timeout: float) -> bool:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if probe.probe_models(port):
            return True
        time.sleep(3)
    return False
