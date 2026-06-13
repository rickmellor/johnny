"""Induction pipeline — resumable state machine (§3.6).

`plan()` is the launch-free preview (discover → audit → hardware-fit → seeded
points + KV-preflight). `run()` executes the tune sweep with per-point resumability
(state.json), pins the tuning seat against the reaper, synthesizes the winner per
use-case, and writes the placement + report.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import config as C
from ..engine import all_seats, load_config
from ..engine.placement import assign_gpus, free_gpus
from ..hardware import detect as hwd
from ..registry import store
from ..telemetry import collect
from . import grid, report, stages


def _state_dir(model_id: str) -> Path:
    return C.get_paths().state_dir / "induct" / model_id.replace("/", "__")


def _load_state(model_id: str) -> dict:
    p = _state_dir(model_id) / "state.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_state(model_id: str, st: dict) -> None:
    d = _state_dir(model_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(st, indent=2))


def plan(model_ref: str, max_points: int | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg if cfg is not None else load_config()
    hw = hwd.detect()
    model_id, path = stages.discover(model_ref, cfg)
    a = stages.audit(path)
    free = free_gpus(hw, all_seats(cfg))
    viable, pruned = stages.hardware_fit(a, hw, len(free))
    priors = grid.seed_priors(store.load(), model_id)
    points = grid.candidate_points(viable, priors, max_points=max_points)
    return {
        "model_id": model_id,
        "path": path,
        "audit": {
            "arch": a.get("arch"), "quant": a.get("quant"),
            "size_gb": round(a.get("size_bytes", 0) / 1e9, 1),
            "native_ctx": (a.get("dims") or {}).get("ctx"),
        },
        "free_gpus": free,
        "viable": viable,
        "pruned": pruned,
        "priors": len(priors),
        "points": points,
    }


def run(
    model_ref: str,
    use_case: str | None = None,
    bench: bool = False,
    resume: bool = False,
    max_points: int | None = None,
    cfg: dict | None = None,
) -> dict:
    cfg = cfg if cfg is not None else load_config()
    hw = hwd.detect()
    model_id, path = stages.discover(model_ref, cfg)

    st = _load_state(model_id) if resume else {}
    st.setdefault("model_id", model_id)
    st.setdefault("results", {})

    a = stages.audit(path)
    st["audit_summary"] = {"arch": a.get("arch"), "quant": a.get("quant"), "size_gb": round(a.get("size_bytes", 0) / 1e9, 1)}
    _save_state(model_id, st)

    viable, pruned = stages.hardware_fit(a, hw, len(free_gpus(hw, all_seats(cfg))))
    if not viable:
        return {"model_id": model_id, "error": "no viable placement on this hardware", "pruned": pruned}

    points = grid.candidate_points(viable, grid.seed_priors(store.load(), model_id), max_points=max_points)

    for point in points:
        sig = report._point_sig(point)
        if sig in st["results"]:  # resumable: skip already-benched points
            continue
        gpus = assign_gpus(point["tp"], hw, free_gpus(hw, all_seats(cfg)))
        if len(gpus) < point["tp"]:
            r = {"point": point, "ok": False, "error": f"insufficient free GPUs for tp={point['tp']}"}
        else:
            collect.add_pin(stages.TUNING_CONTAINER)  # reaper-safe for the run
            try:
                r = stages.tune_point(model_id, path, point, gpus, cfg, hw)
            finally:
                collect.remove_pin(stages.TUNING_CONTAINER)
        st["results"][sig] = r
        _save_state(model_id, st)

    results = list(st["results"].values())
    winner = stages.synthesize(results, use_case)
    run_dir = C.get_paths().runs_dir / f"induct-{model_id.replace('/', '__')}"
    report_path = report.write_report(run_dir, model_id, a, results, winner)

    placement = None
    if winner:
        rtv = ((cfg.get("docker") or {}).get("vllm_image", "") or "").split(":")[-1] or "unknown"
        placement = report.to_placement(model_id, winner, a, hw, rtv, use_case)
        report.write_placement(model_id, a, placement, hw)

    st["done"] = True
    _save_state(model_id, st)
    return {
        "model_id": model_id,
        "points": len(points),
        "results": results,
        "winner": winner,
        "placement_id": placement["id"] if placement else None,
        "report": str(report_path),
        # The quality harness (lm-eval/arc/humaneval/needle) is heavy + opt-in; its
        # orchestration is wired but a real run is the user's GPU-time call.
        "bench": "requested (run the quality harness explicitly)" if bench else "skipped (tuning-only default)",
    }
