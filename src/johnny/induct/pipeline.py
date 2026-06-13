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
    progress=None,
) -> dict:
    _p = progress or (lambda *_: None)
    cfg = cfg if cfg is not None else load_config()
    hw = hwd.detect()
    model_id, path = stages.discover(model_ref, cfg)

    st = _load_state(model_id) if resume else {}
    st.setdefault("model_id", model_id)
    st.setdefault("results", {})

    a = stages.audit(path)
    st["audit_summary"] = {"arch": a.get("arch"), "quant": a.get("quant"), "size_gb": round(a.get("size_bytes", 0) / 1e9, 1)}
    _save_state(model_id, st)
    _p(f"audit: {a.get('arch')} · {round(a.get('size_bytes', 0) / 1e9, 1)}GB · quant={a.get('quant')}")

    viable, pruned = stages.hardware_fit(a, hw, len(free_gpus(hw, all_seats(cfg))))
    _p(f"hardware-fit: {len(viable)} viable, {len(pruned)} pruned")
    if not viable:
        return {"model_id": model_id, "error": "no viable placement on this hardware", "pruned": pruned}

    points = grid.candidate_points(viable, grid.seed_priors(store.load(), model_id), max_points=max_points)
    done = sum(1 for p in points if report._point_sig(p) in st["results"])
    _p(f"sweep: {len(points)} point(s)" + (f" ({done} already done, resuming)" if done else ""))

    for i, point in enumerate(points, 1):
        sig = report._point_sig(point)
        if sig in st["results"]:  # resumable: skip already-benched points
            _p(f"[{i}/{len(points)}] {sig}: cached, skipping")
            continue
        gpus = assign_gpus(point["tp"], hw, free_gpus(hw, all_seats(cfg)))
        if len(gpus) < point["tp"]:
            r = {"point": point, "ok": False, "error": f"insufficient free GPUs for tp={point['tp']}"}
            _p(f"[{i}/{len(points)}] {sig}: skipped (insufficient free GPUs)")
        else:
            _p(f"[{i}/{len(points)}] {sig}: launching on GPU {gpus} + benching…")
            collect.add_pin(stages.TUNING_CONTAINER)  # reaper-safe for the run
            try:
                r = stages.tune_point(model_id, path, point, gpus, cfg, hw)
            finally:
                collect.remove_pin(stages.TUNING_CONTAINER)
            if r.get("ok"):
                _p(f"[{i}/{len(points)}] {sig}: peak {r.get('peak_tok_s')} tok/s, single {r.get('single_tok_s')} tok/s")
            else:
                _p(f"[{i}/{len(points)}] {sig}: FAILED — {(r.get('error') or '')[:80]}")
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
        md = (cfg.get("roots") or {}).get("models_dir")
        try:
            lp = str(Path(path).relative_to(Path(md).expanduser())) if md else None
        except (ValueError, TypeError):
            lp = None
        report.write_placement(model_id, a, placement, hw, local_path=lp)

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
