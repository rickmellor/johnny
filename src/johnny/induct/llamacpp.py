"""llama.cpp (GGUF) induction — self-contained backend path.

The vLLM induction pipeline is TP/VRAM/arch-registry shaped and can't model a GGUF
seat (layer-offload, not tensor-parallel; a single file, not a config.json dir). This
module mirrors the same shape — discover → audit → small knob sweep → tune (launch a
throwaway `llama-server` seat, run the backend-agnostic bench.sh, tear down) →
synthesize → write placement — but for the llamacpp backend. pipeline.plan()/run()
dispatch here when the ref resolves to a GGUF, leaving the vLLM path untouched.

Reuses the generic helpers from `stages` (_wait_ready) and `report` (write_placement),
and the bundled bench.sh (hits /v1/completions — works against llama-server).
"""

from __future__ import annotations

import re
from pathlib import Path

from .. import config as C
from ..backends import get_driver
from ..engine import all_seats, load_config
from ..engine.placement import assign_gpus, free_gpus
from ..hardware import detect as hwd
from ..registry import store
from ..telemetry import collect
from . import report, stages

TUNING_CONTAINER = "llamacpp-johnny-tuning"
TUNING_PORT = 9001  # distinct from the vLLM tuning port (9000)

_SHARD_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d+)-of-(?P<tot>\d+)\.gguf$")


# --- discovery ---------------------------------------------------------------
def gguf_ref(model_ref: str, cfg: dict) -> tuple[str, str] | None:
    """(model_id, gguf_path) if model_ref points at a GGUF, else None.

    Accepts: a direct .gguf path, a .gguf under models_dir, or a registry model id
    whose identity.local_path is a .gguf / that has a llamacpp placement.
    """
    md = (cfg.get("roots") or {}).get("models_dir")

    def _mid(p: Path) -> str:
        name = p.name
        m = _SHARD_RE.match(name)
        stem = m.group("base") if m else p.stem
        return stem

    p = Path(model_ref).expanduser()
    if p.suffix == ".gguf" and p.exists():
        return _mid(p), str(p.resolve())
    if md:
        cand = Path(md).expanduser() / model_ref
        if cand.suffix == ".gguf" and cand.exists():
            return _mid(cand), str(cand.resolve())

    reg = store.load()
    m = (reg.get("models") or {}).get(model_ref)
    if m:
        ident = m.get("identity") or {}
        lp = ident.get("local_path") or ""
        is_llama = any((pl.get("backend") == "llamacpp") for pl in (m.get("placements") or []))
        if (lp.endswith(".gguf") or is_llama) and md and lp:
            full = Path(md).expanduser() / lp
            if full.exists():
                return model_ref, str(full.resolve())
    return None


def _shard_size_bytes(path: str) -> int:
    """Sum this quant's shards (…-NNNNN-of-MMMMM.gguf); else the single file size."""
    p = Path(path)
    m = _SHARD_RE.match(p.name)
    if not m:
        return p.stat().st_size if p.exists() else 0
    base, tot = m.group("base"), m.group("tot")
    total = 0
    for shard in p.parent.glob(f"{base}-*-of-{tot}.gguf"):
        try:
            total += shard.stat().st_size
        except OSError:
            pass
    return total


def audit(path: str) -> dict:
    """GGUF audit via the llamacpp driver's header probe + shard size."""
    info = get_driver("llamacpp").probe_model(path)
    info["size_bytes"] = _shard_size_bytes(path)
    return info


# --- candidate grid ----------------------------------------------------------
def _gpu_count_for(size_bytes: int, total_gpu_count: int, per_gpu_gb: float = 28.0) -> int:
    """How many GPUs the weights need (fit at ~per_gpu_gb usable), capped to the box's
    total GPU count. Independent of what's currently free (that gates the *tune*, below)."""
    size_gb = size_bytes / 1e9
    need = max(1, -(-int(size_gb) // int(per_gpu_gb)))  # ceil
    return max(1, min(need, total_gpu_count or 1))


def candidate_points(audit_info: dict, gpu_count: int, max_points: int | None) -> list[dict]:
    """Small, meaningful llama-server sweep. Model fits all-GPU here, so we sweep the
    real throughput knob (`--parallel`, the concurrent-slot count) at a working ctx."""
    native = audit_info.get("native_context") or 32768
    ctx = min(32768, native)
    pts = [
        {"backend": "llamacpp", "gpu_count": gpu_count, "n_gpu_layers": 999,
         "flash_attn": "off", "max_model_len": ctx, "parallel": 4},
        {"backend": "llamacpp", "gpu_count": gpu_count, "n_gpu_layers": 999,
         "flash_attn": "off", "max_model_len": ctx, "parallel": 16},
    ]
    if max_points:
        pts = pts[:max_points]
    return pts


def _point_sig(p: dict) -> str:
    return f"llama-ngl{p.get('n_gpu_layers')}-par{p.get('parallel')}-mml{p.get('max_model_len')}"


# --- tuning ------------------------------------------------------------------
def _tuning_spec(model_id: str, gguf_path: str, point: dict, gpus: list[int], cfg: dict, hardware) -> dict:
    roots = cfg.get("roots") or {}
    docker = cfg.get("docker") or {}
    md = roots.get("models_dir")
    model_path = (
        f"/models/{Path(gguf_path).relative_to(Path(md).expanduser())}"
        if md and str(gguf_path).startswith(str(Path(md).expanduser()))
        else gguf_path
    )
    visible_env = "HIP_VISIBLE_DEVICES" if (hardware and hardware.vendor == "amd") else "CUDA_VISIBLE_DEVICES"
    return {
        "container_name": TUNING_CONTAINER,
        "image": docker.get("llamacpp_image"),
        "served_model_name": model_id,
        "model_path": model_path,
        "models_dir": md,
        "port": TUNING_PORT,
        "bind_address": "127.0.0.1",
        "gpus": gpus,
        "visible_env": visible_env,
        "shm_size": docker.get("shm_size", "16g"),
        "knobs": {
            "n_gpu_layers": point.get("n_gpu_layers", 999),
            "flash_attn": point.get("flash_attn", "off"),
            "max_model_len": point.get("max_model_len"),
            "n_cpu_moe": point.get("n_cpu_moe"),
            "parallel": point.get("parallel"),
        },
        "extra": {"jinja": True, "override_tensor": point.get("override_tensor")},
        "env": {},
        "labels": {"johnny.tuning": "1", "johnny.backend": "llamacpp", "johnny.model": model_id},
    }


def tune_point(model_id: str, gguf_path: str, point: dict, gpus: list[int], cfg: dict, hardware) -> dict:
    drv = get_driver("llamacpp", image=(cfg.get("docker") or {}).get("llamacpp_image"))
    spec = _tuning_spec(model_id, gguf_path, point, gpus, cfg, hardware)
    result = {"point": point, "ok": False}
    try:
        drv.launch(spec)
        ready, why = stages._wait_ready(drv, TUNING_CONTAINER, TUNING_PORT, timeout=600)
        if not ready:
            logtail = stages._diagnose(drv, TUNING_CONTAINER)
            result["error"] = (why or "tuning seat not ready") + (f" — {logtail}" if logtail else "")
            return result
        from ..util import run

        # llama.cpp-appropriate bench (concurrency 1..32), not the vLLM 16..1024 sweep.
        scripts = cfg.get("scripts") or {}
        bench_script = scripts.get("bench_llamacpp") or str(
            Path(__file__).resolve().parent.parent / "scripts" / "bench_llamacpp.sh"
        )
        rc, out, errout = run(["bash", bench_script, str(TUNING_PORT), model_id], timeout=1800)
        parsed = stages._parse_bench(out)
        if parsed.get("peak_tok_s") is None:
            parsed["error"] = (errout or out)[-300:]
        result.update(parsed)
        result["ok"] = parsed.get("peak_tok_s") is not None
    finally:
        drv.stop(TUNING_CONTAINER)
    return result


# --- placement ---------------------------------------------------------------
def to_placement(model_id: str, winner: dict, audit_info: dict, hardware, runtime_version: str, use_case: str | None) -> dict:
    p = winner["point"]
    return {
        "id": f"induct-{_point_sig(p)}",
        "backend": "llamacpp",
        "image": (load_config().get("docker") or {}).get("llamacpp_image"),
        "use_case": use_case,
        "knobs": {
            "gpu_count": p.get("gpu_count"),
            "n_gpu_layers": p.get("n_gpu_layers", 999),
            "flash_attn": p.get("flash_attn", "off"),
            "max_model_len": p.get("max_model_len"),
            "n_cpu_moe": p.get("n_cpu_moe"),
            "parallel": p.get("parallel"),
        },
        "extra": {"jinja": True, "nickname": audit_info.get("name") or model_id},
        "env": {},
        "perf": {"peak_tok_s": winner.get("peak_tok_s"), "single_stream_tok_s": winner.get("single_tok_s")},
        "validation_key": {
            "hardware_fingerprint": hardware.fingerprint,
            "backend": "llamacpp",
            "runtime_version": runtime_version,
        },
        "validated_at": None,
        "source": "induction",
    }


def _write_report(run_dir: Path, model_id: str, a: dict, results: list[dict], winner: dict | None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# TUNING_REPORT — {model_id} (llamacpp)", "",
        f"- arch: {a.get('arch')}  quant: {a.get('quant') or a.get('file_type_id')}  "
        f"size: {a.get('size_bytes', 0) / 1e9:.1f} GB  native_ctx: {a.get('native_context')}", "",
        "## Sweep", "",
        "| ngl | parallel | mml | peak tok/s | single tok/s | ok |",
        "|-----|----------|-----|-----------|--------------|----|",
    ]
    for r in results:
        p = r["point"]
        lines.append(
            f"| {p.get('n_gpu_layers')} | {p.get('parallel')} | {p.get('max_model_len')} | "
            f"{r.get('peak_tok_s')} | {r.get('single_tok_s')} | {'✓' if r.get('ok') else '✗'} |"
        )
    if winner:
        wp = winner["point"]
        lines += ["", f"## Winner\n\nngl={wp.get('n_gpu_layers')} parallel={wp.get('parallel')} "
                  f"mml={wp.get('max_model_len')} → peak {winner.get('peak_tok_s')} tok/s, "
                  f"single {winner.get('single_tok_s')} tok/s"]
    path = run_dir / "TUNING_REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# --- entry points ------------------------------------------------------------
def plan(model_id: str, gguf_path: str, max_points: int | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg if cfg is not None else load_config()
    hw = hwd.detect()
    a = audit(gguf_path)
    free = free_gpus(hw, all_seats(cfg))
    total_gpus = len(free_gpus(hw, []))  # all GPUs, ignoring current occupancy
    gc = _gpu_count_for(a.get("size_bytes", 0), total_gpus)
    pts = candidate_points(a, gc, max_points)
    return {
        "model_id": model_id, "path": gguf_path, "backend": "llamacpp",
        "audit": {"arch": a.get("arch"), "quant": a.get("quant") or a.get("file_type_id"),
                  "size_gb": round(a.get("size_bytes", 0) / 1e9, 1), "native_ctx": a.get("native_context")},
        "free_gpus": free, "device": "gpu", "gpu_count": gc,
        "arch_supported": True, "arch_warning": None,
        "viable": [{"gpu_count": gc, "n_gpu_layers": 999}], "pruned": [], "priors": 0,
        "points": pts,
    }


def run(model_id: str, gguf_path: str, use_case: str | None = None, max_points: int | None = None,
        cfg: dict | None = None, progress=None) -> dict:
    _p = progress or (lambda *_: None)
    cfg = cfg if cfg is not None else load_config()
    hw = hwd.detect()
    a = audit(gguf_path)
    _p(f"audit: {a.get('arch')} · {round(a.get('size_bytes', 0) / 1e9, 1)}GB · "
       f"experts={a.get('n_expert')} · ctx={a.get('native_context')}")

    free = free_gpus(hw, all_seats(cfg))
    total_gpus = len(free_gpus(hw, []))  # all GPUs, ignoring current occupancy
    gc = _gpu_count_for(a.get("size_bytes", 0), total_gpus)
    points = candidate_points(a, gc, max_points)
    _p(f"backend=llamacpp · gpu_count={gc} · {len(free)} free GPU(s) · {len(points)} point(s)")
    if len(free) < gc:
        return {"model_id": model_id, "error": f"need {gc} free GPUs, {len(free)} free "
                f"(down a seat first)", "backend": "llamacpp"}

    results = []
    for i, point in enumerate(points, 1):
        gpus = assign_gpus(gc, hw, free_gpus(hw, all_seats(cfg)))
        if len(gpus) < gc:
            results.append({"point": point, "ok": False, "error": "insufficient free GPUs"})
            _p(f"[{i}/{len(points)}] {_point_sig(point)}: skipped (insufficient GPUs)")
            continue
        _p(f"[{i}/{len(points)}] {_point_sig(point)}: launching on GPU {gpus} + benching…")
        collect.add_pin(TUNING_CONTAINER)
        try:
            r = tune_point(model_id, gguf_path, point, gpus, cfg, hw)
        finally:
            collect.remove_pin(TUNING_CONTAINER)
        if r.get("ok"):
            _p(f"[{i}/{len(points)}] {_point_sig(point)}: peak {r.get('peak_tok_s')} tok/s, "
               f"single {r.get('single_tok_s')} tok/s")
        else:
            _p(f"[{i}/{len(points)}] {_point_sig(point)}: FAILED — {(r.get('error') or '')[:80]}")
        results.append(r)

    winner = stages.synthesize(results, use_case)
    run_dir = C.get_paths().runs_dir / f"induct-{model_id.replace('/', '__')}"
    report_path = _write_report(run_dir, model_id, a, results, winner)

    placement = None
    if winner:
        rtv = str((cfg.get("docker") or {}).get("llamacpp_image", "")).split(":")[-1] or "unknown"
        placement = to_placement(model_id, winner, a, hw, rtv, use_case)
        md = (cfg.get("roots") or {}).get("models_dir")
        try:
            lp = str(Path(gguf_path).relative_to(Path(md).expanduser())) if md else None
        except (ValueError, TypeError):
            lp = None
        report.write_placement(model_id, a, placement, hw, local_path=lp)

    return {
        "model_id": model_id, "backend": "llamacpp", "points": len(points),
        "results": results, "winner": winner,
        "placement_id": placement["id"] if placement else None,
        "report": str(report_path),
        "bench": "throughput sweep complete (bench.sh)",
    }
