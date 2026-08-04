"""Quality + perf benchmark harness for inducted placements (`johnny bench`, §3.6 step 5).

Resolves a registry placement, benches it against a matching *running* seat when one
exists (no relaunch), else a temporary reaper-pinned tuning seat launched from the
placement's knobs (same machinery as induction). Suites:

- ``perf``: throughput/single-stream via the bundled bench.sh + KV readback — what
  induction measures per point; refreshes the placement's ``perf``.
- ``arc``:  ARC-Challenge CoT accuracy via the bundled arc_eval.py. Needs the optional
  eval deps (``openai`` + ``datasets``): ``pipx inject johnny-fleet openai datasets``
  or ``pip install 'johnny-fleet[bench]'``.

``humaneval`` and ``needle`` are planned but not wired (they need an lm-eval
--log_samples run / a code-corpus build respectively). Scores land in the placement's
``quality`` block in the registry plus a BENCH_REPORT.md under runs/.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path

from . import config as C
from .engine import all_seats, load_config
from .engine.placement import assign_gpus, free_gpus
from .hardware import detect as hwd
from .induct import stages
from .registry import store
from .telemetry import collect

SUITES = ("perf", "arc")
PLANNED = {
    "humaneval": "needs an lm-eval --log_samples run first (humaneval_chat_score.py re-scores it)",
    "needle": "needs a code corpus build (corpus.json) before code_needle.py can run",
}
_ARC_TIMEOUT = 4 * 3600  # full ARC-Challenge with CoT can run for a while


def resolve_target(reg: dict, target: str) -> list[tuple[str, dict]]:
    """(model_id, placement) candidates for a model id or placement id. Exact model id →
    all its placements; else exact placement id; else substring on either. The caller
    disambiguates >1 (picker) and errors on 0."""
    models = reg.get("models") or {}
    if target in models:
        return [(target, p) for p in (models[target].get("placements") or [])]
    exact = [(mid, p) for mid, m in models.items()
             for p in (m.get("placements") or []) if p.get("id") == target]
    if exact:
        return exact
    sub = [(mid, p) for mid, m in models.items()
           for p in (m.get("placements") or []) if target in (p.get("id") or "")]
    if sub:
        return sub
    return [(mid, p) for mid, m in models.items() if target in mid
            for p in (m.get("placements") or [])]


def find_running_seat(model_id: str, placement_id: str, cfg: dict):
    """A live seat launched from this exact placement (johnny.* labels), or None.
    Knobs define the perf numbers, so only an exact placement match is reusable."""
    for s in all_seats(cfg):
        labels = (getattr(s, "extra", None) or {}).get("labels") or {}
        if labels.get("johnny.model") == model_id and labels.get("johnny.placement") == placement_id:
            return s
    return None


def point_from_placement(placement: dict) -> dict:
    """Invert report.to_placement: registry knobs → an induction-style config point,
    so the tuning-seat spec builders can launch it verbatim."""
    k = placement.get("knobs") or {}
    ex = placement.get("extra") or {}
    if ex.get("device") == "cpu":
        return {"device": "cpu", "cpuset": ex.get("cpuset"),
                "embeddings": ex.get("runner") == "pooling",
                "max_model_len": k.get("max_model_len"),
                "max_num_seqs": k.get("max_num_seqs"),
                "max_num_batched_tokens": k.get("max_num_batched_tokens")}
    return {"tp": k.get("tensor_parallel_size") or k.get("gpu_count") or 1,
            "quant": k.get("quant"), "max_model_len": k.get("max_model_len"),
            "gpu_memory_util": k.get("gpu_memory_util"),
            "max_num_seqs": k.get("max_num_seqs"),
            "max_num_batched_tokens": k.get("max_num_batched_tokens"),
            "kv_cache_dtype": k.get("kv_cache_dtype", "auto"),
            "mtp": k.get("mtp") or {"enabled": False},
            "embeddings": ex.get("runner") == "pooling"}


def _local_path(model_id: str, reg: dict, cfg: dict) -> str | None:
    ident = ((reg.get("models") or {}).get(model_id) or {}).get("identity") or {}
    lp, md = ident.get("local_path"), (cfg.get("roots") or {}).get("models_dir")
    if lp and md:
        p = Path(md).expanduser() / lp
        if p.exists():
            return str(p)
    if lp and Path(lp).expanduser().exists():
        return str(Path(lp).expanduser())
    try:
        _, path = stages.discover(ident.get("repo_id") or model_id, cfg)
        return path
    except Exception:
        return None


def arc_deps_missing() -> list[str]:
    return [m for m in ("openai", "datasets") if importlib.util.find_spec(m) is None]


def parse_arc_output(out: str) -> dict | None:
    """arc_eval.py summary → scores. 'Accuracy: 1114/1172 = 95.05%' + no-extraction/errors."""
    m = re.search(r"Accuracy:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", out)
    if not m:
        return None
    res = {"accuracy_pct": float(m.group(3)), "correct": int(m.group(1)), "total": int(m.group(2))}
    ne = re.search(r"No extraction:\s*(\d+)", out)
    if ne:
        res["no_extraction"] = int(ne.group(1))
    er = re.search(r"API errors:\s*(\d+)", out)
    if er:
        res["api_errors"] = int(er.group(1))
    return res


def _stream_run(cmd: list[str], timeout: float, progress) -> tuple[int, str]:
    """Run a long eval subprocess, echoing its progress lines live; returns (rc, output)."""
    lines: list[str] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + timeout
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            lines.append(line)
            if line.strip():
                progress(line.rstrip())
            if time.monotonic() > deadline:
                proc.kill()
                lines.append("\n[timeout]")
                break
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    return proc.returncode or 0, "".join(lines)


def _run_arc(port: int, model_id: str, run_dir: Path, cfg: dict, limit: int | None,
             concurrency: int, thinking: bool, progress) -> dict:
    from .bundled import resolve_script

    missing = arc_deps_missing()
    if missing:
        return {"ok": False, "error": f"missing eval deps: {', '.join(missing)} — "
                "`pipx inject johnny-fleet openai datasets` (or pip install 'johnny-fleet[bench]')"}
    script = resolve_script("arc_eval", cfg)
    if not script:
        return {"ok": False, "error": "arc_eval.py unavailable (not bundled, no scripts.arc_eval override)"}
    out_path = run_dir / "arc_samples.jsonl"
    cmd = [sys.executable, script, "--base-url", f"http://127.0.0.1:{port}/v1",
           "--model", model_id, "--concurrency", str(concurrency), "--out", str(out_path)]
    if limit:
        cmd += ["--limit", str(limit)]
    if not thinking:  # PLAN §3.6: thinking-off plumbed, else reasoning models score 0
        cmd += ["--disable-thinking"]
    rc, out = _stream_run(cmd, _ARC_TIMEOUT, progress)
    scores = parse_arc_output(out)
    if not scores:
        return {"ok": False, "error": f"arc_eval produced no score (rc={rc}): {out[-300:]}"}
    scores.update({"ok": True, "limit": limit, "samples": str(out_path)})
    return scores


def _run_perf(port: int, container: str | None, model_id: str, point: dict, cfg: dict, progress) -> dict:
    """Induction's throughput bench against an already-running seat (no teardown)."""
    from .backends.vllm import VllmDriver
    from .bundled import resolve_script
    from .util import run as _run

    result: dict = {}
    if container:  # KV readback is best-effort — startup lines scroll off long-lived seats
        try:
            drv = VllmDriver(image=C.resolve_image(cfg, device="cpu" if point.get("device") == "cpu" else "gpu",
                                                   model_id=model_id))
            result.update({k: v for k, v in stages._parse_kv_cache(
                drv.logs(container, tail=2000) or "").items() if v is not None})
        except Exception:
            pass
    if point.get("embeddings"):
        parsed = stages._bench_embeddings(port, model_id)
    else:
        script = resolve_script("bench", cfg)
        if not script:
            return {"ok": False, "error": "bench script unavailable (not bundled and no scripts.bench override)"}
        progress("perf: bench.sh sweep (concurrency ramp + single-stream)…")
        rc, out, errout = _run(["bash", script, str(port), model_id], timeout=1200)
        parsed = stages._parse_bench(out)
        if parsed.get("peak_tok_s") is None:
            parsed["error"] = (errout or out)[-300:]
    result.update(parsed)
    result["ok"] = parsed.get("peak_tok_s") is not None
    return result


def write_scores(model_id: str, placement_id: str, perf: dict | None = None,
                 quality: dict | None = None) -> bool:
    """Fold bench results into the placement: perf refresh + quality scores (stamped)."""
    reg = store.load()
    for p in ((reg.get("models") or {}).get(model_id) or {}).get("placements", []):
        if p.get("id") == placement_id:
            if perf:
                p["perf"] = {**(p.get("perf") or {}),
                             "peak_tok_s": perf.get("peak_tok_s"),
                             "single_stream_tok_s": perf.get("single_tok_s")}
            if quality:
                p.setdefault("quality", {}).update(quality)
            store.save(reg)
            return True
    return False


def write_report(run_dir: Path, model_id: str, placement_id: str, results: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# BENCH_REPORT — {model_id}", "", f"- placement: `{placement_id}`",
             f"- date: {time.strftime('%Y-%m-%d %H:%M')}", ""]
    for suite, r in results.items():
        lines.append(f"## {suite}")
        lines.append("")
        if not r.get("ok"):
            lines.append(f"failed: {r.get('error')}")
        elif suite == "perf":
            kv = r.get("kv_cache_tokens")
            lines.append(f"peak {r.get('peak_tok_s')} tok/s · single {r.get('single_tok_s')} tok/s"
                         + (f" · KV {kv/1e6:.2f}M tok ({r.get('max_concurrency')}x)" if kv else ""))
        elif suite == "arc":
            lines.append(f"ARC-Challenge accuracy {r.get('accuracy_pct')}% "
                         f"({r.get('correct')}/{r.get('total')}"
                         + (f", limit={r['limit']}" if r.get("limit") else "")
                         + f") · no-extraction {r.get('no_extraction', 0)} · errors {r.get('api_errors', 0)}")
        lines.append("")
    path = run_dir / "BENCH_REPORT.md"
    path.write_text("\n".join(lines))
    return path


def run(model_id: str, placement: dict, suites: list[str], cfg: dict | None = None,
        limit: int | None = None, concurrency: int = 8, thinking: bool = False,
        progress=None) -> dict:
    """Bench one placement: reuse its running seat or launch a temp tuning seat, run the
    suites, write scores to the registry + a BENCH_REPORT. Returns per-suite results."""
    _p = progress or (lambda *_: None)
    cfg = cfg if cfg is not None else load_config()
    pid = placement.get("id") or "?"
    if (placement.get("backend") or "vllm") != "vllm":
        return {"error": f"backend {placement.get('backend')!r} isn't bench-wired yet (vLLM only)"}
    point = point_from_placement(placement)
    run_dir = C.get_paths().runs_dir / f"bench-{model_id.replace('/', '__')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    seat = find_running_seat(model_id, pid, cfg)
    drv = None
    if seat is not None:
        port, container = seat.port, getattr(seat, "name", None)
        _p(f"reusing running seat {container} (port {port}) — no relaunch")
    else:
        from .backends.vllm import VllmDriver

        reg = store.load()
        path = _local_path(model_id, reg, cfg)
        if not path:
            return {"error": f"no local weights found for {model_id} (identity.local_path missing?)"}
        hw = hwd.detect()
        is_cpu = point.get("device") == "cpu"
        gpus: list[int] = []
        if not is_cpu:
            gpus = assign_gpus(point["tp"], hw, free_gpus(hw, all_seats(cfg)))
            if len(gpus) < point["tp"]:
                return {"error": f"insufficient free GPUs for tp={point['tp']} — down a seat first"}
        drv = VllmDriver(image=C.resolve_image(cfg, device="cpu" if is_cpu else "gpu", model_id=model_id))
        spec = stages._cpu_tuning_spec(model_id, path, point, cfg) if is_cpu \
            else stages._tuning_spec(model_id, path, point, gpus, cfg, hw)
        # Bench what production runs: carry the placement's parsers/template into the seat.
        spec["extra"] = {**spec.get("extra", {}), **(placement.get("extra") or {})}
        port, container = stages.TUNING_PORT, stages.TUNING_CONTAINER
        _p(f"launching temp seat from {pid} on " + ("CPU" if is_cpu else f"GPU {gpus}") + "…")
        drv.launch(spec)
        ready, why = stages._wait_ready(drv, container, port, timeout=900 if is_cpu else 600)
        if not ready:
            tail = stages._diagnose(drv, container)
            drv.stop(container)
            return {"error": (why or "seat did not become ready") + (f" — {tail}" if tail else "")}

    results: dict = {}
    collect.add_pin(container)  # reaper-safe while benching (borrowed seats too)
    try:
        for s in suites:
            if s == "perf":
                results["perf"] = _run_perf(port, container, model_id, point, cfg, _p)
            elif s == "arc":
                if point.get("embeddings"):
                    results["arc"] = {"ok": False, "error": "embeddings model — ARC needs a generative seat"}
                else:
                    results["arc"] = _run_arc(port, model_id, run_dir, cfg, limit, concurrency, thinking, _p)
            ok = results[s].get("ok")
            _p(f"{s}: " + ("done" if ok else f"FAILED — {results[s].get('error')}"))
    finally:
        collect.remove_pin(container)
        if drv is not None:  # only tear down what we launched
            drv.stop(container)
            _p("temp seat stopped")

    perf = results.get("perf") if (results.get("perf") or {}).get("ok") else None
    quality = None
    arc = results.get("arc")
    if arc and arc.get("ok"):
        quality = {"arc": {k: arc.get(k) for k in
                           ("accuracy_pct", "correct", "total", "no_extraction", "api_errors", "limit")}
                   | {"date": time.strftime("%Y-%m-%d")}}
    wrote = write_scores(model_id, pid, perf=perf, quality=quality) if (perf or quality) else False
    report = write_report(run_dir, model_id, pid, results)
    return {"model_id": model_id, "placement_id": pid, "results": results,
            "registry_updated": wrote, "report": str(report),
            "seat": "reused" if drv is None else "temporary"}
