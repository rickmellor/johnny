"""Quality + perf benchmark harness for inducted placements (`johnny bench`, §3.6 step 5).

Resolves a registry placement, benches it against a matching *running* seat when one
exists (no relaunch), else a temporary reaper-pinned tuning seat launched from the
placement's knobs (same machinery as induction). Suites:

- ``perf``: throughput/single-stream via the bundled bench.sh + KV readback — what
  induction measures per point; refreshes the placement's ``perf``.
- ``arc``:    ARC-Challenge CoT accuracy via the bundled arc_eval.py. Needs the optional
  eval deps (``openai`` + ``datasets``): ``pipx inject johnny-fleet openai datasets``
  or ``pip install 'johnny-fleet[bench]'``.
- ``icl``:    four-category in-context-learning probe (bundled icl_eval.py) — small
  few-shot pattern-completion tasks with a verified closed-form rule each, checking
  whether the model induces the rule rather than pattern-matching surface tokens.
  Only needs ``openai``.
- ``needle``: positional-recall / "lost in the middle" probe (bundled code_needle.py)
  against a code corpus built on first use from the bundled cli.py (bundled
  build_corpus.py), cached under runs/needle-corpus/. Only needs ``openai``.
- ``depth``:  prefill/decode throughput + time-to-first-response as a function of
  context depth, via the ``llama-benchy`` (github.com/eugr/llama-benchy) OpenAI
  client — the one thing perf/arc/icl/needle never measure (they're all short,
  fixed-length prompts). Backend-agnostic by construction (plain HTTP against
  vLLM or llama-server). Needs ``llama-benchy``.
- ``humaneval``: real HumanEval pass@1 via ``lm-eval`` (``local-chat-completions``
  connector, ``--apply_chat_template --log_samples``) re-scored by the bundled
  ``humaneval_chat_score.py`` — lm-eval's own bundled HumanEval filter assumes
  raw-completion mode and silently scores a chat model's markdown-fenced answer
  as pass@1=0, so this suite runs lm-eval only to get real generations on disk,
  then re-runs the official HumanEval ``check()`` itself. Needs ``lm-eval[api]``
  (``openai``'s chat-completions-shaped API client plus the ``evaluate``/
  ``datasets`` libs lm-eval itself depends on).
- ``automationbench``: real agentic tool-use eval via Zapier's public AutomationBench
  (600 tasks, 100/domain across sales/marketing/operations/support/finance/HR — every
  "SaaS tool" is a local simulation, no live creds/network needed). Shells out to the
  benchmark's own OpenAI-compatible tool-calling agent loop (self-bootstraps a vendored
  `uv`-managed checkout under state_dir on first use — it's a whole separate agent
  harness, not a task+scorer pair like the others). A single task's transcript can pass
  40K+ tokens once tool-call history accumulates — pick a placement with real per-request
  context headroom (vLLM's max_model_len already is one; llama.cpp's is split across
  ``--parallel``, see AGENTS.md). Needs ``uv`` on PATH.
- ``ctxsafe``: empirical context-safety probe (see ``ctxsafe.py``) — walks real
  needle-in-haystack requests at progressively deeper depths up to the placement's
  configured ``max_model_len`` against a *dedicated, always-fresh* probe seat (never
  a live production seat — this suite deliberately tries to crash things), with live
  ``rocm-smi`` VRAM polling during every request and real crash detection (container
  liveness, not just a dropped connection). Born from a real 2026-08-06 incident: two
  llama.cpp placements crashed (HIP GPU0 VRAM exhaustion) at real prefill depths well
  short of their configured context — see AGENTS.md's "Context safety" section and
  the Ornith Featherweight placements' ``extra.note`` in the registry. Writes
  ``quality.ctxsafe`` with the empirically verified-safe depth, distinct from both
  the model's trained ``native_context`` and the placement's configured
  ``max_model_len`` — the gap between those three is exactly what caused the
  incident. No extra deps (uses ``openai`` like the other probes; ``tiktoken`` if
  present for precise depth construction, else a char-count estimate).

Scores land in the placement's ``quality`` block in the registry plus a
BENCH_REPORT.md under runs/.
"""

from __future__ import annotations

import importlib.util
import json
import os
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

SUITES = ("perf", "arc", "icl", "needle", "depth", "humaneval", "ctxsafe", "automationbench", "planbench")
PLANNED: dict[str, str] = {}
_ARC_TIMEOUT = 4 * 3600  # full ARC-Challenge with CoT can run for a while
_ICL_TIMEOUT = 20 * 60  # 16 short single-turn cases — should be minutes, not hours
_NEEDLE_TIMEOUT = 30 * 60  # 16 long-context reads + 800-token completions
_DEPTH_TIMEOUT = 20 * 60  # a handful of depths, a few runs each — minutes, not hours
_DEPTH_SWEEP = (0, 4096, 8192)  # modest on purpose — a quick suite, not an hours-long one
_HUMANEVAL_TIMEOUT = 90 * 60  # 164 problems up to 2048 gen toks each — generous for a slow (e.g. llamacpp) seat
_HUMANEVAL_SCORE_TIMEOUT = 30 * 60  # re-scorer: up to 164 subprocess test runs, 10s cap each
_CTXSAFE_LAUNCH_TIMEOUT = 900  # a big long-context seat can be slow to load (large KV reservation)

# Dedicated bench-tuning container/port for llamacpp temp seats — distinct from both
# the vLLM tuning seat (stages.TUNING_PORT/CONTAINER, 9000) and induct/llamacpp.py's
# own (transient, --rm) llama-bench container (port 9001), so none of the three can
# ever collide if run back to back.
_LLAMACPP_TUNING_CONTAINER = "llamacpp-johnny-bench-tuning"
_LLAMACPP_TUNING_PORT = 9002


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
    so the tuning-seat spec builders can launch it verbatim.

    llamacpp placements already store backend-native knobs (n_gpu_layers, gpu_count,
    ...) — see induct/llamacpp.py's to_placement — so there's no vLLM-style point to
    invert; only the fields bench.py's own dispatch needs (device/embeddings/gpu_count)
    are pulled out, and the launch spec is built straight from placement.knobs/.extra
    (engine.launch.build_spec, the same path `johnny up` uses)."""
    k = placement.get("knobs") or {}
    ex = placement.get("extra") or {}
    if (placement.get("backend") or "vllm") == "llamacpp":
        return {"device": "cpu" if ex.get("device") == "cpu" else "gpu",
                "gpu_count": k.get("gpu_count") or 1,
                "embeddings": ex.get("runner") == "pooling"}
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
    """Host-side absolute path to a model's weights, or None if nothing resolves.

    Tries, in order: config.resolve_weights_path (roots.models_dir, falling back to
    roots.nas_dir — handles a `nas:`-prefixed identity.local_path explicitly, and
    existence-based even without the prefix), an absolute local_path, then HF-cache
    discovery. Existence-only — callers that need the container-relative path (which
    mount, /models vs /nas) go through engine.launch.build_spec instead; this is for
    "is there something to launch at all" pre-checks."""
    ident = ((reg.get("models") or {}).get(model_id) or {}).get("identity") or {}
    lp = ident.get("local_path")
    if lp:
        resolved = C.resolve_weights_path(lp, cfg)
        if resolved:
            return resolved.host_path
        clean = lp[len("nas:"):] if lp.startswith("nas:") else lp
        if Path(clean).expanduser().exists():
            return str(Path(clean).expanduser())
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


def _stream_run(cmd: list[str], timeout: float, progress, env: dict | None = None,
                cwd: str | None = None) -> tuple[int, str]:
    """Run a long eval subprocess, echoing its progress lines live; returns (rc, output).
    env, if given, replaces the subprocess environment wholesale (pass a copy of os.environ
    plus overrides — not a delta). cwd, if given, runs the subprocess there — for external
    tools that are their own project (e.g. automationbench's `uv run` needs its checkout's
    pyproject.toml as cwd) rather than something invoked via an absolute script path."""
    lines: list[str] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd)
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
    else:
        # Thinking needs generation budget AND per-request time, not just the longer
        # wall timeout: at the 512-token default a reasoning model clips mid-think and
        # the answer never emits (Qwen3.8: 18% ARC, below the 25% MCQ floor), and at
        # arc_eval's old 60s client timeout most contended thinking requests died as
        # "Request timed out" (31% — timeouts, not wrong answers). 2026-08-17/18.
        cmd += ["--max-tokens", "2048", "--timeout", "600"]
    rc, out = _stream_run(cmd, _ARC_TIMEOUT, progress)
    scores = parse_arc_output(out)
    if not scores:
        return {"ok": False, "error": f"arc_eval produced no score (rc={rc}): {out[-300:]}"}
    scores.update({"ok": True, "limit": limit, "samples": str(out_path)})
    return scores


def _run_planbench(port: int, model_id: str, run_dir: Path, cfg: dict, limit: int | None,
                   concurrency: int, thinking: bool, progress) -> dict:
    """PlanBench task_1 plan generation — isolates PLANNING from tool loops and long context.

    Short one-shot PDDL (Blocksworld) problems: no tools, no exploration, ~1-2K context, so a
    weak score here is a planning limit rather than an agentic-loop or context-pressure artifact
    (the ambiguity automationbench alone can't resolve). Scored as exact action-sequence match
    against the reference plan — a strict LOWER BOUND, since a different-but-valid plan scores 0
    (upstream PlanBench uses the VAL validator to accept those); `plan_prefix_pct` gives partial
    credit for a correct leading prefix. Comparable across models on identical instances, which
    is what it's for.
    """
    from .bundled import resolve_script

    missing = arc_deps_missing()          # same deps: openai + datasets
    if missing:
        return {"ok": False, "error": f"missing eval deps: {', '.join(missing)} — "
                "`pipx inject johnny-fleet openai datasets` (or pip install 'johnny-fleet[bench]')"}
    script = resolve_script("planbench_eval", cfg)
    if not script:
        return {"ok": False, "error": "planbench_eval.py unavailable (not bundled, no scripts.planbench_eval override)"}
    out_path = run_dir / "planbench_results.json"
    out_path.unlink(missing_ok=True)      # never score a prior run's file (see _run_icl)
    cmd = [sys.executable, script, "--base-url", f"http://127.0.0.1:{port}/v1",
           "--model", model_id, "--concurrency", str(concurrency), "--out", str(out_path),
           "--limit", str(limit or 100)]
    if not thinking:
        cmd += ["--disable-thinking"]
    else:
        cmd += ["--max-tokens", "4096", "--timeout", "900"]
    rc, out = _stream_run(cmd, _ARC_TIMEOUT, progress)
    m = re.search(r"PLANBENCH_JSON (\{.*\})", out)
    if not m:
        return {"ok": False, "error": f"planbench_eval produced no score (rc={rc}): {out[-300:]}"}
    scores = json.loads(m.group(1))
    scores.update({"ok": True, "limit": limit, "samples": str(out_path)})
    return scores


def _openai_dep_missing() -> list[str]:
    return [m for m in ("openai",) if importlib.util.find_spec(m) is None]


def _run_icl(port: int, model_id: str, run_dir: Path, cfg: dict, limit: int | None,
             thinking: bool, progress) -> dict:
    from .bundled import resolve_script

    missing = _openai_dep_missing()
    if missing:
        return {"ok": False, "error": f"missing eval deps: {', '.join(missing)} — "
                "`pipx inject johnny-fleet openai` (or pip install 'johnny-fleet[bench]')"}
    script = resolve_script("icl_eval", cfg)
    if not script:
        return {"ok": False, "error": "icl_eval.py unavailable (not bundled, no scripts.icl_eval override)"}
    out_path = run_dir / "icl_results.json"
    cmd = [sys.executable, script, "--base-url", f"http://127.0.0.1:{port}/v1",
           "--model", model_id, "--out", str(out_path)]
    if limit:
        cmd += ["--limit", str(limit)]
    if not thinking:  # PLAN §3.6: thinking-off plumbed, else reasoning models score 0
        cmd += ["--disable-thinking"]
    else:
        # Same thinking-budget rule as ARC: the 800-token default clips a long thinker
        # mid-reasoning and the answer never emits (Ornith-397B: 16/16 cases at exactly
        # the cap, extracted=None, 2026-08-18). Time to match: 4096 tok at ~20 t/s
        # needs minutes, not icl_eval's 60s default client timeout.
        cmd += ["--max-tokens", "4096", "--timeout", "600"]
    # Same stale-export hazard automationbench already fixed: a run killed before its
    # write leaves the PREVIOUS run's file in place, and it scores as fresh (bit the
    # Ornith icl rerun 2026-08-18: 7 streamed PASSes recorded as the old file's 0/16).
    out_path.unlink(missing_ok=True)
    # Thinking wall: 16 cases can legitimately take minutes each (up to the 600s
    # per-request timeout) — the 20-min wall killed a healthy run at case 12.
    rc, out = _stream_run(cmd, 3 * 3600 if thinking else _ICL_TIMEOUT, progress)
    if rc != 0:
        return {"ok": False, "error": f"icl_eval exited rc={rc} before writing results — {out[-300:]}"}
    try:
        data = json.loads(out_path.read_text())
    except (OSError, ValueError) as e:
        return {"ok": False, "error": f"icl_eval produced no parseable output (rc={rc}): {e} — {out[-300:]}"}
    agg = data.get("aggregate") or {}
    return {"ok": True, "pass": agg.get("pass"), "fail": agg.get("fail"),
            "category_breakdown": data.get("category_breakdown"), "limit": limit,
            "samples": str(out_path)}


def _ensure_corpus(cfg: dict, progress) -> Path | None:
    """Build (once) and cache the needle corpus under the shared runs dir — deterministic
    given the bundled source file, so every placement/run reuses it instead of rebuilding."""
    from .bundled import resolve_script

    corpus_path = C.get_paths().runs_dir / "needle-corpus" / "corpus.json"
    if corpus_path.exists():
        return corpus_path
    script = resolve_script("build_corpus", cfg)
    if not script:
        return None
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    progress("needle: building code corpus (first use, cached under runs/needle-corpus/)…")
    proc = subprocess.run([sys.executable, script, "--out", str(corpus_path)],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not corpus_path.exists():
        return None
    return corpus_path


def _run_needle(port: int, model_id: str, run_dir: Path, cfg: dict, thinking: bool, progress) -> dict:
    from .bundled import resolve_script

    missing = _openai_dep_missing()
    if missing:
        return {"ok": False, "error": f"missing eval deps: {', '.join(missing)} — "
                "`pipx inject johnny-fleet openai` (or pip install 'johnny-fleet[bench]')"}
    corpus_path = _ensure_corpus(cfg, progress)
    if not corpus_path:
        return {"ok": False, "error": "could not build the needle corpus (build_corpus.py unavailable or failed)"}
    script = resolve_script("code_needle", cfg)
    if not script:
        return {"ok": False, "error": "code_needle.py unavailable (not bundled, no scripts.code_needle override)"}
    out_path = run_dir / "needle_results.json"
    # code_needle.py has no --limit — it's a fixed 16-target probe, not a subsettable suite.
    cmd = [sys.executable, script, "--corpus", str(corpus_path), "--base-url", f"http://127.0.0.1:{port}/v1",
           "--model", model_id, "--out", str(out_path)]
    if not thinking:  # PLAN §3.6: thinking-off plumbed, else reasoning models score 0
        cmd += ["--disable-thinking"]
    else:
        # Same thinking-budget rule as arc/icl: at the 800-token default a reasoning
        # model burns the whole budget in-think over the 30K corpus and the answer
        # never emits — Qwen3.8 "scored" 1/16 with 15 zero-output targets (2026-08-18),
        # which read as a recall failure until the transcripts showed empty content.
        # 4096 + a patient per-request timeout gives the answer room to exist.
        cmd += ["--max-tokens", "4096", "--timeout", "600"]
    out_path.unlink(missing_ok=True)  # never score a prior run's file (see _run_icl)
    rc, out = _stream_run(cmd, 2 * 3600 if thinking else _NEEDLE_TIMEOUT, progress)
    if rc != 0:
        return {"ok": False, "error": f"code_needle exited rc={rc} before writing results — {out[-300:]}"}
    try:
        data = json.loads(out_path.read_text())
    except (OSError, ValueError) as e:
        return {"ok": False, "error": f"code_needle produced no parseable output (rc={rc}): {e} — {out[-300:]}"}
    agg = data.get("aggregate") or {}
    return {"ok": True, "pass": agg.get("pass"), "fail": agg.get("fail"),
            "position_bias": data.get("position_bias"), "corpus_tokens": data.get("corpus_tokens"),
            "samples": str(out_path)}


def _llama_benchy_dep_missing() -> list[str]:
    return [] if importlib.util.find_spec("llama_benchy") else ["llama-benchy"]


def _parse_depth_output(data: dict) -> list[dict]:
    """llama-benchy's --format json 'benchmarks' list (one entry per swept
    concurrency/depth/pp/tg combo — bench.py only sweeps depth, so each entry here is
    exactly one depth) → [{depth, pp_tok_s, tg_tok_s, ttfr_ms}], sorted by depth."""
    points = []
    for b in data.get("benchmarks") or []:
        points.append({
            "depth": b.get("context_size"),
            "pp_tok_s": round((b.get("pp_throughput") or {}).get("mean") or 0, 1),
            "tg_tok_s": round((b.get("tg_throughput") or {}).get("mean") or 0, 1),
            "ttfr_ms": round((b.get("ttfr") or {}).get("mean") or 0, 1),
        })
    return sorted(points, key=lambda p: p.get("depth") or 0)


def _run_depth(port: int, model_id: str, run_dir: Path, cfg: dict, progress) -> dict:
    """Prefill/decode throughput + time-to-first-response as a function of context
    depth, via llama-benchy (github.com/eugr/llama-benchy) — a backend-agnostic
    OpenAI-client bench (plain HTTP against /v1/..., works against vLLM and
    llama-server alike, no CUDA/ROCm-specific code). The one real gap the other three
    suites leave open: perf/arc/icl/needle all use short, fixed-length prompts and
    never measure latency or how throughput degrades as context fills up.
    """
    missing = _llama_benchy_dep_missing()
    if missing:
        return {"ok": False, "error": f"missing eval deps: {', '.join(missing)} — "
                "`pipx inject johnny-fleet llama-benchy` (or pip install 'johnny-fleet[bench]')"}
    out_path = run_dir / "depth_results.json"
    cmd = [sys.executable, "-m", "llama_benchy", "--base-url", f"http://127.0.0.1:{port}/v1",
           "--model", model_id, "--depth", *[str(d) for d in _DEPTH_SWEEP],
           "--format", "json", "--save-result", str(out_path)]
    rc, out = _stream_run(cmd, _DEPTH_TIMEOUT, progress)
    try:
        data = json.loads(out_path.read_text())
    except (OSError, ValueError) as e:
        return {"ok": False, "error": f"llama-benchy produced no parseable output (rc={rc}): {e} — {out[-300:]}"}
    points = _parse_depth_output(data)
    if not points:
        return {"ok": False, "error": f"llama-benchy produced no benchmark points (rc={rc}): {out[-300:]}"}
    return {"ok": True, "points": points, "samples": str(out_path)}


def _lm_eval_dep_missing() -> list[str]:
    # tenacity only ships with the `lm-eval[api]` extra — it's what local-chat-completions
    # needs for retries; a bare `pip install lm-eval` installs `evaluate`/`datasets` (the
    # HumanEval task's own deps) but not this, and fails at model-init time with a
    # confusing error rather than up front here.
    return [m for m in ("lm_eval", "tenacity") if importlib.util.find_spec(m) is None]


def parse_humaneval_score(out: str) -> dict | None:
    """humaneval_chat_score.py's summary line: 'HumanEval pass@1: 121/164 = 73.78%'
    (+ an optional 'Failed entry points: [...]' line)."""
    m = re.search(r"HumanEval pass@1:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", out)
    if not m:
        return None
    res = {"passed": int(m.group(1)), "total": int(m.group(2)), "pass_at_1_pct": float(m.group(3))}
    fp = re.search(r"Failed entry points:\s*(.+)", out)
    if fp:
        res["failed_entry_points_sample"] = fp.group(1).strip()
    return res


def _run_humaneval(port: int, model_id: str, run_dir: Path, cfg: dict, limit: int | None,
                    concurrency: int, thinking: bool, progress) -> dict:
    """Real HumanEval pass@1 for a chat-completion seat.

    lm-eval doesn't ship a runner shape like arc_eval/icl_eval/code_needle (those are
    bundled scripts hitting /v1/... directly) — it's a whole harness with its own model
    connectors, task registry and output-directory convention, so this suite shells out
    to it instead: `lm_eval run --model local-chat-completions` against the live seat's
    /v1/chat/completions, `--apply_chat_template` (HumanEval's own doc_to_text is a bare
    "{{prompt}}" — a chat model needs the template to turn that into a real message),
    `--log_samples` so real generations land on disk. lm-eval's own scoring of those
    samples is then thrown away — its bundled HumanEval filter assumes raw-completion
    continuation, not a markdown-fenced chat answer, and silently reports pass@1=0 on
    correct code — and humaneval_chat_score.py re-scores the logged samples for real.

    HF_ALLOW_CODE_EVAL=1 is required: humaneval's task utils.py runs a `evaluate`
    code_eval self-test at import time (executes a hardcoded example function+test to
    confirm the sandboxed-eval pipeline works) and raises without it, before a single
    request is even sent — lm-eval's own safety gate on executing model-written code,
    separate from --confirm_run_unsafe_code (which is the CLI's ack of the same thing).
    """
    from .bundled import resolve_script

    missing = _lm_eval_dep_missing()
    if missing:
        return {"ok": False, "error": f"missing eval deps: {', '.join(missing)} — "
                "`pipx inject johnny-fleet 'lm-eval[api]'` (or pip install 'johnny-fleet[bench]')"}
    score_script = resolve_script("humaneval_score", cfg)
    if not score_script:
        return {"ok": False, "error": "humaneval_chat_score.py unavailable (not bundled, "
                "no scripts.humaneval_score override)"}
    lmeval_out = run_dir / "humaneval_lmeval"
    # Thinking-on models (or ones like Kimi-K2.7 whose template has no off switch) spend
    # most of the budget inside the think block — give them room or they truncate to 0.
    gen_kwargs = [f"max_gen_toks={10240 if thinking else 2048}", "until=[]"]  # default max_gen_toks=1024 + raw-completion
    # stop sequences (\nclass, \ndef, ...) truncate real chat answers mid-function; see script docstring.
    if not thinking:  # PLAN §3.6: thinking-off plumbed, else reasoning models emit an unclosed
        # <think> block that eats the whole generation budget and leaves content empty/truncated.
        # Same mechanism arc_eval/icl_eval/code_needle use (extra_body.chat_template_kwargs) — here
        # it rides in as a plain gen_kwargs key, since lm-eval's local-chat-completions merges
        # unrecognized gen_kwargs straight into the chat-completions request body.
        gen_kwargs.append("chat_template_kwargs={'enable_thinking': False}")
    cmd = [sys.executable, "-m", "lm_eval", "run",
           "--model", "local-chat-completions",
           "--model_args", f"base_url=http://127.0.0.1:{port}/v1/chat/completions,model={model_id},"
                            f"num_concurrent={concurrency},max_retries=3,tokenized_requests=False,"
                            # single-digit-tok/s CPU-MoE seats + thinking need far more than
                            # lm-eval's 300s default per request; env-tunable like perf's timeout.
                            f"timeout={os.environ.get('JOHNNY_BENCH_REQUEST_TIMEOUT', 3600 if thinking else 600)}",
           "--tasks", "humaneval", "--apply_chat_template", "--log_samples",
           "--output_path", str(lmeval_out), "--confirm_run_unsafe_code",
           "--gen_kwargs", *gen_kwargs]
    if limit:
        cmd += ["--limit", str(limit)]
    env = {**os.environ, "HF_ALLOW_CODE_EVAL": "1"}
    rc, out = _stream_run(cmd, _HUMANEVAL_TIMEOUT, progress, env=env)
    # lm-eval's own output convention: <output_path>/<sanitized model arg>/samples_<task>_<ts>.jsonl —
    # glob rather than construct the path, since the sanitization rule isn't part of its public API.
    matches = sorted(lmeval_out.glob("**/samples_humaneval_*.jsonl"), key=lambda p: p.stat().st_mtime) \
        if lmeval_out.exists() else []
    if not matches:
        return {"ok": False, "error": f"lm_eval produced no samples_humaneval_*.jsonl (rc={rc}): {out[-500:]}"}
    samples_path = matches[-1]
    score_rc, score_out = _stream_run([sys.executable, score_script, str(samples_path)],
                                      _HUMANEVAL_SCORE_TIMEOUT, progress)
    scores = parse_humaneval_score(score_out)
    if not scores:
        return {"ok": False, "error": f"humaneval_chat_score.py produced no parseable score "
                f"(rc={score_rc}): {score_out[-300:]}"}
    scores.update({"ok": True, "limit": limit, "samples": str(samples_path)})
    return scores


# --- automationbench (Zapier, agentic tool-use over simulated SaaS) ----------
_AUTOMATIONBENCH_REPO = "https://github.com/zapier/AutomationBench.git"
_AUTOMATIONBENCH_SETUP_TIMEOUT = 600  # `uv sync` fetches its own Python 3.13 + a real dep tree
_AUTOMATIONBENCH_TIMEOUT = 3 * 3600  # up to 50 tool-calling steps/task, up to 100 tasks/domain


def _automationbench_dir(cfg: dict) -> Path:
    """Vendored checkout location. This is a whole separate `uv`-managed project (its own
    pyproject.toml, Python 3.13, a real dep tree — verifiers/transformers/anthropic/etc.) —
    it doesn't fit johnny-fleet's own venv the way lm-eval/openai/datasets do, so it lives
    under state_dir (persistent, not git-tracked) rather than being a pip extra."""
    return C.get_paths().state_dir / "tools" / "automationbench"


def _ensure_automationbench(cfg: dict, progress) -> str | None:
    """None once a working `auto-bench` is available at _automationbench_dir(cfg); else an
    error string. Self-bootstraps on first use — clones zapier/AutomationBench (MIT) and
    `uv sync`s it — since (unlike the other suites) there's no lighter-weight pip install:
    it's an agent harness with its own client/runner stack, not a task+scorer pair."""
    import shutil as _shutil

    from .util import run as _run

    if not _shutil.which("uv"):
        return "`uv` not found on PATH — install it (https://docs.astral.sh/uv/) to run automationbench"
    d = _automationbench_dir(cfg)
    if not (d / "pyproject.toml").exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        progress(f"automationbench: cloning {_AUTOMATIONBENCH_REPO} → {d}…")
        rc, out, err = _run(["git", "clone", "--depth", "1", _AUTOMATIONBENCH_REPO, str(d)],
                            timeout=180)
        if rc != 0:
            return f"git clone failed (rc={rc}): {(err or out)[-300:]}"
    progress("automationbench: uv sync (first run only fetches Python 3.13 + deps)…")
    rc, out = _stream_run(["uv", "sync"], _AUTOMATIONBENCH_SETUP_TIMEOUT, progress, cwd=str(d))
    if rc != 0:
        return f"uv sync failed (rc={rc}): {out[-300:]}"
    return None


def parse_automationbench_export(path: Path) -> dict | None:
    """The exported results JSON's `meta`/`summary` (+ a per-domain pass-rate breakdown
    derived from `tasks`, since the export doesn't group by domain itself — task names are
    `<domain>.<task>`)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    summary, meta = data.get("summary") or {}, data.get("meta") or {}
    tasks = data.get("tasks") or []
    by_domain: dict[str, list[bool]] = {}
    for t in tasks:
        name = t.get("name") or t.get("id") or ""
        domain = name.split(".", 1)[0] if "." in name else "?"
        by_domain.setdefault(domain, []).append(bool(t.get("passed")))
    domains = [{"domain": d, "passed": sum(v), "total": len(v),
               "pass_rate_pct": round(100 * sum(v) / len(v), 2) if v else None}
              for d, v in sorted(by_domain.items())]
    total = summary.get("passed_count", 0) + summary.get("failed_count", 0)
    return {
        "pass_rate_pct": round(100 * summary["pass_rate"], 2) if summary.get("pass_rate") is not None else None,
        "avg_score_pct": round(100 * summary["avg_score"], 2) if summary.get("avg_score") is not None else None,
        "passed": summary.get("passed_count"), "total": total,
        "aborted": len(summary.get("aborted_tasks") or []),
        "domains": domains, "domains_run": meta.get("domains"),
        "max_steps": meta.get("max_steps"), "duration_s": meta.get("duration_seconds"),
        "export": str(path),
    }


def _run_automationbench(port: int, model_id: str, run_dir: Path, cfg: dict, domains: str,
                         num_examples: int | None, concurrency: int, thinking: bool, progress) -> dict:
    """Real agentic tool-use eval via Zapier's AutomationBench (public 600-task set: 100
    tasks each across sales/marketing/operations/support/finance/HR, fully self-contained —
    every "SaaS tool" is a local simulation, no live creds/network needed). Ships its own
    OpenAI-compatible tool-calling agent loop (`auto-bench --base-url ... --api chat_completions`)
    against the live seat's /v1/chat/completions — no harness of our own to write, just shelling
    out (see _run_humaneval for the same shape against lm-eval).

    Unlike arc/icl/needle/humaneval's short fixed-length prompts, a single task's transcript
    can grow past 40K tokens by the time tool-call history accumulates over many steps — this
    suite needs a placement with real per-request context headroom. For llama.cpp specifically,
    remember `-c` is split across `--parallel` (AGENTS.md gotcha): a 32K/parallel=4 seat only
    gives each request 8K, which aborts long tool-use transcripts outright. vLLM's max_model_len
    is already the true per-request ceiling (no such split), so it's the safer target for this
    suite as-is.

    `--extra-body chat_template_kwargs.enable_thinking` mirrors the same off-by-default
    convention arc/icl/needle/humaneval use (see their docstrings) — a reasoning model that
    burns its whole step budget inside an unclosed <think> block never gets to call a tool."""
    setup_err = _ensure_automationbench(cfg, progress)
    if setup_err:
        return {"ok": False, "error": setup_err}
    d = _automationbench_dir(cfg)
    out_dir = run_dir / "automationbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = out_dir / f"{model_id.replace('/', '__')}.json"
    # A killed/crashed run must never be mistaken for a real result: the export path is
    # fixed per model (so re-runs are easy to find), which means a prior run's file is
    # still sitting there when this one starts. Without removing it first, a run that
    # dies before writing its own output (e.g. killed mid-sweep) would leave the OLD
    # file in place — `export_path.exists()` below would still be true, and we'd
    # silently score/record a stale prior run's numbers as if they were this one's.
    # (Found for real: a killed 25-task run got reported as a "completed" 5-task result
    # left over from an earlier smoke test — same file, different --num-examples.)
    export_path.unlink(missing_ok=True)
    cmd = ["uv", "run", "auto-bench",
           "--model", model_id, "--base-url", f"http://127.0.0.1:{port}/v1",
           "--api", "chat_completions", "--api-key-var", "OPENAI_API_KEY",
           "--domains", domains, "--max-concurrent", str(concurrency),
           "--export-json", str(export_path)]
    if num_examples:
        cmd += ["--num-examples", str(num_examples)]
    if not thinking:
        cmd += ["--extra-body", json.dumps({"chat_template_kwargs": {"enable_thinking": False}})]
    env = {**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "sk-local-unused"}
    rc, out = _stream_run(cmd, _AUTOMATIONBENCH_TIMEOUT, progress, env=env, cwd=str(d))
    # Belt-and-suspenders alongside the unlink above: a killed/timed-out/crashed
    # subprocess is never trusted even if it somehow left a file behind (e.g. a
    # partial write racing the kill signal) — only a clean exit counts.
    if rc != 0:
        return {"ok": False, "error": f"auto-bench exited rc={rc} (killed/crashed/timed out): {out[-500:]}"}
    if not export_path.exists():
        return {"ok": False, "error": f"auto-bench exited 0 but produced no export JSON: {out[-500:]}"}
    scores = parse_automationbench_export(export_path)
    if not scores:
        return {"ok": False, "error": f"export JSON at {export_path} was unparseable"}
    scores.update({"ok": True, "domains_arg": domains, "num_examples": num_examples})
    return scores


def _run_ctxsafe(model_id: str, placement: dict, cfg: dict, limit: int | None, thinking: bool,
                 progress) -> dict:
    """Empirical context-safety probe (see ctxsafe.py's module docstring for the full
    methodology and the 2026-08-06 incident that motivated it).

    Unlike every other suite here, ctxsafe NEVER reuses a running seat — it
    deliberately walks depths toward (and sometimes past) the point of crashing the
    container, so it always launches its own dedicated, disposable probe seat on a
    reserved container/port (ctxsafe.VLLM_*/LLAMACPP_*) and tears it down (or cleans
    up its corpse) when done. `limit`, if given, caps the deepest depth tested — an
    escape hatch for placements whose configured/native context is too large to sweep
    to in one run (e.g. a 1M-token max_model_len); the placement's full configured cap
    is still recorded so the gap between "tested" and "configured" stays visible.
    """
    from openai import OpenAI

    from . import ctxsafe as CS
    from .backends.llamacpp import LlamaCppDriver
    from .backends.vllm import VllmDriver
    from .engine.launch import build_spec
    from .engine.placement import assign_gpus, free_gpus
    from .hardware import detect as hwd
    from .induct import stages
    from .registry import store
    from .telemetry import collect

    missing = _openai_dep_missing()
    if missing:
        return {"ok": False, "error": f"missing eval deps: {', '.join(missing)} — "
                "`pipx inject johnny-fleet openai` (or pip install 'johnny-fleet[bench]')"}

    backend = placement.get("backend") or "vllm"
    knobs = placement.get("knobs") or {}
    mml = knobs.get("max_model_len")
    if not mml:
        return {"ok": False, "error": "placement has no configured max_model_len — nothing to verify"}
    ceiling = min(mml, limit) if limit else mml
    if ceiling < CS.MIN_DEPTH_TOKENS:
        return {"ok": False, "error": f"max_model_len too small to probe usefully ({mml})"}

    reg = store.load()
    model = (reg.get("models") or {}).get(model_id) or {}
    native_context = (model.get("capabilities") or {}).get("native_context")
    path = _local_path(model_id, reg, cfg)
    if not path:
        return {"ok": False, "error": f"no local weights found for {model_id} (identity.local_path missing?)"}

    hw = hwd.detect()
    point = point_from_placement(placement)
    is_cpu = point.get("device") == "cpu"
    gpus: list[int] = []
    if not is_cpu:
        n_gpus = point.get("tp") or point.get("gpu_count") or 1
        gpus = assign_gpus(n_gpus, hw, free_gpus(hw, all_seats(cfg)))
        if len(gpus) < n_gpus:
            return {"ok": False, "error": f"insufficient free GPUs for {n_gpus} — down a seat first"}

    if backend == "llamacpp":
        drv = LlamaCppDriver(image=placement.get("image") or C.resolve_image(cfg, backend="llamacpp", model_id=model_id))
        port, container = CS.LLAMACPP_PORT, CS.LLAMACPP_CONTAINER
        spec = build_spec(model_id, model, placement, gpus, port, cfg, hw)
        spec["container_name"] = container
    elif backend == "vllm":
        drv = VllmDriver(image=C.resolve_image(cfg, device="cpu" if is_cpu else "gpu", model_id=model_id))
        port, container = CS.VLLM_PORT, CS.VLLM_CONTAINER
        spec = stages._cpu_tuning_spec(model_id, path, point, cfg) if is_cpu \
            else stages._tuning_spec(model_id, path, point, gpus, cfg, hw)
        spec["extra"] = {**spec.get("extra", {}), **(placement.get("extra") or {})}
        spec["container_name"] = container
        spec["port"] = port
    else:
        return {"ok": False, "error": f"backend {backend!r} isn't ctxsafe-wired yet (vllm/llamacpp only)"}

    progress(f"ctxsafe: launching dedicated probe seat from {placement.get('id')} on "
             + ("CPU" if is_cpu else f"GPU {gpus}") + f" (container={container}, port={port}, "
             f"ceiling={ceiling:,} tok" + (f", capped from configured {mml:,}" if ceiling < mml else "") + ")…")
    drv.launch(spec)
    ready, why = stages._wait_ready(drv, container, port, timeout=_CTXSAFE_LAUNCH_TIMEOUT)
    if not ready:
        tail = stages._diagnose(drv, container)
        drv.stop(container)
        return {"ok": False, "error": (why or "probe seat did not become ready") + (f" — {tail}" if tail else "")}

    collect.add_pin(container)
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY")
    depths = CS.plan_depths(ceiling)
    tested: list[dict] = []
    crashed_at: int | None = None
    verified_safe: int | None = None
    try:
        for i, depth in enumerate(depths):
            trials_needed = CS.TRIALS_NEAR_CEILING if i >= len(depths) - 2 else 1
            timeout_s = CS.request_timeout_s(depth)
            trial_results = []
            depth_crashed = False
            for t in range(trials_needed):
                progress(f"ctxsafe: depth={depth:,} tok, trial {t + 1}/{trials_needed} "
                         f"(timeout {timeout_s:.0f}s)…")
                r = CS.run_trial(client, model_id, depth, t, container, drv, timeout_s, thinking=thinking)
                trial_results.append(r)
                outcome = "CRASH" if r.get("crashed") else ("PASS" if r.get("ok") else "FAIL (no crash)")
                progress(f"ctxsafe: depth={depth:,} trial {t + 1} -> {outcome} · "
                         f"VRAM peak {r.get('vram_peak_gb')}GB · {r.get('elapsed_s')}s")
                if r.get("crashed"):
                    depth_crashed = True
                    break
            passed = (not depth_crashed) and all(r.get("ok") for r in trial_results)
            tested.append({"depth": depth, "passed": passed, "crashed": depth_crashed,
                           "trials": trial_results})
            if depth_crashed:
                crashed_at = depth
                break
            if passed:
                verified_safe = depth
            else:
                break  # a non-crash failure (bad recall) — deeper depths aren't trustworthy either
    finally:
        collect.remove_pin(container)
        try:
            drv.stop(container)
        except Exception:
            pass
        progress("ctxsafe: probe seat stopped")

    peaks = [tr.get("vram_peak_gb") or 0 for d in tested for tr in d["trials"]]
    vram_peak_gb = round(max(peaks), 2) if peaks else None
    ok = verified_safe is not None
    return {
        "ok": ok,
        "verified_safe_tokens": verified_safe,
        "crashed_at": crashed_at,
        "configured_max_model_len": mml,
        "native_context": native_context,
        "ceiling_tested": ceiling,
        "limited": bool(limit and limit < mml),
        "vram_peak_gb": vram_peak_gb,
        "tested_depths": tested,
        "error": None if ok else "no depth passed safely — see tested_depths",
    }


# bench.sh's default ramp (16..1024, twice) assumes a wide-batch seat. Against a seat
# capped at a handful of sequences (personal seats run max_num_seqs=4) every level
# above the cap just queues — the 2026-09-04 Flash-Next run put 800+ requests in
# vLLM's wait queue, hit the 20-min perf wall, and the abandoned queue then stalled
# the *next* suite's first request for ten minutes. Size the ramp to the seat.
_BENCH_DEFAULT_LEVELS = (16, 32, 64, 128, 256, 512, 1024)


def _perf_sweep_env(point: dict) -> dict:
    """Env overrides for bench.sh sized to the placement's max_num_seqs (empty = defaults)."""
    seqs = point.get("max_num_seqs")
    try:
        seqs = int(seqs) if seqs else None
    except (TypeError, ValueError):
        seqs = None
    if not seqs or seqs >= _BENCH_DEFAULT_LEVELS[-1]:
        return {}
    levels = [n for n in _BENCH_DEFAULT_LEVELS if n <= seqs]
    if not levels:  # seqs < 16: short doubling ladder that ends at the cap
        n = 1
        while n < seqs:
            levels.append(n)
            n *= 2
        levels.append(seqs)
    elif levels[-1] != seqs:
        levels.append(seqs)
    return {"BENCH_CONCURRENCY": " ".join(str(n) for n in levels),
            "WARMUP_CONCURRENCY": str(min(24, seqs))}


def _run_perf(port: int, container: str | None, model_id: str, point: dict, cfg: dict, progress,
             backend: str = "vllm") -> dict:
    """Induction's throughput bench against an already-running seat (no teardown).

    Both bench scripts are plain HTTP clients (/v1/completions) — backend-agnostic in
    principle — but bench.sh's concurrency sweep (16..1024) assumes vLLM's much wider
    parallelism; a GGUF seat has few slots (its `--parallel` knob) and is far slower
    per-token, so llamacpp uses the lighter bundled bench_llamacpp.sh (concurrency
    1..32) instead — same output shape (_parse_bench reads either). Only the KV-cache
    readback below is vLLM-specific: it parses vLLM's startup log lines ('GPU KV cache
    size: N tokens' etc, see stages._parse_kv_cache) and llama-server logs a different
    shape with no equivalent parser here yet, so llamacpp skips this enrichment rather
    than misparsing or erroring — the throughput numbers below still land either way.
    """
    from .bundled import resolve_script
    from .util import run as _run

    result: dict = {}
    if container and backend == "vllm":  # KV readback is best-effort — startup lines scroll off long-lived seats
        try:
            from .backends.vllm import VllmDriver

            drv = VllmDriver(image=C.resolve_image(cfg, device="cpu" if point.get("device") == "cpu" else "gpu",
                                                   model_id=model_id))
            result.update({k: v for k, v in stages._parse_kv_cache(
                drv.logs(container, tail=2000) or "").items() if v is not None})
        except Exception:
            pass
    if point.get("embeddings"):
        parsed = stages._bench_embeddings(port, model_id)
    else:
        script_key = "bench_llamacpp" if backend == "llamacpp" else "bench"
        script = resolve_script(script_key, cfg)
        if not script:
            return {"ok": False, "error": f"{script_key} script unavailable "
                    f"(not bundled and no scripts.{script_key} override)"}
        progress(f"perf: {script_key} sweep (concurrency ramp + single-stream)…")
        # Default 20 min fits normal seats; CPU-MoE giants (single-digit tok/s)
        # need longer — override per-run, e.g. JOHNNY_BENCH_PERF_TIMEOUT=7200.
        perf_timeout = float(os.environ.get("JOHNNY_BENCH_PERF_TIMEOUT", 1200))
        sweep_env = _perf_sweep_env(point) if script_key == "bench" else {}
        if sweep_env:
            progress(f"perf: ramp sized to max_num_seqs={point.get('max_num_seqs')} "
                     f"(BENCH_CONCURRENCY={sweep_env['BENCH_CONCURRENCY']})")
        rc, out, errout = _run(["bash", script, str(port), model_id], timeout=perf_timeout,
                               env={**os.environ, **sweep_env} if sweep_env else None)
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
        if suite == "ctxsafe" and r.get("tested_depths") is not None:
            # A crash (ok=False) is exactly this suite's reason to exist — render the
            # full depth table either way, not just "failed: ...".
            gap = (f"native_context={r.get('native_context')} · "
                   f"configured max_model_len={r.get('configured_max_model_len')} · "
                   f"ceiling_tested={r.get('ceiling_tested')}"
                   + (" (capped by --limit, not the full configured cap)" if r.get("limited") else ""))
            lines.append(gap)
            lines.append(f"**verified_safe_tokens = {r.get('verified_safe_tokens')}** · "
                         f"crashed_at = {r.get('crashed_at')} · VRAM peak {r.get('vram_peak_gb')}GB")
            lines.append("")
            for d in r.get("tested_depths") or []:
                trial_strs = []
                for tr in d.get("trials") or []:
                    out = "CRASH" if tr.get("crashed") else ("PASS" if tr.get("ok") else "FAIL")
                    detail = f"({tr.get('vram_peak_gb')}GB, {tr.get('elapsed_s')}s)"
                    if not tr.get("ok") and not tr.get("crashed"):
                        snippet = tr.get("response_snippet") or tr.get("error") or ""
                        detail += f" [{snippet[:80]!r}]"
                    trial_strs.append(f"{out}{detail}")
                verdict = "CRASHED" if d.get("crashed") else ("safe" if d.get("passed") else "failed (no crash)")
                lines.append(f"- depth={d['depth']:,} tok — {verdict} — " + ", ".join(trial_strs))
            if not r.get("ok") and r.get("error") and not r.get("crashed_at") and not r.get("verified_safe_tokens"):
                lines.append("")
                lines.append(f"note: {r.get('error')}")
        elif not r.get("ok"):
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
        elif suite == "icl":
            total = (r.get("pass") or 0) + (r.get("fail") or 0)
            breakdown = ", ".join(f"{b['category']}={b['score']}" for b in (r.get("category_breakdown") or []))
            lines.append(f"ICL probe {r.get('pass')}/{total}"
                         + (f" (limit={r['limit']})" if r.get("limit") else "")
                         + (f" · {breakdown}" if breakdown else ""))
        elif suite == "needle":
            total = (r.get("pass") or 0) + (r.get("fail") or 0)
            pb = r.get("position_bias") or {}
            lines.append(f"Code Needle {r.get('pass')}/{total}"
                         + (f" ({r['corpus_tokens']:,} corpus tokens)" if r.get("corpus_tokens") else "")
                         + (" · position " + ", ".join(f"{k}={v}" for k, v in pb.items()) if pb else ""))
        elif suite == "depth":
            for p in (r.get("points") or []):
                lines.append(f"depth={p['depth']}: pp {p['pp_tok_s']} tok/s · tg {p['tg_tok_s']} tok/s · "
                             f"ttfr {p['ttfr_ms']}ms")
        elif suite == "humaneval":
            lines.append(f"HumanEval pass@1 {r.get('pass_at_1_pct')}% ({r.get('passed')}/{r.get('total')}"
                         + (f", limit={r['limit']}" if r.get("limit") else "") + ")")
        elif suite == "planbench":
            lines.append(f"PlanBench ({r.get('task')}) exact-plan {r.get('exact_pct')}% "
                         f"({r.get('exact')}/{r.get('total')}) · plan-prefix {r.get('plan_prefix_pct')}% "
                         f"· domains={r.get('domains')} · errors={r.get('errors')}")
            lines.append("  (exact match vs the reference plan is a strict lower bound — a valid "
                         "alternative plan scores 0; prefix is partial credit)")
        elif suite == "automationbench":
            lines.append(f"AutomationBench pass rate {r.get('pass_rate_pct')}% "
                         f"({r.get('passed')}/{r.get('total')}, avg partial credit {r.get('avg_score_pct')}%) "
                         f"· domains={r.get('domains_run')} · aborted={r.get('aborted')}"
                         + (f" · max_steps={r['max_steps']}" if r.get("max_steps") else ""))
            for dm in (r.get("domains") or []):
                lines.append(f"  - {dm['domain']}: {dm['passed']}/{dm['total']} "
                             f"({dm.get('pass_rate_pct')}%)")
        lines.append("")
    path = run_dir / "BENCH_REPORT.md"
    path.write_text("\n".join(lines))
    return path


def run(model_id: str, placement: dict, suites: list[str], cfg: dict | None = None,
        limit: int | None = None, concurrency: int = 8, thinking: bool = False,
        automationbench_domains: str = "all", progress=None) -> dict:
    """Bench one placement: reuse its running seat or launch a temp tuning seat, run the
    suites, write scores to the registry + a BENCH_REPORT. Returns per-suite results.

    automationbench_domains: comma-separated domain filter (or "all") — see
    _run_automationbench. Only consulted when "automationbench" is in suites; the public
    set is 600 tasks (100/domain), so a real first run usually wants this narrowed."""
    _p = progress or (lambda *_: None)
    cfg = cfg if cfg is not None else load_config()
    pid = placement.get("id") or "?"
    backend = placement.get("backend") or "vllm"
    if backend not in ("vllm", "llamacpp"):
        return {"error": f"backend {backend!r} isn't bench-wired yet (vllm/llamacpp only)"}
    point = point_from_placement(placement)
    run_dir = C.get_paths().runs_dir / f"bench-{model_id.replace('/', '__')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ctxsafe never shares the generic seat below — it deliberately walks depths
    # toward (and sometimes past) a crash, so it always launches its own dedicated,
    # disposable probe seat (see _run_ctxsafe) and must never touch a live production
    # seat or the shared tuning seat other suites reuse. Only set up the shared seat
    # if some OTHER suite actually needs it.
    other_suites = [s for s in suites if s != "ctxsafe"]
    seat = find_running_seat(model_id, pid, cfg) if other_suites else None
    drv = None
    port = container = None
    if other_suites and seat is not None:
        port, container = seat.port, getattr(seat, "name", None)
        _p(f"reusing running seat {container} (port {port}) — no relaunch")
    elif other_suites:
        reg = store.load()
        model = (reg.get("models") or {}).get(model_id) or {}
        path = _local_path(model_id, reg, cfg)
        if not path:
            return {"error": f"no local weights found for {model_id} (identity.local_path missing?)"}
        hw = hwd.detect()
        is_cpu = point.get("device") == "cpu"
        gpus: list[int] = []
        if not is_cpu:
            n_gpus = point.get("tp") or point.get("gpu_count") or 1
            gpus = assign_gpus(n_gpus, hw, free_gpus(hw, all_seats(cfg)))
            if len(gpus) < n_gpus:
                return {"error": f"insufficient free GPUs for {n_gpus} — down a seat first"}
        if backend == "llamacpp":
            # llamacpp induction only ever benches via llama-bench (induct/llamacpp.py
            # tune_point) — there's no server-spec builder there to mirror. Reuse
            # engine.launch.build_spec instead: it already turns a placement's own
            # (backend-native) knobs/extra/env into a launch-ready spec — the exact
            # path `johnny up` uses for a real seat — just pinned onto a dedicated
            # bench container/port so it can never collide with one.
            from .backends.llamacpp import LlamaCppDriver
            from .engine.launch import build_spec

            drv = LlamaCppDriver(image=placement.get("image") or C.resolve_image(cfg, backend="llamacpp", model_id=model_id))
            spec = build_spec(model_id, model, placement, gpus, _LLAMACPP_TUNING_PORT, cfg, hw)
            spec["container_name"] = _LLAMACPP_TUNING_CONTAINER
            port, container = _LLAMACPP_TUNING_PORT, _LLAMACPP_TUNING_CONTAINER
        else:
            from .backends.vllm import VllmDriver

            drv = VllmDriver(image=C.resolve_image(cfg, device="cpu" if is_cpu else "gpu", model_id=model_id))
            spec = stages._cpu_tuning_spec(model_id, path, point, cfg) if is_cpu \
                else stages._tuning_spec(model_id, path, point, gpus, cfg, hw)
            # Bench what production runs: carry the placement's parsers/template into the seat.
            spec["extra"] = {**spec.get("extra", {}), **(placement.get("extra") or {})}
            port, container = stages.TUNING_PORT, stages.TUNING_CONTAINER
        _p(f"launching temp seat from {pid} on " + ("CPU" if is_cpu else f"GPU {gpus}") + "…")
        drv.launch(spec)
        # Same slow-ready path as induction's tune_point: Qwen3.8-27B dense takes ~9.5 min
        # to serve (load + compile + capture) — 600s killed healthy seats (2026-08-17).
        ready, why = stages._wait_ready(drv, container, port, timeout=900 if is_cpu else 1200)
        if not ready:
            tail = stages._diagnose(drv, container)
            drv.stop(container)
            return {"error": (why or "seat did not become ready") + (f" — {tail}" if tail else "")}

    results: dict = {}
    if other_suites:
        collect.add_pin(container)  # reaper-safe while benching (borrowed seats too)
    try:
        for s in suites:
            if s == "perf":
                results["perf"] = _run_perf(port, container, model_id, point, cfg, _p, backend=backend)
            elif s == "ctxsafe":
                results["ctxsafe"] = _run_ctxsafe(model_id, placement, cfg, limit, thinking, _p)
            elif s == "arc":
                if point.get("embeddings"):
                    results["arc"] = {"ok": False, "error": "embeddings model — ARC needs a generative seat"}
                else:
                    results["arc"] = _run_arc(port, model_id, run_dir, cfg, limit, concurrency, thinking, _p)
            elif s == "planbench":
                if point.get("embeddings"):
                    results["planbench"] = {"ok": False, "error": "embeddings model — PlanBench needs a generative seat"}
                else:
                    results["planbench"] = _run_planbench(port, model_id, run_dir, cfg, limit, concurrency, thinking, _p)
            elif s == "icl":
                if point.get("embeddings"):
                    results["icl"] = {"ok": False, "error": "embeddings model — ICL needs a generative seat"}
                else:
                    results["icl"] = _run_icl(port, model_id, run_dir, cfg, limit, thinking, _p)
            elif s == "needle":
                if point.get("embeddings"):
                    results["needle"] = {"ok": False, "error": "embeddings model — needle needs a generative seat"}
                else:
                    results["needle"] = _run_needle(port, model_id, run_dir, cfg, thinking, _p)
            elif s == "depth":
                if point.get("embeddings"):
                    results["depth"] = {"ok": False, "error": "embeddings model — depth bench needs a generative seat"}
                else:
                    results["depth"] = _run_depth(port, model_id, run_dir, cfg, _p)
            elif s == "humaneval":
                if point.get("embeddings"):
                    results["humaneval"] = {"ok": False, "error": "embeddings model — HumanEval needs a generative seat"}
                else:
                    results["humaneval"] = _run_humaneval(port, model_id, run_dir, cfg, limit, concurrency,
                                                           thinking, _p)
            elif s == "automationbench":
                if point.get("embeddings"):
                    results["automationbench"] = {"ok": False, "error": "embeddings model — automationbench needs a generative + tool-calling seat"}
                else:
                    results["automationbench"] = _run_automationbench(port, model_id, run_dir, cfg, automationbench_domains,
                                                                       limit, concurrency, thinking, _p)
            ok = results[s].get("ok")
            _p(f"{s}: " + ("done" if ok else f"FAILED — {results[s].get('error')}"))
    finally:
        if other_suites:
            collect.remove_pin(container)
            if drv is not None:  # only tear down what we launched
                drv.stop(container)
                _p("temp seat stopped")

    perf = results.get("perf") if (results.get("perf") or {}).get("ok") else None
    quality: dict = {}
    date = time.strftime("%Y-%m-%d")
    arc = results.get("arc")
    if arc and arc.get("ok"):
        quality["arc"] = {k: arc.get(k) for k in
                          ("accuracy_pct", "correct", "total", "no_extraction", "api_errors", "limit")} \
                         | {"date": date}
    icl = results.get("icl")
    if icl and icl.get("ok"):
        quality["icl"] = {k: icl.get(k) for k in ("pass", "fail", "category_breakdown", "limit")} | {"date": date}
    needle = results.get("needle")
    if needle and needle.get("ok"):
        quality["needle"] = {k: needle.get(k) for k in
                             ("pass", "fail", "position_bias", "corpus_tokens")} | {"date": date}
    depth = results.get("depth")
    if depth and depth.get("ok"):
        quality["depth"] = {"points": depth.get("points"), "date": date}
    humaneval = results.get("humaneval")
    if humaneval and humaneval.get("ok"):
        quality["humaneval"] = {k: humaneval.get(k) for k in
                                ("pass_at_1_pct", "passed", "total", "limit")} | {"date": date}
    automationbench = results.get("automationbench")
    if automationbench and automationbench.get("ok"):
        quality["automationbench"] = {k: automationbench.get(k) for k in
                                      ("pass_rate_pct", "avg_score_pct", "passed", "total", "aborted",
                                       "domains", "domains_run", "max_steps", "num_examples")} | {"date": date}
    ctxsafe = results.get("ctxsafe")
    if ctxsafe is not None and ctxsafe.get("tested_depths") is not None:
        # Recorded even on a failed/crashed sweep (ok=False, verified_safe_tokens possibly
        # null or lower than configured) — a crash finding is exactly the thing this suite
        # exists to surface, not something to silently drop from the registry. Not recorded
        # when the probe never actually ran (missing deps, no free GPUs, seat wouldn't come
        # up) — that's an infra error, not a safety finding.
        tested_summary = [{"depth": d["depth"], "passed": d["passed"], "crashed": d["crashed"]}
                          for d in (ctxsafe.get("tested_depths") or [])]
        quality["ctxsafe"] = {
            "verified_safe_tokens": ctxsafe.get("verified_safe_tokens"),
            "crashed_at": ctxsafe.get("crashed_at"),
            "configured_max_model_len": ctxsafe.get("configured_max_model_len"),
            "native_context": ctxsafe.get("native_context"),
            "ceiling_tested": ctxsafe.get("ceiling_tested"),
            "limited": ctxsafe.get("limited"),
            "tested_depths": tested_summary,
            "vram_peak_gb": ctxsafe.get("vram_peak_gb"),
            "date": date,
        }
    quality = quality or None
    wrote = write_scores(model_id, pid, perf=perf, quality=quality) if (perf or quality) else False
    report = write_report(run_dir, model_id, pid, results)
    seat_desc = "reused" if (other_suites and seat is not None) else \
        ("temporary" if other_suites else "ctxsafe-dedicated")
    return {"model_id": model_id, "placement_id": pid, "results": results,
            "registry_updated": wrote, "report": str(report), "seat": seat_desc}
