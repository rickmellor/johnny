"""Write induction artifacts: a TUNING_REPORT and the registry placement."""

from __future__ import annotations

from pathlib import Path

from .. import config as C
from ..registry import store


def _point_sig(point: dict) -> str:
    if point.get("device") == "cpu":
        return (f"cpu-{point.get('cpuset')}-mml{point.get('max_model_len')}"
                f"-bt{point.get('max_num_batched_tokens')}-seqs{point.get('max_num_seqs')}")
    return (
        f"tp{point.get('tp')}-gmu{point.get('gpu_memory_util')}-seqs{point.get('max_num_seqs')}"
        f"-bt{point.get('max_num_batched_tokens')}-mml{point.get('max_model_len')}"
    )


def to_placement(model_id: str, winner: dict, audit: dict, hardware, runtime_version: str, use_case: str | None) -> dict:
    point = winner["point"]
    vkey = {"hardware_fingerprint": hardware.fingerprint, "backend": "vllm", "runtime_version": runtime_version}
    perf = {"peak_tok_s": winner.get("peak_tok_s"), "single_stream_tok_s": winner.get("single_tok_s")}
    if point.get("device") == "cpu":
        extra = {"device": "cpu", "cpuset": point.get("cpuset")}
        if point.get("embeddings"):
            extra["runner"] = "pooling"
            extra["trust_remote_code"] = True
        return {
            "id": f"induct-cpu-{point.get('cpuset')}",
            "backend": "vllm",
            "image": None,
            "use_case": use_case,
            "knobs": {
                "gpu_count": 0, "tensor_parallel_size": None,
                "max_model_len": point.get("max_model_len"),
                "gpu_memory_util": None,
                "max_num_seqs": point.get("max_num_seqs"),
                "max_num_batched_tokens": point.get("max_num_batched_tokens"),
                "kv_cache_dtype": "auto", "mtp": {"enabled": False},
            },
            "extra": extra,
            "env": {"VLLM_CPU_KVCACHE_SPACE": "4"},
            "perf": perf,
            "validation_key": vkey,
            "validated_at": None,
            "source": "induction",
        }
    return {
        "id": f"induct-{_point_sig(point)}",
        "backend": "vllm",
        "image": None,
        "use_case": use_case,
        "knobs": {
            "gpu_count": point["tp"],
            "tensor_parallel_size": point["tp"],
            "quant": point.get("quant"),
            "max_model_len": point.get("max_model_len"),
            "gpu_memory_util": point.get("gpu_memory_util"),
            "max_num_seqs": point.get("max_num_seqs"),
            "max_num_batched_tokens": point.get("max_num_batched_tokens"),
            "kv_cache_dtype": point.get("kv_cache_dtype", "auto"),
            "mtp": point.get("mtp") or {"enabled": False},
        },
        "extra": {},
        "env": {},
        "perf": {"peak_tok_s": winner.get("peak_tok_s"), "single_stream_tok_s": winner.get("single_tok_s")},
        "validation_key": {"hardware_fingerprint": hardware.fingerprint, "backend": "vllm", "runtime_version": runtime_version},
        "validated_at": None,
        "source": "induction",
    }


def write_placement(model_id: str, audit: dict, placement: dict, hardware, local_path: str | None = None) -> None:
    reg = store.load()
    models = reg.setdefault("models", {})
    m = models.setdefault(model_id, {
        "identity": {}, "capabilities": {}, "placements": [], "lifecycle": {},
    })
    ident = m["identity"]
    ident.setdefault("repo_id", model_id)
    if local_path:
        ident.setdefault("local_path", local_path)
    ident.setdefault("arch", audit.get("arch"))
    ident.setdefault("quant", audit.get("quant"))
    cap = m["capabilities"]
    cap.setdefault("multimodal", audit.get("multimodal"))
    cap.setdefault("mtp_head", audit.get("mtp_head"))
    cap.setdefault("native_context", audit.get("native_context") or (audit.get("dims") or {}).get("ctx"))
    # replace any prior induction placement with the same id; else append
    m["placements"] = [p for p in m["placements"] if p.get("id") != placement["id"]]
    m["placements"].append(placement)
    fps = set(reg.get("fingerprints") or []) | {hardware.fingerprint}
    reg["fingerprints"] = sorted(fps)
    store.save(reg)


def write_report(run_dir: Path, model_id: str, audit: dict, results: list[dict], winner: dict | None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# TUNING_REPORT — {model_id}",
        "",
        f"- arch: {audit.get('arch')}  quant: {audit.get('quant')}  "
        f"size: {audit.get('size_bytes', 0) / 1e9:.1f} GB  native_ctx: {(audit.get('dims') or {}).get('ctx')}",
        "",
        "## Sweep",
        "",
        "| tp | gmu | seqs | batched | mml | KV tok | conc | peak tok/s | single tok/s | ok |",
        "|----|-----|------|---------|-----|--------|------|-----------|--------------|----|",
    ]
    for r in results:
        p = r["point"]
        kv = r.get("kv_cache_tokens")
        kv_s = f"{kv/1e6:.2f}M" if kv else "—"
        conc = f"{r.get('max_concurrency')}x" if r.get("max_concurrency") else "—"
        lines.append(
            f"| {p.get('tp')} | {p.get('gpu_memory_util')} | {p.get('max_num_seqs')} | "
            f"{p.get('max_num_batched_tokens')} | {p.get('max_model_len')} | {kv_s} | {conc} | "
            f"{r.get('peak_tok_s')} | {r.get('single_tok_s')} | {'✓' if r.get('ok') else '✗'} |"
        )
    if winner:
        wp = winner["point"]
        lines += ["", f"## Winner\n\nTP={wp.get('tp')} gmu={wp.get('gpu_memory_util')} "
                  f"mml={wp.get('max_model_len')} → peak {winner.get('peak_tok_s')} tok/s, "
                  f"single {winner.get('single_tok_s')} tok/s"]
    path = run_dir / "TUNING_REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path
