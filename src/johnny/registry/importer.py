"""Import bash launchers into the registry (§3.4).

Parses each `~/vllm/launchers/*.sh` for its vLLM flags (the same grep approach the
old bash johnny used, extended), cross-references the model's config.json via the
vLLM driver's probe_model for arch/quant/multimodal/MTP, and emits one **placement**
per launcher stamped with a validation key (hardware fingerprint × backend ×
runtime/image version) and `source = imported`. Multiple launchers for the same
served-model become multiple placements on one model — exactly the round-trip the
P2 verify checks. Measured perf/quality from TUNING_REPORTs is left for induction to
fill (best-effort here = empty).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..backends.vllm import VllmDriver

_QUANT_TOKENS = [("fp8", "fp8"), ("awq", "awq"), ("int4", "int4"), ("gptq", "gptq"), ("bf16", "bf16")]


def _search(pattern: str, text: str, group: int = 1):
    m = re.search(pattern, text)
    return m.group(group) if m else None


def quant_from_path(s: str | None) -> str | None:
    if not s:
        return None
    low = s.lower()
    for tok, val in _QUANT_TOKENS:
        if tok in low:
            return val
    return None


def parse_launcher(path: Path) -> dict:
    text = Path(path).read_text()
    gpus_raw = _search(r"(?:HIP|CUDA)_VISIBLE_DEVICES=([\d,]+)", text)
    gpus = [int(x) for x in gpus_raw.split(",") if x.strip().isdigit()] if gpus_raw else []

    mtp = None
    if "--speculative-config" in text and "mtp" in text:
        n = _search(r'num_speculative_tokens"?\s*:\s*(\d+)', text)
        mtp = {"enabled": True, "num_speculative_tokens": int(n) if n else None}

    # Port: literal -p first, else a ${VAR:-NNNN} default.
    port = _search(r'-p\s+"?(\d+):8000', text) or _search(r":-(\d+)\}", text)

    return {
        "file": str(path),
        "container": _search(r'CONTAINER_NAME=["\']?([\w.\-]+)', text),
        "nickname": _search(r'MODEL_NICKNAME=["\']?([^"\'\n]+)', text),
        "image": _search(r"(vllm/[\w.\-]+:[\w.\-]+)", text),
        "model_path": _search(r"(/models/\S+)", text),
        "served": _search(r"--served-model-name\s+([^\s\\]+)", text),
        "tp": _int(_search(r"--tensor-parallel-size\s+(\d+)", text)),
        "mml": _int(_search(r"--max-model-len\s+(\d+)", text)),
        "gmu": _float(_search(r"--gpu-memory-utilization\s+([\d.]+)", text)),
        "seqs": _int(_search(r"--max-num-seqs\s+(\d+)", text)),
        "batched": _int(_search(r"--max-num-batched-tokens\s+(\d+)", text)),
        "kv": _search(r"--kv-cache-dtype\s+([^\s\\]+)", text),
        "tool_parser": _search(r"--tool-call-parser\s+([^\s\\]+)", text),
        "reasoning_parser": _search(r"--reasoning-parser\s+([^\s\\]+)", text),
        "mtp": mtp,
        "gpus": gpus,
        "port": _int(port),
        "runner_pooling": bool(re.search(r"--runner\s+pooling", text)),
        "env": dict(re.findall(r"-e\s+([A-Z_]+)=([^\s\\]+)", text)),
    }


def _int(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _guess_use_case(name: str) -> str | None:
    n = name.lower()
    if "coder" in n:
        return "latency"
    if "orchestrator" in n or "orch" in n:
        return "context"
    if "batch" in n:
        return "throughput"
    return None


def _model_skeleton(local_path: str | None, vendor: str | None, probe: dict, quant: str | None) -> dict:
    return {
        "identity": {
            "repo_id": local_path,
            "local_path": local_path,
            "vendor": vendor,
            "arch": probe.get("arch"),
            "params": None,
            "quant": probe.get("quant") or quant,
        },
        "capabilities": {
            "multimodal": probe.get("multimodal"),
            "mtp_head": probe.get("mtp_head"),
            "native_context": probe.get("native_context"),
            "tool_call_parser": None,
            "reasoning_parser": None,
            "thinking_toggle": None,
            "chat_template": None,
        },
        "placements": [],
        "lifecycle": {"last_loaded": None, "last_served": None, "last_tuned": None},
    }


def _build_placement(L: dict, fingerprint: str, quant: str | None, runtime_version: str) -> dict:
    pooling = L.get("runner_pooling")
    return {
        "id": Path(L["file"]).stem,
        "backend": "vllm",
        "use_case": _guess_use_case(L["file"]),
        "knobs": {
            "gpu_count": 0 if pooling else (len(L["gpus"]) or L.get("tp")),
            "tensor_parallel_size": None if pooling else L.get("tp"),
            "quant": quant,
            "max_model_len": L.get("mml"),
            "gpu_memory_util": L.get("gmu"),
            "max_num_seqs": L.get("seqs"),
            "max_num_batched_tokens": L.get("batched"),
            "kv_cache_dtype": L.get("kv") or "auto",
            "mtp": L.get("mtp") or {"enabled": False},
        },
        "extra": {
            "tool_call_parser": L.get("tool_parser"),
            "reasoning_parser": L.get("reasoning_parser"),
            "runner": "pooling" if pooling else None,
            "device": "cpu" if pooling else None,
            "container_name": L.get("container"),
            "nickname": L.get("nickname"),
            "gpus": L.get("gpus"),
            "port_hint": L.get("port"),
        },
        "env": L.get("env") or {},
        "perf": {},  # filled by induction / a TUNING_REPORT pass later
        "validation_key": {
            "hardware_fingerprint": fingerprint,
            "backend": "vllm",
            "runtime_version": runtime_version,
        },
        "validated_at": None,
        "source": "imported",
    }


def import_launchers(launchers_dir: str, models_dir: str | None, fingerprint: str) -> dict:
    from .. import config as C

    reg = C.registry_stub()
    drv = VllmDriver(models_dir=models_dir)
    ldir = Path(launchers_dir).expanduser()
    if not ldir.exists():
        return reg

    for f in sorted(ldir.glob("*.sh")):
        L = parse_launcher(f)
        if not L.get("served"):
            continue
        mp = L.get("model_path")
        local_path = mp.replace("/models/", "", 1).rstrip("/\\ ") if mp else None
        vendor = local_path.split("/")[0] if local_path else None
        image = L.get("image") or ""
        runtime_version = image.split(":")[-1] if ":" in image else "unknown"

        probe: dict = {}
        if local_path and models_dir:
            full = Path(models_dir).expanduser() / local_path
            if full.exists():
                probe = drv.probe_model(str(full))
        quant = probe.get("quant") or quant_from_path(local_path) or quant_from_path(f.name)

        model_id = L["served"]
        if model_id not in reg["models"]:
            reg["models"][model_id] = _model_skeleton(local_path, vendor, probe, quant)
        m = reg["models"][model_id]
        cap = m["capabilities"]
        cap["tool_call_parser"] = cap["tool_call_parser"] or L.get("tool_parser")
        cap["reasoning_parser"] = cap["reasoning_parser"] or L.get("reasoning_parser")
        cap["native_context"] = cap["native_context"] or L.get("mml")
        m["placements"].append(_build_placement(L, fingerprint, quant, runtime_version))

    reg["fingerprints"] = [fingerprint] if fingerprint else []
    return reg
