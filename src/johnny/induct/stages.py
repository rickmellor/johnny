"""Induction stages — discover / audit / hardware-fit / tune / synthesize.

The tune stage reuses bench.sh against a throwaway tuning seat (container
`vllm-johnny-tuning`, port 9000 — the separate tuning stack, never production) and
the vLLM driver to launch it; it's the one heavy stage (a launch + bench per point).
Everything before it is launch-free.
"""

from __future__ import annotations

import re
from pathlib import Path

from .. import config as C
from ..backends.vllm import VllmDriver
from ..runtime import probe
from . import grid

TUNING_CONTAINER = "vllm-johnny-tuning"
TUNING_PORT = 9000


def discover(model_ref: str, cfg: dict) -> tuple[str, str]:
    """Resolve a ref (local path or 'vendor/name') to (model_id, local_path).

    HF download is P5; here the model must already be on disk.
    """
    p = Path(model_ref).expanduser()
    if p.exists() and (p / "config.json").exists():
        return p.name, str(p)
    models_dir = (cfg.get("roots") or {}).get("models_dir")
    if models_dir:
        full = Path(models_dir).expanduser() / model_ref
        if (full / "config.json").exists():
            return Path(model_ref).name, str(full)
    raise FileNotFoundError(
        f"model '{model_ref}' not found on disk (download/acquire arrives at P5; "
        f"place it under {models_dir} or pass a local path)"
    )


def audit(path: str) -> dict:
    drv = VllmDriver()
    info = drv.probe_model(path)
    config = grid.read_config(path)
    info["dims"] = grid.dims(config)
    info["size_bytes"] = grid.model_size_bytes(path)
    if not info.get("quant"):
        low = path.lower()
        for tok in ("fp8", "awq", "int4", "bf16"):
            if tok in low:
                info["quant"] = tok
                break
    return info


_GENERATIVE_ARCH = ("ForCausalLM", "ForConditionalGeneration", "LMHeadModel", "ForSeq2SeqLM")


def is_embeddings(audit_info: dict) -> bool:
    """Heuristic: an encoder/bert-like arch with no generative head is a pooling/
    embeddings model (e.g. NomicBertModel). Override with --embeddings/--no-embeddings."""
    arch = audit_info.get("arch") or ""
    if any(g in arch for g in _GENERATIVE_ARCH):
        return False
    return bool(arch)


def hardware_fit(audit_info: dict, hardware, free_count: int) -> tuple[list, list]:
    return grid.viable_placements(
        audit_info.get("size_bytes", 0), audit_info.get("dims", {}), audit_info.get("quant"), hardware, free_count
    )


def _tuning_spec(model_id: str, local_path: str, point: dict, gpus: list[int], cfg: dict, hardware) -> dict:
    roots = cfg.get("roots") or {}
    docker = cfg.get("docker") or {}
    visible_env = "HIP_VISIBLE_DEVICES" if (hardware and hardware.vendor == "amd") else "CUDA_VISIBLE_DEVICES"
    env = {}
    if gpus and len(gpus) > 1:  # multi-GPU correctness on this box
        env = {"NCCL_PROTO": "Simple", "HIP_FORCE_DEV_KERNARG": "1", "SAFETENSORS_FAST_GPU": "1"}
    return {
        "container_name": TUNING_CONTAINER,
        "image": docker.get("vllm_image"),
        "served_model_name": model_id,
        "model_path": f"/models/{Path(local_path).relative_to(Path(roots['models_dir']).expanduser())}"
        if roots.get("models_dir") and str(local_path).startswith(str(Path(roots["models_dir"]).expanduser()))
        else local_path,
        "models_dir": roots.get("models_dir"),
        "vllm_cache": roots.get("vllm_cache"),
        "port": TUNING_PORT,
        "bind_address": "127.0.0.1",
        "gpus": gpus,
        "visible_env": visible_env,
        "shm_size": docker.get("shm_size", "16g"),
        "knobs": {
            "tensor_parallel_size": point["tp"],
            "max_model_len": point.get("max_model_len"),
            "gpu_memory_util": point.get("gpu_memory_util"),
            "max_num_seqs": point.get("max_num_seqs"),
            "max_num_batched_tokens": point.get("max_num_batched_tokens"),
            "kv_cache_dtype": point.get("kv_cache_dtype", "auto"),
            "mtp": point.get("mtp") or {"enabled": False},
        },
        "extra": {},
        "env": env,
        "labels": {"johnny.tuning": "1", "johnny.model": model_id},
    }


def _cpu_tuning_spec(model_id: str, local_path: str, point: dict, cfg: dict) -> dict:
    roots = cfg.get("roots") or {}
    docker = cfg.get("docker") or {}
    cpu_image = docker.get("cpu_image") or "vllm/vllm-openai-cpu:v0.20.2"
    md = roots.get("models_dir")
    model_path = (
        f"/models/{Path(local_path).relative_to(Path(md).expanduser())}"
        if md and str(local_path).startswith(str(Path(md).expanduser()))
        else local_path
    )
    env = {"VLLM_CPU_KVCACHE_SPACE": "4"}
    if point.get("cpuset"):
        env["VLLM_CPU_OMP_THREADS_BIND"] = point["cpuset"]
    extra = {"device": "cpu", "cpuset": point.get("cpuset")}
    if point.get("embeddings"):
        extra["runner"] = "pooling"
        extra["trust_remote_code"] = True
    return {
        "container_name": TUNING_CONTAINER,
        "image": cpu_image,
        "served_model_name": model_id,
        "model_path": model_path,
        "models_dir": md,
        "vllm_cache": roots.get("vllm_cache"),
        "port": TUNING_PORT,
        "bind_address": "127.0.0.1",
        "gpus": [],
        "visible_env": "CUDA_VISIBLE_DEVICES",
        "shm_size": docker.get("shm_size", "16g"),
        "knobs": {
            "max_model_len": point.get("max_model_len"),
            "max_num_seqs": point.get("max_num_seqs"),
            "max_num_batched_tokens": point.get("max_num_batched_tokens"),
            "kv_cache_dtype": "auto",
            "mtp": {"enabled": False},
        },
        "extra": extra,
        "env": env,
        "labels": {"johnny.tuning": "1", "johnny.model": model_id},
    }


def _bench_embeddings(port: int, model: str) -> dict:
    """Tiny embeddings throughput probe: docs/s under light concurrency + single."""
    import concurrent.futures
    import json as _j
    import time
    import urllib.request

    url = f"http://127.0.0.1:{port}/v1/embeddings"

    def _embed(batch):
        data = _j.dumps({"model": model, "input": batch}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (localhost)
            r.read()

    inputs = [f"sentence number {i} about a variety of technical topics" for i in range(8)]
    try:
        # throughput: 8 concurrent requests, 8 docs each
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda _: _embed(inputs), range(8)))
        peak = (8 * 8) / max(time.time() - t0, 1e-6)
        # single doc
        t1 = time.time()
        _embed(["one short sentence"])
        single = 1.0 / max(time.time() - t1, 1e-6)
        return {"peak_tok_s": round(peak, 1), "single_tok_s": round(single, 1)}
    except Exception as e:
        return {"peak_tok_s": None, "single_tok_s": None, "error": f"embed bench failed: {e}"}


def _parse_bench(out: str) -> dict:
    """peak tok/s = max over the sweep; single ≈ first tok/s in the latency section."""
    nums = [float(x) for x in re.findall(r"([\d.]+)\s*tok/s", out)]
    peak = max(nums) if nums else None
    single = None
    if "Single-request" in out:
        tail = out.split("Single-request", 1)[1]
        m = re.search(r"([\d.]+)\s*tok/s", tail)
        if m:
            single = float(m.group(1))
    return {"peak_tok_s": peak, "single_tok_s": single}


def tune_point(model_id: str, local_path: str, point: dict, gpus: list[int], cfg: dict, hardware) -> dict:
    """Launch a tuning seat for one config point (GPU or CPU), bench it, tear it down.

    Bench selection: embeddings models use the embeddings throughput probe; generative
    models reuse bench.sh (/v1/completions).
    """
    docker = cfg.get("docker") or {}
    is_cpu = point.get("device") == "cpu"
    drv = VllmDriver(image=(docker.get("cpu_image") if is_cpu else docker.get("vllm_image")))
    spec = _cpu_tuning_spec(model_id, local_path, point, cfg) if is_cpu \
        else _tuning_spec(model_id, local_path, point, gpus, cfg, hardware)

    result = {"point": point, "ok": False}
    try:
        drv.launch(spec)
        # CPU loads (and big models) can be slow; give CPU a longer ready window.
        if not _wait_ready(TUNING_PORT, timeout=900 if is_cpu else 600):
            result["error"] = "tuning seat did not become ready in time"
            return result
        if point.get("embeddings"):
            parsed = _bench_embeddings(TUNING_PORT, model_id)
        else:
            bench_script = (cfg.get("scripts") or {}).get("bench")
            if not bench_script or not Path(bench_script).exists():
                result["error"] = "no bench script configured (scripts.bench)"
                return result
            from ..util import run

            rc, out, errout = run(["bash", bench_script, str(TUNING_PORT), model_id], timeout=1200)
            parsed = _parse_bench(out)
            if parsed.get("peak_tok_s") is None:
                parsed["error"] = (errout or out)[-300:]
        result.update(parsed)
        result["ok"] = parsed.get("peak_tok_s") is not None
    finally:
        drv.stop(TUNING_CONTAINER)
    return result


def _wait_ready(port: int, timeout: float) -> bool:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if probe.probe_models(port):
            return True
        time.sleep(3)
    return False


def synthesize(results: list[dict], use_case: str | None) -> dict | None:
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return None
    if use_case == "latency":
        return max(ok, key=lambda r: r.get("single_tok_s") or r.get("peak_tok_s") or 0)
    if use_case == "context":
        return max(ok, key=lambda r: (r["point"].get("max_model_len") or 0))
    return max(ok, key=lambda r: r.get("peak_tok_s") or 0)  # throughput / default
