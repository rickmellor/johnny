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
    """Resolve a ref (local path, 'vendor/name' under models_dir, or a registry id) to
    (model_id, local_path). HF download is P5; here the model must already be on disk.

    A registry id resolves via its identity.local_path AND keeps the id as the model_id,
    so re-tuning writes perf back to the same registry entry (mirrors llamacpp.gguf_ref,
    which already did this for the GGUF path).
    """
    models_dir = (cfg.get("roots") or {}).get("models_dir")

    p = Path(model_ref).expanduser()
    if p.exists() and (p / "config.json").exists():
        p = p.resolve()  # absolute: the container path→/models mount translation needs it
        return p.name, str(p)
    if models_dir:
        full = (Path(models_dir).expanduser() / model_ref).resolve()
        if (full / "config.json").exists():
            return full.name, str(full)

    # Registry id -> its identity.local_path; keep the id so induction updates *this* entry.
    from ..registry import store

    m = (store.load().get("models") or {}).get(model_ref)
    if m:
        ident = m.get("identity") or {}
        lp = ident.get("local_path") or ident.get("repo_id")
        if lp:
            cand = Path(lp).expanduser()
            if not cand.is_absolute() and models_dir:
                cand = Path(models_dir).expanduser() / lp
            cand = cand.resolve()
            if (cand / "config.json").exists():
                return model_ref, str(cand)
            raise FileNotFoundError(
                f"registry model '{model_ref}' → '{lp}' has no weights at {cand} "
                f"(download/acquire arrives at P5; place them there or pass a local path)"
            )

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


# Heads that emit tokens (autoregressive *and* diffusion LMs) — never embeddings.
_GENERATIVE_ARCH = (
    "ForCausalLM", "ForConditionalGeneration", "LMHeadModel", "ForSeq2SeqLM",
    "ForBlockDiffusion", "ForDiffusionLM", "ForDiffusion",  # diffusion LMs generate text
)
# Pooling/encoder arch names with no generative head (e.g. NomicBertModel).
_EMBEDDING_ARCH = ("Model", "Encoder", "ForSequenceClassification", "ForMaskedLM")


def is_embeddings(audit_info: dict) -> bool:
    """Heuristic: an encoder/bert-like arch with no generative head is a pooling/
    embeddings model (e.g. NomicBertModel). Generative heads — including diffusion
    LMs and multimodal generators — are not. Override with --embeddings/--no-embeddings.

    Fails *closed* to generative: an unrecognized arch is treated as generative (it
    benches against /v1/completions) rather than embeddings (which would force the
    /v1/embeddings path a generative model doesn't serve)."""
    arch = audit_info.get("arch") or ""
    if not arch:
        return False
    if any(g in arch for g in _GENERATIVE_ARCH):
        return False
    # A vision tower over a generative stack ⇒ a multimodal generator, not a pooler.
    if audit_info.get("multimodal"):
        return False
    # Positive signal only: bare encoder/pooling/classifier arch names.
    return any(arch.endswith(e) for e in _EMBEDDING_ARCH)


def arch_supported(arch: str | None, image: str | None) -> tuple[bool, str | None]:
    """Pre-flight: does `image`'s vLLM register `arch`? (Fix for the 'unsupported
    model → 8 doomed seats → mystery timeout' failure mode.)

    Returns (ok, reason_if_not). Fails *open* — when support can't be determined
    (docker missing, query failed), returns ok=True so a transient probe failure
    never blocks a legitimate induct."""
    if not arch or not image:
        return True, None
    archs = VllmDriver(image=image).supported_archs(image)
    if archs is None:  # couldn't determine — don't block
        return True, None
    if arch in archs:
        return True, None
    return False, (
        f"architecture {arch!r} is not registered by vLLM image {image!r} "
        f"({len(archs)} archs available; none match) — this model needs a newer "
        f"or forked vLLM build, so no tuning config could launch"
    )


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


def _parse_kv_cache(log: str) -> dict:
    """Actual KV capacity vLLM computed at load — the real context a gmu buys.

    Parses the engine startup log: 'GPU KV cache size: N tokens' (total KV token
    pool) + 'Maximum concurrency for M tokens per request: Xx' + 'Available KV cache
    memory: G GiB'. The requested mml is only a ceiling; this is what actually fits.
    """
    out: dict = {"kv_cache_tokens": None, "max_concurrency": None, "kv_cache_gib": None}
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log)
    if m:
        out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x", log)
    if m:
        out["max_concurrency"] = float(m.group(1))
    m = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", log)
    if m:
        out["kv_cache_gib"] = float(m.group(1))
    return out


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
        ready, why = _wait_ready(drv, TUNING_CONTAINER, TUNING_PORT, timeout=900 if is_cpu else 600)
        if not ready:
            # Surface the real reason (dead container / vLLM error), not a bare timeout.
            logtail = _diagnose(drv, TUNING_CONTAINER)
            result["error"] = (why or "tuning seat did not become ready in time") + (
                f" — {logtail}" if logtail else ""
            )
            return result
        # Actual KV capacity vLLM sized for this gmu — the real context/concurrency the
        # gmu buys (the requested mml is only a ceiling, identical across gmu points).
        result.update(_parse_kv_cache(drv.logs(TUNING_CONTAINER, tail=500) or ""))
        if point.get("embeddings"):
            parsed = _bench_embeddings(TUNING_PORT, model_id)
        else:
            from ..bundled import resolve_script

            bench_script = resolve_script("bench", cfg)  # bundled by default; config-overridable
            if not bench_script:
                result["error"] = "bench script unavailable (not bundled and no scripts.bench override)"
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


# Known vLLM startup-failure signatures → a short, human-readable cause.
_FAILURE_SIGNATURES = (
    ("are not supported for now", "unsupported architecture for this vLLM image"),
    ("not supported", "unsupported model/feature for this vLLM image"),
    ("trust_remote_code", "needs --trust-remote-code / custom modeling code"),
    ("out of memory", "GPU OOM during load"),
    ("CUDA out of memory", "GPU OOM during load"),
    ("HIP out of memory", "GPU OOM during load"),
    ("No such file or directory", "model path/file missing in container"),
    ("Address already in use", "tuning port already in use"),
    ("NCCL", "NCCL/multi-GPU init failure"),
)


def _container_exited(drv, container: str) -> tuple[bool, int | None]:
    """(exited?, exit_code) from docker inspect; (False, None) if state unknown."""
    from ..util import run

    rc, out, _ = run(
        ["docker", "inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", container], timeout=8
    )
    if rc != 0:
        return False, None
    parts = out.strip().split()
    if len(parts) != 2:
        return False, None
    running = parts[0] == "true"
    try:
        code = int(parts[1])
    except ValueError:
        code = None
    return (not running), code


# vLLM logs the real exception, then a generic summary; don't report the summary.
_GENERIC_ERR = (
    "Engine core initialization failed",
    "See root cause above",
    "Failed core proc",
    "EngineCore failed to start",
    "Engine core proc",
)


def _root_cause(log: str) -> str | None:
    """The most specific `*Error:/Exception:` line that isn't vLLM's generic
    'Engine core initialization failed' summary — the actual root cause."""
    cands = []
    for ln in log.splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.search(r"\w*(?:Error|Exception)\s*:", s) and not any(g in s for g in _GENERIC_ERR):
            cands.append(s)
    return cands[-1][:240] if cands else None


def _diagnose(drv, container: str) -> str | None:
    """Tail the container log and surface the real cause: specific exception line
    first (above vLLM's generic summary), then known signatures, then last line."""
    try:
        log = drv.logs(container, tail=200) or ""
    except Exception:
        return None
    rc = _root_cause(log)
    if rc:
        return rc
    for needle, msg in _FAILURE_SIGNATURES:
        if needle in log:
            return msg
    lines = [ln.strip() for ln in log.splitlines() if ln.strip()]
    return lines[-1][:200] if lines else None


def _wait_ready(drv, container: str, port: int, timeout: float) -> tuple[bool, str | None]:
    """Poll the endpoint until ready, OR bail early if the container dies.

    Returns (ready, reason_if_not). Watching container liveness is the fix for
    polling a crashed seat for the full timeout window."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if probe.probe_models(port):
            return True, None
        exited, code = _container_exited(drv, container)
        if exited:
            return False, f"tuning container exited (code {code}) before serving"
        time.sleep(3)
    return False, "tuning seat did not become ready in time"


def synthesize(results: list[dict], use_case: str | None) -> dict | None:
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return None
    if use_case == "latency":
        return max(ok, key=lambda r: r.get("single_tok_s") or r.get("peak_tok_s") or 0)
    if use_case == "context":
        return max(ok, key=lambda r: (r["point"].get("max_model_len") or 0))
    return max(ok, key=lambda r: r.get("peak_tok_s") or 0)  # throughput / default
