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
from ..backends.base import gpu_group_args
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
def _gguf_stem(p: Path) -> str:
    """Variant stem of a GGUF file: the filename minus any -NNNNN-of-NNNNN split."""
    m = _SHARD_RE.match(p.name)
    return m.group("base") if m else p.stem


def _dir_ref(d: Path) -> tuple[str, str] | None:
    """Resolve a directory ref (repo root or quant subdir) to the GGUF inside.
    Exactly one variant → (stem, first-shard path). Several (`download` fetches
    one, but a repo accreted by hand can hold many) → interactive picker, or an
    error naming them when there's no TTY to ask. No GGUFs → None, so non-GGUF
    dirs fall through to the vLLM path."""
    if not d.is_dir():
        return None
    variants: dict[str, Path] = {}
    sizes: dict[str, int] = {}
    for f in sorted(d.rglob("*.gguf")):
        stem = _gguf_stem(f)
        variants.setdefault(stem, f)  # sorted → shard -00001 wins
        sizes[stem] = sizes.get(stem, 0) + f.stat().st_size
    if not variants:
        return None
    if len(variants) > 1:
        from ..external import picker

        order = sorted(variants, key=lambda s: sizes[s])
        idx = picker.select(
            order, lambda s: f"{s}  [dim]{sizes[s] / 1e9:.1f} GB[/]",
            title=f"{len(order)} GGUF variants in {d.name} — induct which?",
        ) if picker._interactive_capable() else None
        if idx is None:
            raise FileNotFoundError(
                f"{len(variants)} GGUF variants under {d}: {', '.join(order)} "
                "— pass the variant's subdirectory or a .gguf path")
        stem = order[idx]
        return stem, str(variants[stem].resolve())
    stem, f = variants.popitem()
    return stem, str(f.resolve())


def gguf_ref(model_ref: str, cfg: dict) -> tuple[str, str] | None:
    """(model_id, gguf_path) if model_ref points at a GGUF, else None.

    Accepts: a direct .gguf path, a .gguf under models_dir, a directory (absolute
    or under models_dir — e.g. the 'vendor/repo' a `johnny download` produced,
    where the GGUF sits in a quant subdir), or a registry model id whose
    identity.local_path is a .gguf / that has a llamacpp placement.
    """
    md = (cfg.get("roots") or {}).get("models_dir")

    p = Path(model_ref).expanduser()
    if p.suffix == ".gguf" and p.exists():
        return _gguf_stem(p), str(p.resolve())
    hit = _dir_ref(p)
    if hit:
        return hit
    if md:
        cand = Path(md).expanduser() / model_ref
        if cand.suffix == ".gguf" and cand.exists():
            return _gguf_stem(cand), str(cand.resolve())
        hit = _dir_ref(cand)
        if hit:
            return hit

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


def audit(path: str) -> dict:
    """GGUF audit via the llamacpp driver's header probe + shard-aware size and
    quant mix. `quant` is overridden with the tensor-mix label (ground truth across
    every shard) whenever one is derivable — the header's own probe_model() quant
    is a single whole-file file_type stamp, unreliable for a custom per-tensor mix
    (see backends.llamacpp.quant_mix_label)."""
    from ..backends.llamacpp import gguf_shard_paths, quant_mix, quant_mix_label

    info = get_driver("llamacpp").probe_model(path)
    shards = gguf_shard_paths(Path(path))
    info["size_bytes"] = sum(s.stat().st_size for s in shards if s.exists())
    mix = quant_mix(shards)
    if mix:
        info["quant_mix"] = mix
        label = quant_mix_label(mix)
        if label:
            info["quant"] = label
    return info


# --- candidate grid ----------------------------------------------------------
def _gpu_count_for(size_bytes: int, total_gpu_count: int, per_gpu_gb: float = 28.0) -> int:
    """How many GPUs the weights need (fit at ~per_gpu_gb usable), capped to the box's
    total GPU count. Independent of what's currently free (that gates the *tune*, below)."""
    size_gb = size_bytes / 1e9
    need = max(1, -(-int(size_gb) // int(per_gpu_gb)))  # ceil
    return max(1, min(need, total_gpu_count or 1))


# VRAM heuristics (per GPU): fit decision vs. weight-target when offloading. The gap
# (~5 GB) is reserved for KV cache + compute buffers so an offload point doesn't OOM.
# 27 GB weight-target is what the validated balanced -ot bf16 run actually used/GPU.
_FIT_GB_PER_GPU = 30.0
_WEIGHT_TARGET_GB_PER_GPU = 27.0


def _offload_regex(n_layer: int, gpu_count: int, k_per_group: int) -> str:
    """Balanced expert-offload: push the first k experts of each of the gpu_count
    layer-groups to CPU. Spreading across groups keeps GPU load even (a contiguous
    --n-cpu-moe empties GPU0 and OOMs the last GPU). per_group uses CEIL to match
    llama.cpp's actual layer->GPU split (~ceil(n_layer/gpu_count) per device incl. the
    output layer); floor skews offload toward the low GPUs and OOMs the high ones."""
    import math
    per_group = max(1, math.ceil(n_layer / gpu_count))
    layers: list[int] = []
    for g in range(gpu_count):
        base = g * per_group
        for i in range(k_per_group):
            L = base + i
            if L < n_layer:
                layers.append(L)
    alt = "|".join(str(L) for L in sorted(set(layers)))
    return rf"blk\.({alt})\.ffn_(gate|up|down)_exps\.weight=CPU"


def candidate_points(audit_info: dict, gpu_count: int, max_points: int | None,
                     mml_override: int | None = None) -> list[dict]:
    """llama-server sweep. If the weights fit all-GPU, sweep the throughput knob
    (`--parallel`). If not (e.g. bf16 > VRAM), sweep two balanced expert-offload
    levels (`-ot` → experts on CPU RAM), since that's the only way it runs."""
    native = audit_info.get("native_context") or 32768
    ctx = mml_override or min(32768, native)
    size_gb = (audit_info.get("size_bytes") or 0) / 1e9
    n_layer = audit_info.get("n_layer") or 0
    # Flash-attn off is a DeepSeek-only workaround (head-dim-512 kernel unstable on
    # RDNA4); other archs measure fine with fa on (qwen35moe validated) and fa is a
    # prerequisite for quantized KV cache, so don't pay the tax globally.
    fa = "off" if "deepseek" in (audit_info.get("arch") or "").lower() else "on"

    def _mk(**kw):
        return {"backend": "llamacpp", "gpu_count": gpu_count, "n_gpu_layers": 999,
                "flash_attn": fa, "max_model_len": ctx, "parallel": 4, **kw}

    if not n_layer or size_gb <= gpu_count * _WEIGHT_TARGET_GB_PER_GPU:
        pts = [_mk(parallel=4), _mk(parallel=16)]  # fits with headroom: sweep concurrency
    else:
        import math
        layer_gb = size_gb / n_layer
        overflow = size_gb - gpu_count * _WEIGHT_TARGET_GB_PER_GPU
        need = max(1, math.ceil(overflow / layer_gb))  # expert-layers to shed
        k = max(1, math.ceil(need / gpu_count))  # per-group, balanced
        pts = []
        # Boundary zone (validated: Ornith 397B, 119.5 GB on 4×32 GB): weights squeeze
        # into raw VRAM but leave <5 GB/GPU for KV+buffers — sweep the no-offload point
        # too and let the measured run decide, instead of trusting the fit line.
        if size_gb <= gpu_count * _FIT_GB_PER_GPU:
            pts.append(_mk(parallel=4))
        # k is an overestimate (layer_gb counts attention/shared weights, but only the
        # expert tensors move), so sweep k-1 as well: lighter offload = faster decode
        # (Ornith measured: k=1 20.1 t/s vs k=2 18.5 t/s).
        for kk in [x for x in (k - 1, k, k + 1) if x >= 1]:
            layers = min(kk * gpu_count, n_layer)
            pts.append(_mk(override_tensor=_offload_regex(n_layer, gpu_count, kk),
                           offload_layers=layers))
    if max_points:
        pts = pts[:max_points]
    return pts


def cpu_candidate_points(audit_info: dict, ncpu: int, max_points: int | None,
                         mml_override: int | None = None) -> list[dict]:
    """CPU-only sweep (`-ngl 0`): everything runs on CPU, so the knob that matters is the
    thread count. Sweep logical vs. physical-ish (ncpu, ncpu/2) — llama.cpp is often fastest
    at physical cores, and CPU decode saturates on memory bandwidth well before all threads."""
    native = audit_info.get("native_context") or 32768
    ctx = mml_override or min(32768, native)
    threads = list(dict.fromkeys([ncpu, max(1, ncpu // 2)]))
    pts = [{"backend": "llamacpp", "device": "cpu", "n_gpu_layers": 0, "flash_attn": "off",
            "max_model_len": ctx, "threads": t, "gpu_count": 0, "parallel": 1} for t in threads]
    return pts[:max_points] if max_points else pts


def _point_sig(p: dict) -> str:
    if p.get("device") == "cpu":
        return f"llama-cpu-t{p.get('threads')}-mml{p.get('max_model_len')}"
    base = f"llama-ngl{p.get('n_gpu_layers')}-par{p.get('parallel')}-mml{p.get('max_model_len')}"
    if p.get("override_tensor"):
        base += f"-ot{p.get('offload_layers')}"
    return base


# --- tuning (via llama-bench: clean single-stream prefill/decode) -------------
def _parse_llama_bench(out: str) -> tuple[float | None, float | None]:
    """From llama-bench's markdown table, pull prefill (pp*) and decode (tg*) t/s.
    Rows look like: | ... | pp512 | 311.28 ± 5.50 |  and  | ... | tg128 | 15.30 ± 0.77 |"""
    pp = tg = None
    for line in out.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        test = next((c for c in cells if c.startswith(("pp", "tg"))), None)
        if not test:
            continue
        # t/s is the rightmost numeric cell (may be "311.28 ± 5.50")
        nums = [c for c in cells if re.match(r"^[\d.]+(\s*(±|\+/-).*)?$", c)]
        if not nums:
            continue
        num = float(re.split(r"\s*(±|\+/-)\s*", nums[-1])[0])
        if test.startswith("pp"):
            pp = num
        elif test.startswith("tg"):
            tg = num
    return pp, tg


_BENCH_ENTRYPOINT_CACHE: dict[str, tuple[str, list[str], str]] = {}


def _bench_entrypoint(image: str) -> tuple[str, list[str], str]:
    """(entrypoint, prefix_args, fa_off_value) for this image's llama-bench convention.

    Two conventions coexist on this box: older custom builds (`johnny-llamacpp-
    qwen35:gfx1201`, `-dsv4`, etc.) ship a standalone `/opt/llamacpp/bin/llama-bench`
    binary with boolean `-fa <0|1>`. The stock `ghcr.io/ggml-org/llama.cpp:server-
    vulkan` image (`johnny-llamacpp-vulkan:latest`) restructured its CLI into a single
    `/app/llama <subcommand>` entrypoint — no standalone llama-bench — with a
    tri-state `-fa <on|off|auto>`. Hardcoding the old path/flag broke real sweeps for
    any model on the new image. Detected via a throwaway `docker run --entrypoint
    /bin/sh` probe, cached per image for the process lifetime."""
    if image in _BENCH_ENTRYPOINT_CACHE:
        return _BENCH_ENTRYPOINT_CACHE[image]
    from ..util import run

    rc, out, _ = run(
        ["docker", "run", "--rm", "--entrypoint", "/bin/sh", image,
         "-c", "test -x /opt/llamacpp/bin/llama-bench && echo legacy || echo restructured"],
        timeout=60,
    )
    if rc == 0 and "legacy" in (out or ""):
        result = ("/opt/llamacpp/bin/llama-bench", [], "0")
    else:
        result = ("/app/llama", ["bench"], "off")
    _BENCH_ENTRYPOINT_CACHE[image] = result
    return result


def tune_point(model_id: str, gguf_path: str, point: dict, gpus: list[int], cfg: dict, hardware) -> dict:
    """Speed-bench one config via llama-bench (loads the GGUF directly — no server).
    Returns clean single-stream prefill/decode t/s; supports -ncmoe/-ot offload points."""
    from ..util import run

    image = C.resolve_image(cfg, backend="llamacpp", model_id=model_id)
    entrypoint, bench_prefix, fa_off = _bench_entrypoint(image)
    md = (cfg.get("roots") or {}).get("models_dir")
    cpath = (
        f"/models/{Path(gguf_path).relative_to(Path(md).expanduser())}"
        if md and str(gguf_path).startswith(str(Path(md).expanduser()))
        else gguf_path
    )
    if point.get("device") == "cpu":
        # CPU-only: no GPU devices at all (the HIP build detects no ROCm device and falls
        # back to CPU) — so the seat truly runs off-GPU and doesn't need free cards.
        import os

        argv = [
            "docker", "run", "--rm", "--name", TUNING_CONTAINER, "--ipc=host",
            "-v", f"{md}:/models:ro",
            "--entrypoint", entrypoint, image, *bench_prefix,
            "-m", cpath, "-ngl", "0", "-fa", fa_off,
            "-t", str(point.get("threads") or os.cpu_count() or 8),
            "-p", "512", "-n", "128", "-r", "2",
        ]
    else:
        visible_env = "HIP_VISIBLE_DEVICES" if (hardware and hardware.vendor == "amd") else "CUDA_VISIBLE_DEVICES"
        argv = [
            "docker", "run", "--rm", "--name", TUNING_CONTAINER,
            "--device=/dev/kfd", "--device=/dev/dri", *gpu_group_args(),
            "--security-opt", "seccomp=unconfined", "--ipc=host",
            "-v", f"{md}:/models:ro",
            "-e", f"{visible_env}={','.join(str(g) for g in gpus)}",
            "--entrypoint", entrypoint, image, *bench_prefix,
            "-m", cpath, "-ngl", str(point.get("n_gpu_layers", 999)), "-fa", fa_off,
            "-p", "512", "-n", "128", "-r", "2",
        ]
        if point.get("n_cpu_moe"):
            argv += ["-ncmoe", str(point["n_cpu_moe"])]
        if point.get("override_tensor"):
            argv += ["-ot", point["override_tensor"]]

    result = {"point": point, "ok": False}
    run(["docker", "rm", "-f", TUNING_CONTAINER], timeout=20)  # idempotent
    rc, out, errout = run(argv, timeout=1800)
    pp, tg = _parse_llama_bench(out)
    if tg is not None:
        # synthesize ranks by peak_tok_s/single_tok_s; use decode t/s for both.
        result.update({"prefill_tok_s": pp, "decode_tok_s": tg,
                       "peak_tok_s": tg, "single_tok_s": tg, "ok": True})
    else:
        result["error"] = (errout or out)[-300:]
    return result


# --- memory split ------------------------------------------------------------
def _split_tensors(tensors, override_tensor: str | None, n_cpu_moe) -> tuple[int, int]:
    """(vram bytes, ram bytes) over [(name, bytes)] for an offload config: -ot
    'regex=CPU' sends matching tensors to RAM; -ncmoe N sends the first N layers'
    routed-expert FFNs. Everything else lands on GPU (ngl 999 placements)."""
    pat = re.compile(override_tensor.split("=", 1)[0]) if override_tensor else None
    ncmoe = int(n_cpu_moe or 0)
    vram = ram = 0
    for name, nbytes in tensors:
        to_cpu = bool(pat and pat.search(name))
        if not to_cpu and ncmoe:
            m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.", name)
            to_cpu = bool(m and int(m.group(1)) < ncmoe)
        if to_cpu:
            ram += nbytes
        else:
            vram += nbytes
    return vram, ram


def weight_split(gguf_path: str, override_tensor: str | None = None, n_cpu_moe=None) -> dict | None:
    """{'vram_gb', 'ram_gb'} weight placement for an offload config, from the
    GGUF tensor table across all shards. None when the file can't be parsed.
    Weights only — KV cache and compute buffers land on top of the VRAM side."""
    from ..backends.llamacpp import gguf_shard_paths, gguf_tensors

    shards = gguf_shard_paths(Path(gguf_path))
    tensors = [t for s in shards for t in gguf_tensors(s)]
    if not tensors:
        return None
    vram, ram = _split_tensors(tensors, override_tensor, n_cpu_moe)
    return {"vram_gb": round(vram / 1024**3, 1), "ram_gb": round(ram / 1024**3, 1)}


# --- placement ---------------------------------------------------------------
def to_placement(model_id: str, winner: dict, audit_info: dict, hardware, runtime_version: str, use_case: str | None,
                 gguf_path: str | None = None) -> dict:
    p = winner["point"]
    is_cpu = p.get("device") == "cpu"
    mem = weight_split(gguf_path, p.get("override_tensor"), p.get("n_cpu_moe")) if gguf_path else None
    if mem and is_cpu:
        mem = {"vram_gb": 0.0, "ram_gb": round(mem["vram_gb"] + mem["ram_gb"], 1)}
    return {
        "id": f"induct-{_point_sig(p)}",
        **({"mem": mem} if mem else {}),
        "backend": "llamacpp",
        "image": C.resolve_image(load_config(), backend="llamacpp", model_id=model_id),
        "use_case": use_case,
        "knobs": {
            "gpu_count": p.get("gpu_count"),
            "n_gpu_layers": p.get("n_gpu_layers", 999),
            "flash_attn": p.get("flash_attn", "off"),
            "max_model_len": p.get("max_model_len"),
            "n_cpu_moe": p.get("n_cpu_moe"),
            "parallel": p.get("parallel"),
            **({"threads": p.get("threads")} if is_cpu else {}),
        },
        "extra": {"jinja": True, "nickname": audit_info.get("name") or model_id,
                  **({"device": "cpu"} if is_cpu else {}),
                  **({"override_tensor": p["override_tensor"]} if p.get("override_tensor") else {})},
        "env": {},
        # Keep the llama-native prefill/decode AND the standardized peak/single the registry
        # view + status read (decode is the single-stream rate for llama.cpp).
        "perf": {"prefill_tok_s": winner.get("prefill_tok_s"), "decode_tok_s": winner.get("decode_tok_s"),
                 "peak_tok_s": winner.get("peak_tok_s"), "single_stream_tok_s": winner.get("single_tok_s")},
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
        f"- arch: {a.get('arch')}  quant: {a.get('quant') or a.get('file_type_name') or a.get('file_type_id')}  "
        f"size: {a.get('size_bytes', 0) / 1e9:.1f} GB  native_ctx: {a.get('native_context')}", "",
        "## Sweep (llama-bench, single-stream)", "",
        "| ngl | offload | mml | prefill tok/s | decode tok/s | ok |",
        "|-----|---------|-----|--------------|--------------|----|",
    ]
    for r in results:
        p = r["point"]
        off = p.get("offload_layers") or (p.get("n_cpu_moe") and f"ncmoe{p['n_cpu_moe']}") or "—"
        lines.append(
            f"| {p.get('n_gpu_layers')} | {off} | {p.get('max_model_len')} | "
            f"{r.get('prefill_tok_s')} | {r.get('decode_tok_s')} | {'✓' if r.get('ok') else '✗'} |"
        )
    if winner:
        wp = winner["point"]
        lines += ["", f"## Winner\n\nngl={wp.get('n_gpu_layers')} "
                  f"offload={wp.get('offload_layers') or wp.get('n_cpu_moe') or 0} "
                  f"mml={wp.get('max_model_len')} → prefill {winner.get('prefill_tok_s')} tok/s, "
                  f"decode {winner.get('decode_tok_s')} tok/s"]
    path = run_dir / "TUNING_REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# --- entry points ------------------------------------------------------------
def plan(model_id: str, gguf_path: str, max_points: int | None = None, cfg: dict | None = None,
         device: str = "auto", mml_override: int | None = None) -> dict:
    cfg = cfg if cfg is not None else load_config()
    hw = hwd.detect()
    a = audit(gguf_path)
    audit_view = {"arch": a.get("arch"), "quant": a.get("quant") or a.get("file_type_name") or a.get("file_type_id"),
                  "size_gb": round(a.get("size_bytes", 0) / 1e9, 1), "native_ctx": a.get("native_context")}
    if device == "cpu":
        import os

        pts = cpu_candidate_points(a, os.cpu_count() or 8, max_points, mml_override=mml_override)
        return {
            "model_id": model_id, "path": gguf_path, "backend": "llamacpp",
            "audit": audit_view, "free_gpus": free_gpus(hw, all_seats(cfg)),
            "device": "cpu", "gpu_count": 0, "arch_supported": True, "arch_warning": None,
            "viable": [{"device": "cpu", "per_host_gb": audit_view["size_gb"]}],
            "pruned": [], "priors": 0, "points": pts,
        }
    free = free_gpus(hw, all_seats(cfg))
    total_gpus = len(free_gpus(hw, []))  # all GPUs, ignoring current occupancy
    gc = _gpu_count_for(a.get("size_bytes", 0), total_gpus)
    pts = candidate_points(a, gc, max_points, mml_override=mml_override)
    return {
        "model_id": model_id, "path": gguf_path, "backend": "llamacpp",
        "audit": audit_view,
        "free_gpus": free, "device": "gpu", "gpu_count": gc,
        "arch_supported": True, "arch_warning": None,
        "viable": [{"gpu_count": gc, "n_gpu_layers": 999}], "pruned": [], "priors": 0,
        "points": pts,
    }


def run(model_id: str, gguf_path: str, use_case: str | None = None, max_points: int | None = None,
        cfg: dict | None = None, progress=None, device: str = "auto", mml_override: int | None = None,
        select=None) -> dict:
    """select: optional (results, winner) -> results to write placements for (see pipeline.run)."""
    _p = progress or (lambda *_: None)
    cfg = cfg if cfg is not None else load_config()
    hw = hwd.detect()
    a = audit(gguf_path)
    _p(f"audit: {a.get('arch')} · {round(a.get('size_bytes', 0) / 1e9, 1)}GB · "
       f"experts={a.get('n_expert')} · ctx={a.get('native_context')}")

    results = []
    if device == "cpu":
        import os

        points = cpu_candidate_points(a, os.cpu_count() or 8, max_points, mml_override=mml_override)
        _p(f"backend=llamacpp · device=cpu · {len(points)} point(s)")
        for i, point in enumerate(points, 1):
            _p(f"[{i}/{len(points)}] {_point_sig(point)}: benching on CPU (t={point.get('threads')})…")
            collect.add_pin(TUNING_CONTAINER)
            try:
                r = tune_point(model_id, gguf_path, point, [], cfg, hw)  # gpus=[] — CPU-only
            finally:
                collect.remove_pin(TUNING_CONTAINER)
            if r.get("ok"):
                _p(f"[{i}/{len(points)}] {_point_sig(point)}: prefill {r.get('prefill_tok_s')} tok/s, "
                   f"decode {r.get('decode_tok_s')} tok/s")
            else:
                _p(f"[{i}/{len(points)}] {_point_sig(point)}: FAILED — {(r.get('error') or '')[:80]}")
            results.append(r)
    else:
        free = free_gpus(hw, all_seats(cfg))
        total_gpus = len(free_gpus(hw, []))  # all GPUs, ignoring current occupancy
        gc = _gpu_count_for(a.get("size_bytes", 0), total_gpus)
        points = candidate_points(a, gc, max_points, mml_override=mml_override)
        _p(f"backend=llamacpp · gpu_count={gc} · {len(free)} free GPU(s) · {len(points)} point(s)")
        if len(free) < gc:
            return {"model_id": model_id, "error": f"need {gc} free GPUs, {len(free)} free "
                    f"(down a seat first)", "backend": "llamacpp"}
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
                _p(f"[{i}/{len(points)}] {_point_sig(point)}: prefill {r.get('prefill_tok_s')} tok/s, "
                   f"decode {r.get('decode_tok_s')} tok/s")
            else:
                _p(f"[{i}/{len(points)}] {_point_sig(point)}: FAILED — {(r.get('error') or '')[:80]}")
            results.append(r)

    winner = stages.synthesize(results, use_case)
    run_dir = C.get_paths().runs_dir / f"induct-{model_id.replace('/', '__')}"
    report_path = _write_report(run_dir, model_id, a, results, winner)

    placements, winner_pid = [], None
    if winner:
        rtv = str((cfg.get("docker") or {}).get("llamacpp_image", "")).split(":")[-1] or "unknown"
        chosen = (select(results, winner) if select else None) or [winner]
        md = (cfg.get("roots") or {}).get("models_dir")
        try:
            lp = str(Path(gguf_path).relative_to(Path(md).expanduser())) if md else None
        except (ValueError, TypeError):
            lp = None
        for r in chosen:
            # Only the winner carries the use-case tag (see pipeline.run).
            placement = to_placement(model_id, r, a, hw, rtv, use_case if r is winner else None,
                                     gguf_path=gguf_path)
            report.write_placement(model_id, a, placement, hw, local_path=lp)
            placements.append(placement)
            if r is winner:
                winner_pid = placement["id"]

    return {
        "model_id": model_id, "backend": "llamacpp", "points": len(points),
        "results": results, "winner": winner,
        "placement_id": winner_pid or (placements[0]["id"] if placements else None),
        "placement_ids": [p["id"] for p in placements],
        "report": str(report_path),
        "bench": "throughput sweep complete (bench.sh)",
    }
