"""llama.cpp driver (#3) — docker-based `llama-server`, Linux only.

Serves GGUF models via an OpenAI-compatible `llama-server` in docker. Unlike vLLM
(tensor-parallel, whole-model-on-GPU), llama.cpp offloads *layers* to the GPU
(`-ngl`) and can push MoE experts to CPU RAM (`--n-cpu-moe` / `-ot`), so it can run
models that don't fit VRAM. Same docker/GPU mechanics as vLLM (`/dev/kfd`, `/dev/dri`,
`{visible_env}` pinning); readiness/metrics via the OpenAI `/v1/models` + `/metrics`
endpoints. Docker backend => goes through the full GPU-assignment/port path in the
engine (NOT the LM Studio bypass).

The image is expected to ship `llama-server` on PATH as its ENTRYPOINT (a
self-contained build), so `compose()` appends only server args after the image name.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

from ..runtime import probe
from ..util import run, which
from .base import Capabilities, Driver, ModelInfo, SeatInfo, gpu_group_args

# llama-server always listens on this port inside the container; we publish host:port -> here.
_CONTAINER_PORT = 8080


class LlamaCppDriver(Driver):
    name = "llamacpp"

    def __init__(self, models_dir: str | None = None, image: str | None = None):
        self.models_dir = Path(models_dir).expanduser() if models_dir else None
        self.image = image

    def available(self) -> bool:
        if not which("docker"):
            return False
        return probe.docker_available()  # memoized: resolve hits this 2x per call

    def capabilities(self) -> Capabilities:
        return Capabilities(
            kind="docker",
            tunable_knobs=True,      # -ngl / --ctx-size / --n-cpu-moe / -ot sweepable
            per_gpu_placement=True,  # we pin GPUs via {visible_env}
            metrics=True,            # Prometheus /metrics when launched with --metrics
            logs=True,
            structured_output=True,  # --jinja + GBNF grammars
            jit_native=False,
            ttl_native=False,
        )

    # ---- discovery ----------------------------------------------------------
    def list_local(self) -> list[ModelInfo]:
        """Scan models_dir for GGUF files. Sharded models collapse to shard 00001."""
        out: list[ModelInfo] = []
        if not self.models_dir or not self.models_dir.exists():
            return out
        seen: set[str] = set()
        for gguf in sorted(self.models_dir.rglob("*.gguf")):
            name = gguf.name
            # collapse shards: only surface 00001-of-N (or non-sharded files)
            if "-of-" in name and "00001-of-" not in name:
                continue
            try:
                rel = gguf.relative_to(self.models_dir)
            except ValueError:
                continue
            key = str(rel)
            if key in seen:
                continue
            seen.add(key)
            out.append(ModelInfo(id=key, path=str(gguf), backend="llamacpp"))
        return out

    # ---- GGUF metadata probe ------------------------------------------------
    def probe_model(self, path: str) -> dict:
        """Read GGUF header KV metadata (arch/context/quant/experts/layers).

        Accepts a .gguf file or a directory (picks shard 00001 / first .gguf).
        Returns {} on any parse trouble so callers fail open.
        """
        p = Path(path)
        if p.is_dir():
            cands = sorted(p.glob("*.gguf"))
            cands = [c for c in cands if "-of-" not in c.name or "00001-of-" in c.name] or cands
            if not cands:
                return {}
            p = cands[0]
        if not p.exists():
            return {}
        try:
            return _gguf_metadata(p)
        except Exception:
            return {}

    # ---- runtime ------------------------------------------------------------
    def runtime_state(self) -> list[SeatInfo]:
        seats: list[SeatInfo] = []
        if not probe.docker_available():
            return seats
        for c in probe.docker_ps():
            labels = _labels(c)
            backend_label = labels.get("johnny.backend")
            image = c.get("Image", "")
            # Claim only our own containers: explicit backend label, or a llama image
            # carrying johnny labels. Never claim vLLM/other-backend seats.
            if backend_label:
                if backend_label != "llamacpp":
                    continue
            elif "llama" not in image.lower() or not any(k.startswith("johnny.") for k in labels):
                continue
            name = c.get("Names", "")
            ports = probe.host_ports(c.get("Ports", ""))
            jgpus = labels.get("johnny.gpus")
            gpus = (
                [int(x) for x in jgpus.split() if x.isdigit()]
                if jgpus
                else _gpus_from_inspect(name)
            )
            model, port, state = None, None, "running"
            for p in ports:
                m = probe.probe_models(p)
                if m:
                    model, port = m, p
                    break
            if port is None and ports:
                port = ports[0]
            if model:
                state = "ready"
            elif backend_label:
                state = "loading"
            seats.append(
                SeatInfo("llamacpp", name, model, port, gpus, state, {"image": image, "labels": labels})
            )
        return seats

    # ---- launch mechanics ---------------------------------------------------
    def compose(self, spec: dict) -> list[str]:
        """Pure: launch spec -> `docker run` argv. Image ENTRYPOINT is llama-server,
        so everything after the image name is a llama-server flag."""
        extra = spec.get("extra") or {}
        knobs = spec.get("knobs") or {}
        gpus = spec.get("gpus") or []

        args = ["docker", "run", "-d", "--name", spec["container_name"]]
        for k, v in (spec.get("labels") or {}).items():
            args += ["--label", f"{k}={v}"]
        if gpus:
            args += ["--device=/dev/kfd", "--device=/dev/dri", *gpu_group_args()]
        args += ["--ipc=host", "--shm-size", spec.get("shm_size", "16g")]
        if spec.get("models_dir"):
            args += ["-v", f"{spec['models_dir']}:/models:ro"]
        if spec.get("nas_dir"):
            # Second, read-only mount for weights that resolved onto roots.nas_dir
            # (engine.launch.build_spec / config.resolve_weights_path) — only attached
            # when this launch's weights actually came from there, so a launch that
            # never references the NAS doesn't pay for (or expose) the extra mount.
            args += ["-v", f"{spec['nas_dir']}:/nas:ro"]
        if spec.get("weights_dir"):
            # Absolute-path weights (outside both roots) — see engine.launch.build_spec.
            args += ["-v", f"{spec['weights_dir']}:/weights:ro"]
        args += ["-p", f"{spec.get('bind_address', '127.0.0.1')}:{spec['port']}:{_CONTAINER_PORT}"]
        if gpus:
            args += ["-e", f"{spec.get('visible_env', 'CUDA_VISIBLE_DEVICES')}={','.join(str(g) for g in gpus)}"]
        for k, v in (spec.get("env") or {}).items():
            if k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
                continue
            args += ["-e", f"{k}={v}"]

        args += [spec["image"]]  # ENTRYPOINT = llama-server

        # model file: engine.launch.build_spec already resolved extra.gguf_file (if
        # any) as an override over the identity path, and resolved which root/mount
        # (/models or /nas) it actually lives under — spec["model_path"] is the final,
        # already-container-relative path. Don't re-derive it here.
        args += ["-m", spec.get("model_path")]
        args += ["--host", "0.0.0.0", "--port", str(_CONTAINER_PORT)]
        args += ["--alias", spec["served_model_name"]]
        # n_gpu_layers: absent -> 999 (historical default); explicit null -> omit the
        # flag entirely so llama-server's --fit can auto-place layers/experts (fit
        # aborts layer adjustment when -ngl is user-set — needed for CPU-MoE models
        # whose even layer split would overfill one GPU, e.g. Kimi-K2.7 1T).
        ngl = knobs.get("n_gpu_layers", 999)
        if ngl is not None:
            args += ["-ngl", str(ngl)]
        if knobs.get("threads"):  # CPU placement: serve at the tuned thread count
            args += ["-t", str(knobs["threads"])]
        # Flash-attn: DeepSeek-V4 head-dim-512 kernel is unstable on RDNA4 -> default off.
        args += ["-fa", str(knobs.get("flash_attn", "off"))]
        if knobs.get("max_model_len"):
            args += ["-c", str(knobs["max_model_len"])]
        if knobs.get("n_cpu_moe"):
            args += ["--n-cpu-moe", str(knobs["n_cpu_moe"])]
        ot = extra.get("override_tensor")
        if ot:
            # mmap stays ON (llama.cpp default): CPU-offloaded expert tensors page
            # in lazily via the OS cache, so the seat is ready after the VRAM upload
            # (~1-2 min) instead of first copying 100+ GB into RSS single-threaded
            # (10+ min on the 222 GB GLM quants), and a restart reuses the warm
            # cache. Induction benches via llama-bench with mmap on, so serving now
            # matches the measured config. First tokens after a cold start fault
            # experts in from disk — brief warm-up, not a regression. Opt out
            # per-placement with extra_flags: ["--no-mmap"].
            args += ["--override-tensor", ot]
        if knobs.get("parallel"):
            args += ["--parallel", str(knobs["parallel"])]
        args += ["--metrics"]  # expose Prometheus /metrics
        if extra.get("jinja", True):
            args += ["--jinja"]
        args += list(extra.get("extra_flags") or [])
        return args

    def launch(self, spec: dict) -> SeatInfo:
        run(["docker", "rm", "-f", spec["container_name"]], timeout=20)  # idempotent
        argv = self.compose(spec)
        rc, _out, errout = run(argv, timeout=120)
        if rc != 0:
            raise RuntimeError(f"docker run failed: {errout.strip() or 'unknown error'}")
        return SeatInfo(
            "llamacpp",
            spec["container_name"],
            spec["served_model_name"],
            spec.get("port"),
            spec.get("gpus") or [],
            "loading",
            {"image": spec.get("image"), "labels": spec.get("labels", {})},
        )

    def stop(self, seat: str) -> None:
        run(["docker", "rm", "-f", seat], timeout=30)

    def metrics(self, seat: str) -> dict:
        from ..telemetry import sources

        for s in self.runtime_state():
            if s.name == seat and s.port:
                m = sources.metrics_for_port(s.port)
                m["seat"] = seat
                return m
        return {"seat": seat, "source": "unavailable"}

    def logs(self, seat: str, follow: bool = False, tail: int = 200):
        cmd = ["docker", "logs"]
        if follow:
            cmd.append("-f")
        cmd += ["--tail", str(tail), seat]
        if follow:
            import subprocess

            return subprocess.call(cmd)
        rc, out, errout = run(cmd, timeout=15)
        return out + errout


# --- module helpers ----------------------------------------------------------
def _labels(c: dict) -> dict:
    labels: dict = {}
    for kv in (c.get("Labels", "") or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            labels[k.strip()] = v.strip()
    return labels


def _gpus_from_inspect(name: str) -> list[int]:
    rc, out, _ = run(
        ["docker", "inspect", "-f", "{{range .Config.Env}}{{println .}}{{end}}", name], timeout=8
    )
    if rc != 0:
        return []
    for line in out.splitlines():
        for key in ("HIP_VISIBLE_DEVICES=", "CUDA_VISIBLE_DEVICES="):
            if line.startswith(key):
                v = line.split("=", 1)[1].strip()
                return [int(x) for x in v.split(",") if x.strip().isdigit()]
    return []


# GGUF value type ids -> struct reader (see ggml gguf spec).
def _gguf_parse(path: Path, want_tensors: bool = False, want_types: bool = False):
    """(version, header KV, tensors). Tensor sizes come from offset deltas within
    the data section (infos are offset-ordered per spec), so no ggml quant-type
    block-size table is needed; the ≤alignment padding per tensor is noise at model
    scale. `tensors` is [(name, bytes)] by default, or with want_types=True (implies
    want_tensors) [(name, bytes, elem_count, ggml_type_id)] — the per-tensor dtype,
    read from the info block instead of skipped."""
    want_tensors = want_tensors or want_types
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return 0, {}, []
        (ver,) = struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))

        def rstr() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rval(t: int):
            if t == 0:
                return struct.unpack("<b", f.read(1))[0]
            if t == 1:
                return struct.unpack("<B", f.read(1))[0]
            if t == 2:
                return struct.unpack("<h", f.read(2))[0]
            if t == 3:
                return struct.unpack("<H", f.read(2))[0]
            if t == 4:
                return struct.unpack("<i", f.read(4))[0]
            if t == 5:
                return struct.unpack("<I", f.read(4))[0]
            if t == 6:
                return struct.unpack("<f", f.read(4))[0]
            if t == 7:
                return struct.unpack("<?", f.read(1))[0]
            if t == 8:
                return rstr()
            if t == 10:
                return struct.unpack("<q", f.read(8))[0]
            if t == 11:
                return struct.unpack("<Q", f.read(8))[0]
            if t == 12:
                return struct.unpack("<d", f.read(8))[0]
            if t == 9:  # array
                (et,) = struct.unpack("<I", f.read(4))
                (ln,) = struct.unpack("<Q", f.read(8))
                return [rval(et) for _ in range(ln)]
            raise ValueError(f"unknown gguf value type {t}")

        kv: dict = {}
        for _ in range(n_kv):
            k = rstr()
            (t,) = struct.unpack("<I", f.read(4))
            v = rval(t)
            # skip huge token/merge arrays
            if isinstance(v, list) and len(v) > 64:
                v = f"<array len={len(v)}>"
            kv[k] = v

        tensors: list[tuple] = []
        if want_tensors:
            infos = []
            for _ in range(n_tensors):
                name = rstr()
                (nd,) = struct.unpack("<I", f.read(4))
                if want_types:
                    dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
                    (gtype,) = struct.unpack("<I", f.read(4))
                else:
                    f.read(8 * nd + 4)  # dims + ggml type — offsets carry the sizes
                (off,) = struct.unpack("<Q", f.read(8))
                if want_types:
                    nelem = 1
                    for d in dims:
                        nelem *= d
                    infos.append((name, off, nelem, gtype))
                else:
                    infos.append((name, off))
            align = kv.get("general.alignment") or 32
            data_start = f.tell()
            data_start += (-data_start) % align
            data_len = path.stat().st_size - data_start
            infos.sort(key=lambda x: x[1])
            for i, info in enumerate(infos):
                off = info[1]
                nxt = infos[i + 1][1] if i + 1 < len(infos) else data_len
                nbytes = max(0, nxt - off)
                tensors.append((info[0], nbytes, *info[2:]) if want_types else (info[0], nbytes))
    return ver, kv, tensors


def gguf_tensors(path: Path) -> list[tuple[str, int]]:
    """[(tensor name, ~bytes)] for one GGUF file; [] on any parse trouble."""
    try:
        return _gguf_parse(path, want_tensors=True)[2]
    except Exception:
        return []


def gguf_tensor_types(path: Path) -> list[tuple[str, int, int, int]]:
    """[(tensor name, ~bytes, elem count, ggml_type id)] for one GGUF file; [] on
    any parse trouble. The ggml_type id is per-tensor on-disk dtype (see
    _GGML_TENSOR_TYPE_MAP) — distinct from general.file_type below, which is a
    single whole-file summary and unreliable for a custom per-tensor quant mix."""
    try:
        return _gguf_parse(path, want_types=True)[2]
    except Exception:
        return []


_SHARD_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d+)-of-(?P<tot>\d+)\.gguf$")


def gguf_shard_paths(path: Path) -> list[Path]:
    """This GGUF's sibling shards (…-NNNNN-of-MMMMM.gguf), sorted; just [path] when
    it isn't part of a split. Single source for the shard-glob convention every
    multi-file GGUF reader (weight_split, quant_mix, size probes) needs."""
    p = Path(path)
    m = _SHARD_RE.match(p.name)
    if not m:
        return [p]
    shards = sorted(p.parent.glob(f"{m.group('base')}-*-of-{m.group('tot')}.gguf"))
    return shards or [p]


# general.file_type enum → quant name, verified against llama.h `enum llama_ftype`
# (ids 4-6 and 33-35 are removed/historical). "MOSTLY_" caveat applies: for custom
# per-tensor mixes the stamp describes the majority type at best (Ornith Featherweight,
# a 2.41 bpw mix, is stamped 7/Q8_0) — informational only, never a trusted identity fill.
_GGUF_FILE_TYPE_MAP = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0",
}

# Per-tensor ggml_type enum → name, verified against ggml.h `enum ggml_type`. A
# DIFFERENT numbering from _GGUF_FILE_TYPE_MAP above (that's general.file_type, a
# single whole-file enum) — this is the actual on-disk dtype of one tensor block,
# ground truth for a custom per-tensor quant mix that file_type can't represent.
_GGML_TENSOR_TYPE_MAP = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0",
}


def quant_mix(paths: Path | list[Path]) -> dict[str, dict[str, int]]:
    """{type_name: {"elems": int, "bytes": int}} across one or more GGUF files
    (shards of the same model — pass gguf_shard_paths(path) for a split). Unknown
    type ids show as '?<id>'. {} when nothing parses."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    mix: dict[str, dict[str, int]] = {}
    for p in paths:
        for _name, nbytes, nelem, gtype in gguf_tensor_types(Path(p)):
            d = mix.setdefault(_GGML_TENSOR_TYPE_MAP.get(gtype, f"?{gtype}"), {"elems": 0, "bytes": 0})
            d["elems"] += nelem
            d["bytes"] += nbytes
    return mix


def quant_mix_label(mix: dict[str, dict[str, int]], min_share: float = 0.10) -> str | None:
    """Compact quant label from a quant_mix() result: types covering >=min_share of
    total elements, joined '/' by descending share (e.g. Ornith Featherweight's
    IQ2_XXS 65%/Q2_K 32.5%/Q8_0 2.4% attention → 'IQ2_XXS/Q2_K' at the 10% default —
    the attention/norm sliver is real but not what the label is for). A single
    dominant type collapses to itself, matching a uniformly-quantized file's existing
    plain label (e.g. 'Q4_0'). None for an empty mix."""
    total = sum(d["elems"] for d in mix.values())
    if not total:
        return None
    sig = sorted((t for t, d in mix.items() if d["elems"] / total >= min_share),
                 key=lambda t: -mix[t]["elems"])
    return "/".join(sig) or None


def _fmt_param_count(n: int) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 10e9:
        return f"{n / 1e9:.0f}B"
    if n >= 999.5e6:  # a "1B" model is ~0.9999e9 — don't render it as '1000M'
        return f"{n / 1e9:.1f}B"
    return f"{n / 1e6:.0f}M"


def params_total_label(mix: dict[str, dict[str, int]]) -> str | None:
    """Human-readable TOTAL parameter count from a quant_mix() result — the sum of
    tensor element counts across all shards (ground truth, unlike size_label's
    MoE shorthand: Kimi-K2.7's '384x14B' header is a ~1T model). '1.03T', '397B',
    '3.8B'; None for an empty mix."""
    total = sum(d["elems"] for d in mix.values())
    return _fmt_param_count(total) if total else None


def params_active_label(paths: Path | list[Path], n_expert: int | None,
                        n_expert_used: int | None) -> str | None:
    """Human-readable ACTIVE-per-token parameter count: non-expert tensors (backbone,
    attention, shared experts) count in full; routed-expert tensors ('*_exps.*')
    count at the router's used/total ratio. None for dense models (no expert counts,
    or all experts used) — a dense model's active IS its total, no suffix needed."""
    if not n_expert or not n_expert_used or n_expert_used >= n_expert:
        return None
    if isinstance(paths, (str, Path)):
        paths = [paths]
    total = exps = 0
    for p in paths:
        for name, _nbytes, nelem, _gtype in gguf_tensor_types(Path(p)):
            total += nelem
            if "_exps." in name:
                exps += nelem
    if not total or not exps:
        return None
    active = (total - exps) + int(exps * n_expert_used / n_expert)
    return _fmt_param_count(active)


def _gguf_metadata(path: Path) -> dict:
    """Minimal GGUF header reader: returns arch/context/quant/experts/layers."""
    ver, kv, _ = _gguf_parse(path)
    if not kv:
        return {}
    arch = kv.get("general.architecture")
    def g(suffix, default=None):
        return kv.get(f"{arch}.{suffix}", default) if arch else default

    # quant: only a string file_type is authoritative enough to persist as identity.
    # The int enum decodes to file_type_name — display-only ("mostly X", wrong for
    # custom mixes) — so identity fills (registry normalize/induct) never see it.
    file_type_raw = kv.get("general.file_type")
    quant_str = file_type_raw if isinstance(file_type_raw, str) else None
    ftype_name = _GGUF_FILE_TYPE_MAP.get(file_type_raw) if isinstance(file_type_raw, int) else None

    info: dict = {
        "arch": arch,
        "gguf_version": ver,
        "native_context": g("context_length"),
        "n_layer": g("block_count"),
        "n_expert": g("expert_count"),
        "n_expert_used": g("expert_used_count"),
        "quant": quant_str,
        "file_type_id": file_type_raw,
        "file_type_name": ftype_name,
        "mtp_head": bool(g("nextn_predict_layers") or 0),
        "multimodal": False,
        "size_label": kv.get("general.size_label"),
        "name": kv.get("general.name"),
    }
    return info
