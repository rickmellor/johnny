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
import struct
from pathlib import Path

from ..runtime import probe
from ..util import run, which
from .base import Capabilities, Driver, ModelInfo, SeatInfo

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
        rc, _, _ = run(["docker", "info"], timeout=6)
        return rc == 0

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
            args += ["--device=/dev/kfd", "--device=/dev/dri", "--group-add=video", "--group-add=render"]
        args += ["--ipc=host", "--shm-size", spec.get("shm_size", "16g")]
        if spec.get("models_dir"):
            args += ["-v", f"{spec['models_dir']}:/models:ro"]
        args += ["-p", f"{spec.get('bind_address', '127.0.0.1')}:{spec['port']}:{_CONTAINER_PORT}"]
        if gpus:
            args += ["-e", f"{spec.get('visible_env', 'CUDA_VISIBLE_DEVICES')}={','.join(str(g) for g in gpus)}"]
        for k, v in (spec.get("env") or {}).items():
            if k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
                continue
            args += ["-e", f"{k}={v}"]

        args += [spec["image"]]  # ENTRYPOINT = llama-server

        # model file: extra.gguf_file (relative to /models) overrides the identity path.
        gguf = extra.get("gguf_file")
        model_path = f"/models/{gguf}" if gguf else spec.get("model_path")
        args += ["-m", model_path]
        args += ["--host", "0.0.0.0", "--port", str(_CONTAINER_PORT)]
        args += ["--alias", spec["served_model_name"]]
        args += ["-ngl", str(knobs.get("n_gpu_layers", 999))]
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
            args += ["--override-tensor", ot, "--no-mmap"]
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
def _gguf_metadata(path: Path) -> dict:
    """Minimal GGUF header reader: returns arch/context/quant/experts/layers."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return {}
        (ver,) = struct.unpack("<I", f.read(4))
        struct.unpack("<Q", f.read(8))  # n_tensors
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

    arch = kv.get("general.architecture")
    def g(suffix, default=None):
        return kv.get(f"{arch}.{suffix}", default) if arch else default

    info: dict = {
        "arch": arch,
        "gguf_version": ver,
        "native_context": g("context_length"),
        "n_layer": g("block_count"),
        "n_expert": g("expert_count"),
        "n_expert_used": g("expert_used_count"),
        "quant": (kv.get("general.file_type") if isinstance(kv.get("general.file_type"), str) else None),
        "file_type_id": kv.get("general.file_type"),
        "mtp_head": bool(g("nextn_predict_layers") or 0),
        "multimodal": False,
        "size_label": kv.get("general.size_label"),
        "name": kv.get("general.name"),
    }
    return info
