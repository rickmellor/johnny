"""vLLM driver (#1) — docker-based, Linux only.

P2 implements the read-only surface: capabilities, list_local (scan the models
dir for config.json), runtime_state (docker ps + johnny.* labels or inspect +
/v1/models probe), probe_model (read config.json for arch/quant/context/MTP).
launch/stop/metrics/logs land at P3.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..runtime import probe
from ..util import run, which
from .base import Capabilities, Driver, ModelInfo, SeatInfo


_ARCH_CACHE: dict[str, set] = {}  # image -> registered archs (per-process)


class VllmDriver(Driver):
    name = "vllm"

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
            tunable_knobs=True,
            per_gpu_placement=True,
            metrics=True,  # Prometheus /metrics
            logs=True,
            structured_output=True,
            jit_native=False,
            ttl_native=False,
        )

    def list_local(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        if not self.models_dir or not self.models_dir.exists():
            return out
        seen: set[str] = set()
        for cfg in self.models_dir.rglob("config.json"):
            d = cfg.parent
            try:
                rel = d.relative_to(self.models_dir)
            except ValueError:
                continue
            # vendor/model layout; skip deep nesting (HF snapshots etc.)
            if len(rel.parts) > 3 or "snapshots" in rel.parts:
                continue
            key = str(rel)
            if key in seen:
                continue
            seen.add(key)
            out.append(ModelInfo(id=key, path=str(d), backend="vllm"))
        return sorted(out, key=lambda m: m.id)

    def runtime_state(self) -> list[SeatInfo]:
        seats: list[SeatInfo] = []
        if not probe.docker_available():
            return seats
        for c in probe.docker_ps():
            image = c.get("Image", "")
            labels = self._labels(c)
            is_johnny = any(k.startswith("johnny.") for k in labels)
            if "vllm" not in image.lower() and not is_johnny:
                continue
            name = c.get("Names", "")
            ports = probe.host_ports(c.get("Ports", ""))
            # GPU occupancy: johnny label first (P3 stamps it), else inspect env.
            jgpus = labels.get("johnny.gpus")
            gpus = (
                [int(x) for x in jgpus.split() if x.isdigit()]
                if jgpus
                else self._gpus_from_inspect(name)
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
            elif is_johnny:
                state = "loading"  # labelled by johnny but endpoint silent => still loading
            seats.append(
                SeatInfo("vllm", name, model, port, gpus, state, {"image": image, "labels": labels})
            )
        return seats

    @staticmethod
    def _labels(c: dict) -> dict:
        labels: dict = {}
        for kv in (c.get("Labels", "") or "").split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                labels[k.strip()] = v.strip()
        return labels

    @staticmethod
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

    def supported_archs(self, image: str | None = None) -> set[str] | None:
        """Architectures this image's vLLM registers (cached per-process).

        Returns None when it can't be determined (docker missing, query failed) so
        callers can fail open rather than block. Used for induction's arch pre-flight
        — catches 'unsupported model → doomed sweep → mystery timeout' for free.
        """
        img = image or self.image
        if not img:
            return None
        if img in _ARCH_CACHE:
            return _ARCH_CACHE[img] or None
        if not self.available():
            return None
        code = (
            "import json,sys;"
            "from vllm.model_executor.models.registry import ModelRegistry;"
            "sys.stdout.write('JOHNNY_ARCHS='+json.dumps(sorted(ModelRegistry.get_supported_archs())))"
        )
        rc, out, _ = run(["docker", "run", "--rm", "--entrypoint", "python3", img, "-c", code], timeout=180)
        archs: set[str] = set()
        if rc == 0:
            for line in out.splitlines():
                if line.startswith("JOHNNY_ARCHS="):
                    try:
                        archs = set(json.loads(line.split("=", 1)[1]))
                    except json.JSONDecodeError:
                        archs = set()
                    break
        _ARCH_CACHE[img] = archs
        return archs or None

    def probe_model(self, path: str) -> dict:
        cfg_path = Path(path) / "config.json"
        info: dict = {}
        if not cfg_path.exists():
            return info
        try:
            cfg = json.loads(cfg_path.read_text())
        except (json.JSONDecodeError, OSError):
            return info
        tc = cfg.get("text_config") or {}
        archs = cfg.get("architectures") or tc.get("architectures") or []
        info["arch"] = archs[0] if archs else None
        info["native_context"] = cfg.get("max_position_embeddings") or tc.get("max_position_embeddings")
        info["multimodal"] = bool(cfg.get("vision_config") or cfg.get("vision_tower"))
        info["mtp_head"] = bool(
            cfg.get("num_nextn_predict_layers")
            or tc.get("num_nextn_predict_layers")
            or cfg.get("mtp_num_hidden_layers")
            or tc.get("mtp_num_hidden_layers")
        )
        q = cfg.get("quantization_config") or {}
        info["quant"] = q.get("quant_method")
        return info

    # --- mutating ops (P3) ---
    def compose(self, spec: dict) -> list[str]:
        """Pure: a launch spec -> the `docker run` argv (the proven launcher shape).

        Reproduces device/mount/env/flag conventions; binds to bind_address (localhost
        by default); stamps johnny.* labels for label-derived occupancy.
        """
        extra = spec.get("extra") or {}
        knobs = spec.get("knobs") or {}
        gpus = spec.get("gpus") or []
        pooling = extra.get("runner") == "pooling"
        is_cpu = extra.get("device") == "cpu" or (pooling and not gpus)

        args = ["docker", "run", "-d", "--name", spec["container_name"]]
        for k, v in (spec.get("labels") or {}).items():
            args += ["--label", f"{k}={v}"]
        if is_cpu:
            if extra.get("cpuset"):
                args += ["--cpuset-cpus", str(extra["cpuset"])]
        else:
            args += ["--device=/dev/kfd", "--device=/dev/dri", "--group-add=video", "--group-add=render"]
        args += ["--ipc=host", "--shm-size", spec.get("shm_size", "16g")]
        if spec.get("models_dir"):
            args += ["-v", f"{spec['models_dir']}:/models"]
        if spec.get("vllm_cache"):
            args += ["-v", f"{spec['vllm_cache']}:/root/.cache/vllm"]
        args += ["-p", f"{spec.get('bind_address', '127.0.0.1')}:{spec['port']}:8000"]
        if gpus:
            args += ["-e", f"{spec.get('visible_env', 'CUDA_VISIBLE_DEVICES')}={','.join(str(g) for g in gpus)}"]
        # The engine owns GPU pinning; drop any *_VISIBLE_DEVICES carried over in env
        # (the importer captures them from the launcher) so we don't emit duplicates.
        for k, v in (spec.get("env") or {}).items():
            if k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
                continue
            args += ["-e", f"{k}={v}"]
        args += [spec["image"], spec["model_path"], "--served-model-name", spec["served_model_name"]]
        if knobs.get("tensor_parallel_size"):
            args += ["--tensor-parallel-size", str(knobs["tensor_parallel_size"])]
        if knobs.get("max_model_len"):
            args += ["--max-model-len", str(knobs["max_model_len"])]
        if knobs.get("gpu_memory_util"):
            args += ["--gpu-memory-utilization", str(knobs["gpu_memory_util"])]
        if not pooling:
            args += ["--enable-prefix-caching"]
        if knobs.get("max_num_seqs"):
            args += ["--max-num-seqs", str(knobs["max_num_seqs"])]
        if knobs.get("max_num_batched_tokens"):
            args += ["--max-num-batched-tokens", str(knobs["max_num_batched_tokens"])]
        kv = knobs.get("kv_cache_dtype")
        if kv and kv != "auto":
            args += ["--kv-cache-dtype", kv]
        mtp = knobs.get("mtp") or {}
        if mtp.get("enabled"):
            n = mtp.get("num_speculative_tokens") or 2
            args += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": n})]
        if extra.get("tool_call_parser"):
            args += ["--enable-auto-tool-choice", "--tool-call-parser", extra["tool_call_parser"]]
        if extra.get("reasoning_parser"):
            args += ["--reasoning-parser", extra["reasoning_parser"]]
        if extra.get("chat_template"):
            args += ["--chat-template", extra["chat_template"]]
        if pooling:
            args += ["--runner", "pooling"]
        if extra.get("trust_remote_code"):
            args += ["--trust-remote-code"]
        # Raw passthrough for backend-native flags (§3.3) — power users / proven
        # launchers (e.g. --limit-mm-per-prompt). Appended verbatim, last.
        args += list(extra.get("extra_flags") or [])
        return args

    def launch(self, spec: dict) -> SeatInfo:
        run(["docker", "rm", "-f", spec["container_name"]], timeout=20)  # idempotent
        argv = self.compose(spec)
        rc, _out, errout = run(argv, timeout=120)
        if rc != 0:
            raise RuntimeError(f"docker run failed: {errout.strip() or 'unknown error'}")
        return SeatInfo(
            "vllm",
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

            return subprocess.call(cmd)  # streams to terminal
        rc, out, errout = run(cmd, timeout=15)
        return out + errout
