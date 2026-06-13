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
