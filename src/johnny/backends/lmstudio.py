"""LM Studio driver — read-only spike (§3.3).

The point of this spike is *seam validation*: exercise capabilities/list_local/
runtime_state against a real second backend so the interface isn't quietly
vLLM-shaped before induction/telemetry calcify around it. No launch/stop yet —
the full driver is P7. Skips cleanly (returns empty) when `lms` is absent, which
is the case on a vLLM-only box.

`lms` JSON shapes vary across LM Studio versions, so parsing is intentionally
defensive (try several key names; never raise).
"""

from __future__ import annotations

import json

from ..util import run, which
from .base import Capabilities, Driver, ModelInfo, SeatInfo


class LmStudioDriver(Driver):
    name = "lmstudio"

    def available(self) -> bool:
        return bool(which("lms"))

    def capabilities(self) -> Capabilities:
        return Capabilities(
            kind="api",
            tunable_knobs=False,  # GPU-offload + context + a few load params only
            per_gpu_placement=False,
            metrics=True,  # limited vs vLLM Prometheus
            logs=True,
            structured_output=True,
            jit_native=True,  # JIT-loads on first request
            ttl_native=True,  # idle-TTL eviction built in
        )

    @staticmethod
    def _json(cmd: list[str]):
        rc, out, _ = run(cmd, timeout=10)
        if rc != 0 or not out.strip():
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _rows(data) -> list[dict]:
        if data is None:
            return []
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        for key in ("models", "loaded", "data"):
            v = data.get(key)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
        return []

    def list_local(self) -> list[ModelInfo]:
        if not self.available():
            return []
        rows = self._rows(self._json(["lms", "ls", "--json"]))
        out = []
        for m in rows:
            mid = m.get("modelKey") or m.get("path") or m.get("name") or ""
            out.append(ModelInfo(id=str(mid), path=m.get("path"), backend="lmstudio", extra=m))
        return out

    def runtime_state(self) -> list[SeatInfo]:
        if not self.available():
            return []
        rows = self._rows(self._json(["lms", "ps", "--json"]))
        seats = []
        for s in rows:
            ident = s.get("identifier") or s.get("modelKey") or ""
            model = s.get("modelKey") or s.get("identifier")
            port = s.get("port")
            seats.append(
                SeatInfo("lmstudio", str(ident), model, int(port) if port else None, [], "ready", s)
            )
        return seats

    def probe_model(self, path: str) -> dict:
        # LM Studio owns model metadata; config.json (if a raw HF dir) is still readable.
        return {}

    DEFAULT_PORT = 1234

    def compose(self, spec: dict) -> list[str]:
        """A launch spec -> the `lms load` argv. Knobs map: context_length, gpu offload, ttl.

        LM Studio owns GPU placement + serves all loaded models on one port, so there's
        no per-GPU pinning or per-seat port here (unlike vLLM).
        """
        args = ["lms", "load", spec["model_key"], "--identifier", spec["identifier"], "--yes"]
        if spec.get("context_length"):
            args += ["--context-length", str(spec["context_length"])]
        gpu = spec.get("gpu_offload")
        if gpu is not None:
            args += ["--gpu", str(gpu)]
        if spec.get("ttl"):
            args += ["--ttl", str(spec["ttl"])]
        return args

    def launch(self, spec: dict) -> SeatInfo:
        if not self.available():
            raise RuntimeError("LM Studio CLI (`lms`) not found on this box")
        argv = self.compose(spec)
        rc, _out, errout = run(argv, timeout=300)
        if rc != 0:
            raise RuntimeError(f"`lms load` failed: {errout.strip() or 'unknown error'}")
        return SeatInfo(
            "lmstudio", spec["identifier"], spec.get("model_key"),
            spec.get("port", self.DEFAULT_PORT), [], "loading",
            {"server": f"http://127.0.0.1:{spec.get('port', self.DEFAULT_PORT)}/v1"},
        )

    def stop(self, seat: str) -> None:
        if self.available():
            run(["lms", "unload", seat], timeout=60)

    def metrics(self, seat: str) -> dict:
        # LM Studio exposes nothing near vLLM's Prometheus; degrade visibly (§3.7).
        return {"seat": seat, "source": "unavailable", "note": "LM Studio has no Prometheus endpoint"}

    def logs(self, seat: str, follow: bool = False, tail: int = 200):
        # `lms log stream` is server-global, not per-seat.
        if not self.available():
            return "(lms not available)"
        if follow:
            import subprocess

            return subprocess.call(["lms", "log", "stream"])
        rc, out, _ = run(["lms", "log", "stream", "--no-follow"], timeout=10)
        return out if rc == 0 else "(no per-seat logs; `lms log stream` is server-global)"
