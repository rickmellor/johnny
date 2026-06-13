"""The backend-driver interface + shared data types.

Read-only methods (capabilities/list_local/runtime_state/probe_model) are
implemented from P2. Mutating methods (launch/stop/metrics/logs) are declared here
but land with the placement/launch engine at P3 — they raise a clear NotImplemented
until then so callers fail loudly rather than silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Capabilities:
    """What a backend can do — drives induction, placement, and UI behavior."""

    kind: str  # "docker" | "api" | "native"
    tunable_knobs: bool  # exposes a sweepable knob surface (vLLM yes; LM Studio/Ollama few)
    per_gpu_placement: bool  # we choose exact GPUs (vLLM yes)
    metrics: bool  # exposes telemetry we can normalize
    logs: bool
    structured_output: bool
    jit_native: bool  # loads on first request itself
    ttl_native: bool  # evicts idle models itself


@dataclass
class SeatInfo:
    """One running model instance as observed at runtime."""

    backend: str
    name: str  # container / process / identifier
    model: str | None  # served-model name
    port: int | None
    gpus: list[int]
    state: str  # ready | loading | running | failed | down
    extra: dict = field(default_factory=dict)


@dataclass
class ModelInfo:
    """A model available in the backend's store / on disk."""

    id: str
    path: str | None
    backend: str
    extra: dict = field(default_factory=dict)


class Driver:
    """Common backend interface. Subclasses override what they support; the engine
    queries capabilities() and never assumes."""

    name: str = "base"

    def available(self) -> bool:
        """True if this backend's CLI/runtime is usable on this box."""
        return True

    def capabilities(self) -> Capabilities:
        raise NotImplementedError

    def list_local(self) -> list[ModelInfo]:
        return []

    def runtime_state(self) -> list[SeatInfo]:
        return []

    def probe_model(self, path: str) -> dict:
        return {}

    # --- mutating ops land at P3 (placement & launch engine) ---
    def launch(self, spec: dict) -> SeatInfo:  # pragma: no cover
        raise NotImplementedError("launch lands at P3 (placement & launch engine)")

    def stop(self, seat: str) -> None:  # pragma: no cover
        raise NotImplementedError("stop lands at P3")

    def metrics(self, seat: str) -> dict:  # pragma: no cover
        raise NotImplementedError("metrics land at P3")

    def logs(self, seat: str, follow: bool = False):  # pragma: no cover
        raise NotImplementedError("logs land at P3")
