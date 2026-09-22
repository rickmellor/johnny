"""systemd backend — a host process johnny owns through a `systemctl --user` unit.

For seats that are not containers: e.g. the `saint-features` sidecar (nomic-embed +
NVIDIA prompt classifier on the RTX 4080, a torch venv). The unit owns its device and
port; johnny starts/stops it, shows it in `status`, and lets a profile pin it so the
fleet view is complete. Placement shape:

    backend: systemd
    knobs:   {gpu_count: 0}                     # not one of johnny's placed GPUs
    extra:   {unit: saint-features.service, port: 8005, served_model: nomic-embed,
              health: "/health", image: "host · ~/.venvs/llmc · RTX 4080"}

One unit may back several seats (one process serving embeddings AND a classifier):
give each placement its own `extra.seat_name` (default = the unit name). Stopping any
of them stops the unit — they are the same process, and status says so.
"""
from __future__ import annotations

import json
import subprocess
import urllib.request

from .base import Capabilities, Driver, SeatInfo


def _systemctl(*args: str, timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=timeout)


def _healthy(port: int | None, path: str | None) -> bool:
    if not port:
        return True
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path or '/health'}", timeout=1.5) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


class SystemdDriver(Driver):
    name = "systemd"

    def available(self) -> bool:
        try:
            return _systemctl("--version", timeout=5).returncode == 0
        except Exception:
            return False

    def capabilities(self) -> Capabilities:
        return Capabilities(kind="native", tunable_knobs=False, per_gpu_placement=False,
                            metrics=False, logs=True, structured_output=False,
                            jit_native=False, ttl_native=False)

    # -- which units are seats: every registry placement with backend systemd
    @staticmethod
    def _placements() -> list[tuple[str, dict]]:
        try:
            from ..registry import store
            reg = store.load()
        except Exception:
            return []
        out = []
        for model_id, m in (reg.get("models") or {}).items():
            for p in m.get("placements") or []:
                if (p.get("backend") or "") == "systemd" and (p.get("extra") or {}).get("unit"):
                    out.append((model_id, p))
        return out

    @classmethod
    def _unit_for(cls, seat: str) -> str:
        """Seat name → unit (seat names may be `unit#suffix` or a custom extra.seat_name)."""
        for _, p in cls._placements():
            extra = p.get("extra") or {}
            if (extra.get("seat_name") or extra["unit"]) == seat:
                return extra["unit"]
        return seat.split("#", 1)[0]

    def runtime_state(self) -> list[SeatInfo]:
        seats = []
        for model_id, p in self._placements():
            extra = p.get("extra") or {}
            unit = extra["unit"]; seat_name = extra.get("seat_name") or unit
            try:
                r = _systemctl("show", unit, "-p", "ActiveState,SubState,MainPID", timeout=5)
            except Exception:
                continue
            props = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
            active = props.get("ActiveState")
            if active not in ("active", "activating"):
                continue                                   # stopped units are not seats
            port = extra.get("port")
            state = "ready" if active == "active" and _healthy(port, extra.get("health")) else "loading"
            seats.append(SeatInfo(
                "systemd", seat_name, extra.get("served_model") or model_id, int(port) if port else None, [], state,
                {"image": extra.get("image") or "host process",
                 "labels": {"johnny.model": model_id, "johnny.placement": p.get("id", ""), "johnny.unit": unit},
                 "pid": props.get("MainPID"), "substate": props.get("SubState")},
            ))
        return seats

    def launch(self, spec: dict) -> SeatInfo:
        unit = spec["unit"]
        r = _systemctl("start", unit)
        if r.returncode != 0:
            raise RuntimeError(f"systemctl --user start {unit}: {r.stderr.strip() or r.stdout.strip()}")
        return SeatInfo("systemd", spec.get("seat_name") or unit, spec.get("model"), spec.get("port"), [], "loading",
                        {"image": spec.get("image") or "host process",
                         "labels": {"johnny.model": spec.get("model_id", ""), "johnny.placement": spec.get("placement", ""),
                                    "johnny.unit": unit}})

    def stop(self, seat: str) -> None:
        unit = self._unit_for(seat)
        r = _systemctl("stop", unit)
        if r.returncode != 0:
            raise RuntimeError(f"systemctl --user stop {unit}: {r.stderr.strip() or r.stdout.strip()}")

    def metrics(self, seat: str) -> dict:
        return {}

    def logs(self, seat: str, follow: bool = False, tail: int = 200):
        cmd = ["journalctl", "--user", "-u", self._unit_for(seat), "-n", str(tail), "--no-pager"] + (["-f"] if follow else [])
        if follow:
            return subprocess.Popen(cmd)
        return subprocess.run(cmd, capture_output=True, text=True).stdout
