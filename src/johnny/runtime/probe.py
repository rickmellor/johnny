"""Read-only runtime probing: docker introspection + endpoint liveness.

P0 uses this to reproduce "today's view" (what's running right now, derived from
docker + `/v1/models`). The full backend-driver abstraction and label-derived
occupancy land at P2/P3; this is the dependency-light stateless core.
"""

from __future__ import annotations

import json
import urllib.request

from ..util import run

# Images that look like an inference seat (best-effort filter for P0 status).
INFER_IMAGE_HINTS = (
    "vllm",
    "ollama",
    "llama",
    "lmstudio",
    "sglang",
    "tgi",
    "text-generation",
)


def docker_available() -> bool:
    rc, _, _ = run(["docker", "info"], timeout=6)
    return rc == 0


def docker_ps() -> list[dict]:
    """Running containers as a list of dicts (docker's --format json, one per line)."""
    rc, out, _ = run(["docker", "ps", "--format", "{{json .}}"], timeout=10)
    if rc != 0:
        return []
    rows: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def host_ports(ports_field: str | None) -> list[int]:
    """Parse published host ports from docker's Ports field.

    e.g. "0.0.0.0:8000->8000/tcp, :::8000->8000/tcp" -> [8000]
    """
    res: set[int] = set()
    for part in (ports_field or "").split(","):
        part = part.strip()
        if "->" not in part:
            continue
        left = part.split("->", 1)[0]
        if ":" in left:
            hp = left.rsplit(":", 1)[1]
            if hp.isdigit():
                res.add(int(hp))
    return sorted(res)


def probe_models(port: int, host: str = "127.0.0.1", timeout: float = 2.0) -> str | None:
    """First served model id from an OpenAI-compatible /v1/models, or None if unreachable."""
    url = f"http://{host}:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (localhost)
            data = json.loads(r.read().decode())
        items = data.get("data") or []
        if items:
            return items[0].get("id")
        return None
    except Exception:
        return None


def list_seats() -> list[dict]:
    """Derive the current seat view from docker + endpoint probes.

    Returns dicts: {seat, image, port, model, state}. state is "ready" when an
    endpoint answers /v1/models, else "running" (container up, endpoint silent —
    loading or non-OpenAI). Distinguishing loading vs failed is a P3 concern.
    """
    seats: list[dict] = []
    for c in docker_ps():
        name = c.get("Names", "")
        image = c.get("Image", "")
        ports = host_ports(c.get("Ports", ""))
        looks_infer = any(k in image.lower() for k in INFER_IMAGE_HINTS)

        model = None
        served_port = None
        for p in ports:
            mid = probe_models(p)
            if mid:
                model, served_port = mid, p
                break

        if not looks_infer and model is None:
            continue  # unrelated container

        if served_port is None and ports:
            served_port = ports[0]
        seats.append(
            {
                "seat": name,
                "image": image,
                "port": served_port,
                "model": model,
                "state": "ready" if model else "running",
            }
        )
    return seats
