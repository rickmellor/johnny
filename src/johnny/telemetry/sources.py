"""Telemetry sources: vLLM Prometheus /metrics (engine-pull).

Parses the subset of vLLM's Prometheus exposition we normalize. Best-effort: a
missing metric is simply absent (the UI shows it as unavailable, never a fake zero).
"""

from __future__ import annotations

import re
import urllib.request

# vLLM metric name -> normalized field.
_GAUGES = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:gpu_cache_usage_perc": "kv_util",
}


def fetch_metrics_text(port: int, host: str = "127.0.0.1", timeout: float = 2.0) -> str | None:
    url = f"http://{host}:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (localhost)
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def parse_prometheus(text: str) -> dict:
    """Return a normalized metric dict from Prometheus exposition text."""
    out: dict = {"source": "engine"}
    if not text:
        return out
    sums: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([\d.eE+-]+)", line)
        if not m:
            continue
        name, _labels, val = m.group(1), m.group(2), m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        if name in _GAUGES:
            # gauges may be per-label; take the max as a coarse fleet view.
            field = _GAUGES[name]
            out[field] = max(out.get(field, 0.0), v) if field in out else v
        elif name in ("vllm:generation_tokens_total", "vllm:prompt_tokens_total"):
            sums[name] = sums.get(name, 0.0) + v
    if "vllm:generation_tokens_total" in sums:
        out["generation_tokens_total"] = sums["vllm:generation_tokens_total"]
    if "vllm:prompt_tokens_total" in sums:
        out["prompt_tokens_total"] = sums["vllm:prompt_tokens_total"]
    # normalize int-ish gauges
    for k in ("running", "waiting"):
        if k in out:
            out[k] = int(out[k])
    return out


def metrics_for_port(port: int) -> dict:
    return parse_prometheus(fetch_metrics_text(port) or "")
