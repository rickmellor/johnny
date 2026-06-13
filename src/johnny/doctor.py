"""`johnny doctor` — honest preflight checks with guided fixes.

A shareable tool lives or dies on first-run diagnosis: is docker up? is there a
GPU runtime? is the host arch compatible with the container images? is there disk
for a large pull? which backend CLIs exist? Returns structured results so both the
Rich table and `--json` render from the same data. Full hardware detection is P1;
this is a light, fast probe.
"""

from __future__ import annotations

import platform
import shutil
from pathlib import Path

from . import config as C
from .util import run, which

# status values: "ok" | "warn" | "fail"


def _c(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def run_checks() -> list[dict]:
    paths = C.get_paths()
    cf = paths.config_file
    checks: list[dict] = []

    # --- config present & versioned ---
    if cf.exists():
        data = C.load_yaml(cf) or {}
        v = data.get("schema_version")
        if v == C.CONFIG_SCHEMA_VERSION:
            checks.append(_c("config", "ok", f"{cf} (schema v{v})"))
        elif v is None:
            checks.append(_c("config", "warn", f"{cf}: no schema_version — run `johnny migrate`"))
        else:
            checks.append(
                _c("config", "warn", f"{cf}: schema v{v}, tool v{C.CONFIG_SCHEMA_VERSION} — run `johnny migrate`")
            )
    else:
        checks.append(_c("config", "warn", "no config yet — run `johnny init`"))

    # --- docker daemon ---
    rc, out, _ = run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=6)
    if rc == 0 and out.strip():
        checks.append(_c("docker", "ok", f"daemon {out.strip()}"))
    elif which("docker"):
        checks.append(_c("docker", "fail", "docker present but daemon unreachable (vLLM backend needs it)"))
    else:
        checks.append(_c("docker", "fail", "docker not found (required for the vLLM backend)"))

    # --- GPU runtime ---
    vendor = C.detect_gpu_vendor()
    if vendor == "amd":
        checks.append(_c("gpu", "ok", "AMD ROCm (/dev/kfd present)"))
    elif vendor == "nvidia":
        rc, out, _ = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=6)
        cnt = len([ln for ln in out.splitlines() if ln.strip()]) if rc == 0 else 0
        checks.append(_c("gpu", "ok", f"NVIDIA CUDA ({cnt} GPU(s))" if cnt else "NVIDIA CUDA"))
    else:
        checks.append(_c("gpu", "warn", "no AMD(/dev/kfd) or NVIDIA(nvidia-smi) GPU — CPU/LM Studio/Ollama only"))

    # --- host arch (image compatibility) ---
    m = platform.machine()
    note = " — needs ARM64 container images" if m in ("aarch64", "arm64") else ""
    checks.append(_c("arch", "ok", f"{m} on {platform.system()}{note}"))

    # --- disk headroom at the models dir ---
    md = None
    if cf.exists():
        md = (C.load_yaml(cf) or {}).get("roots", {}).get("models_dir")
    target = Path(md).expanduser() if md and Path(md).expanduser().exists() else Path.home()
    try:
        free_gb = shutil.disk_usage(target).free / 1e9
        checks.append(_c("disk", "ok" if free_gb >= 20 else "warn", f"{free_gb:.0f} GB free at {target}"))
    except OSError as e:
        checks.append(_c("disk", "warn", f"couldn't stat {target}: {e}"))

    # --- backend CLIs available ---
    avail = []
    if which("docker"):
        avail.append("vllm(docker)")
    if which("lms"):
        avail.append("lmstudio(lms)")
    if which("ollama"):
        avail.append("ollama")
    checks.append(_c("backends", "ok" if avail else "warn", ", ".join(avail) if avail else "no backend CLIs found"))

    # --- induction readiness: bundled scripts resolve + bench prerequisites ---
    from .bundled import all_status

    cfg_data = C.load_yaml(cf) or {}
    sstat = all_status(cfg_data)
    missing = [k for k, v in sstat.items() if v["source"] == "missing"]
    bench_tools = [t for t in ("bash", "curl", "bc") if not which(t)]
    if missing:
        checks.append(_c("induction", "warn", f"bundled scripts not resolving: {', '.join(sorted(missing))}"))
    elif bench_tools:
        checks.append(_c("induction", "warn", f"tuning bench needs: {', '.join(bench_tools)} (install them)"))
    else:
        nb = sum(1 for v in sstat.values() if v["source"] == "bundled")
        checks.append(_c("induction", "ok", f"{nb} bundled scripts + bench prereqs present"))

    return checks
