"""Hardware detection — the portability keystone (§3.2).

`detect() -> Hardware`: vendor, per-GPU VRAM, arch, native dtypes, host RAM, and a
stable fingerprint. Heterogeneous-aware: GPUs are clustered into homogeneous
*groups* (same arch + VRAM), and the fingerprint is per-group so a config tuned on
a gfx1201/32GB pair is recognized on any matching group, regardless of how many
odd cards share the box.

Sources, all host-side where possible (no docker for enumeration):
- AMD: `rocm-smi --json` for per-GPU VRAM/name + KFD topology (`gfx_target_version`)
  for arch. (amd-smi is often container-only; rocm-smi + sysfs avoid a docker hop.)
- NVIDIA: `nvidia-smi --query-gpu`.
Native dtypes come from hardware/dtypes.py (ISA probe, cached; table fallback).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import config as C
from ..util import run
from . import dtypes as D


@dataclass
class GPU:
    index: int
    name: str
    vram_gb: float
    arch: str
    vendor: str


@dataclass
class GpuGroup:
    """A homogeneous set of GPUs (same arch + VRAM)."""

    vendor: str
    arch: str
    vram_gb: float
    count: int
    indices: list[int]
    native_dtypes: list[str]
    fingerprint: str  # per-kind: f"{vendor}-{arch}-{vram}g"


@dataclass
class Hardware:
    vendor: str | None
    gpus: list[GPU]
    groups: list[GpuGroup]
    homogeneous: bool
    unified_memory: bool
    total_vram_gb: float
    host_ram_gb: float
    native_dtypes: list[str]  # union across groups (per-group is authoritative)
    dtype_source: str
    fingerprint: str  # box-level, e.g. "4xamd-gfx1201-32g"


# --------------------------------------------------------------------------- helpers
def _host_ram_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError):
        pass
    return 0.0


def _gfx_from_version(v: int) -> str | None:
    """KFD gfx_target_version -> gfx string. 120001 -> gfx1201; 90010 -> gfx90a."""
    if v <= 0:
        return None
    major = v // 10000
    minor = (v // 100) % 100
    step = v % 100
    return f"gfx{major}{minor}{step:x}"


def _kfd_archs() -> list[str]:
    """gfx arch per compute node from KFD topology (skips the CPU node)."""
    base = Path("/sys/class/kfd/kfd/topology/nodes")
    if not base.exists():
        return []
    archs: list[str] = []
    nodes = sorted(base.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 999)
    for nd in nodes:
        props = nd / "properties"
        if not props.exists():
            continue
        d: dict[str, str] = {}
        for line in props.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                d[parts[0]] = parts[1]
        if int(d.get("simd_count", "0")) > 0:
            g = _gfx_from_version(int(d.get("gfx_target_version", "0")))
            if g:
                archs.append(g)
    return archs


def _detect_amd() -> list[GPU]:
    archs = _kfd_archs()
    rc, out, _ = run(["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"], timeout=15)
    cards: dict = {}
    if rc == 0 and out.strip():
        try:
            cards = json.loads(out)
        except json.JSONDecodeError:
            cards = {}
    items = [(k, v) for k, v in cards.items() if isinstance(v, dict) and "VRAM Total Memory (B)" in v]
    items.sort(key=lambda kv: kv[0])
    gpus: list[GPU] = []
    for i, (_card, info) in enumerate(items):
        vram = int(info.get("VRAM Total Memory (B)", 0)) / 1024**3
        name = info.get("Card SKU") or info.get("Card series") or "AMD GPU"
        arch = archs[i] if i < len(archs) else (archs[0] if archs else "unknown")
        gpus.append(GPU(i, str(name), round(vram, 1), arch, "amd"))
    # Fallback: KFD knows the GPUs even if rocm-smi gave nothing.
    if not gpus and archs:
        gpus = [GPU(i, "AMD GPU", 0.0, a, "amd") for i, a in enumerate(archs)]
    return gpus


def _detect_nvidia() -> list[GPU]:
    rc, out, _ = run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
        timeout=15,
    )
    gpus: list[GPU] = []
    if rc != 0:
        return gpus
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            mem_gb = float(parts[2]) / 1024.0  # MiB -> GiB
        except ValueError:
            continue
        arch = f"sm_{parts[3].replace('.', '')}"
        gpus.append(GPU(idx, parts[1], round(mem_gb, 1), arch, "nvidia"))
    return gpus


def _unified_memory(vendor: str | None, gpus: list[GPU]) -> bool:
    # Discrete by default. Detection for Grace-Blackwell / APUs is refined when we
    # have such a box in the test loop (P-later); discrete VRAM is the common case.
    return False


def detect(refresh: bool = False) -> Hardware:
    paths = C.get_paths()
    cfg = C.load_yaml(paths.config_file) or {}
    vendor = C.detect_gpu_vendor()

    if vendor == "amd":
        gpus = _detect_amd()
    elif vendor == "nvidia":
        gpus = _detect_nvidia()
    else:
        gpus = []

    probe_script = (cfg.get("scripts") or {}).get("probe_dtypes")
    image = (cfg.get("docker") or {}).get("vllm_image")

    # Cluster into homogeneous groups (vendor, arch, rounded VRAM).
    groups_map: dict[tuple, list[GPU]] = {}
    for g in gpus:
        groups_map.setdefault((g.vendor, g.arch, round(g.vram_gb)), []).append(g)

    groups: list[GpuGroup] = []
    all_dtypes: set[str] = set()
    source_used = "none"
    for (gv, arch, vg), members in sorted(groups_map.items()):
        dts, src = D.resolve_native_dtypes(
            gv, arch, probe_script=probe_script, image=image, state_dir=paths.state_dir, refresh=refresh
        )
        all_dtypes |= dts
        source_used = src
        groups.append(
            GpuGroup(
                vendor=gv,
                arch=arch,
                vram_gb=float(vg),
                count=len(members),
                indices=[m.index for m in members],
                native_dtypes=sorted(dts),
                fingerprint=f"{gv}-{arch}-{vg}g",
            )
        )

    if groups:
        fingerprint = "+".join(f"{g.count}x{g.fingerprint}" for g in sorted(groups, key=lambda x: x.fingerprint))
    else:
        fingerprint = f"{vendor or 'none'}-cpu"

    return Hardware(
        vendor=vendor,
        gpus=gpus,
        groups=groups,
        homogeneous=len(groups) <= 1,
        unified_memory=_unified_memory(vendor, gpus),
        total_vram_gb=round(sum(g.vram_gb for g in gpus), 1),
        host_ram_gb=_host_ram_gb(),
        native_dtypes=sorted(all_dtypes),
        dtype_source=source_used,
        fingerprint=fingerprint,
    )
