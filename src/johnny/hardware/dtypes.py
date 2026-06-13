"""Natively-accelerated dtype detection.

Which low-precision matmul dtypes the GPU implements *in hardware* is the lever
induction and placement reason about (RDNA4: fp8 yes / fp4 no; Blackwell: both).
This is a fact to detect, never assume.

Two paths, in order:
  1. **ISA probe (authoritative, AMD).** Wrap the existing `probe-wmma-opcodes.sh`
     (discovered into config as `scripts.probe_dtypes`): it runs `llvm-mc` inside
     the vLLM image over candidate WMMA mnemonics — no GPU needed — and prints
     ACCEPTED/REJECTED. We parse the ACCEPTED set into dtype flags and cache it by
     (vendor, arch, image) so `detect()` stays fast after the first run.
  2. **Curated arch table (fallback / NVIDIA).** When the probe script or docker is
     absent, or for NVIDIA (no amdgcn assembler path), fall back to a small
     per-arch table. Portable, good-enough, and clearly tagged `source = table`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..util import run, which

# Per-arch native dtype sets (fallback / NVIDIA). Keys: AMD gfx*, NVIDIA sm_*.
ARCH_DTYPE_TABLE: dict[str, set[str]] = {
    # AMD RDNA4 (R9700 / RX 9000) — fp8 yes, fp4 no
    "gfx1200": {"bf16", "fp16", "fp8", "int8", "int4"},
    "gfx1201": {"bf16", "fp16", "fp8", "int8", "int4"},
    # AMD RDNA3 — WMMA has f16/bf16/iu8/iu4, no fp8
    "gfx1100": {"bf16", "fp16", "int8", "int4"},
    "gfx1101": {"bf16", "fp16", "int8", "int4"},
    "gfx1102": {"bf16", "fp16", "int8", "int4"},
    # AMD CDNA3 (MI300) — fp8; CDNA2 (MI200) — no fp8
    "gfx942": {"bf16", "fp16", "fp8", "int8"},
    "gfx90a": {"bf16", "fp16", "int8"},
    # NVIDIA Blackwell — fp4 + fp8
    "sm_100": {"bf16", "fp16", "fp8", "fp4", "int8"},
    "sm_120": {"bf16", "fp16", "fp8", "fp4", "int8"},
    "sm_121": {"bf16", "fp16", "fp8", "fp4", "int8"},
    # NVIDIA Hopper (sm_90) + Ada (sm_89) — fp8, no fp4
    "sm_90": {"bf16", "fp16", "fp8", "int8"},
    "sm_89": {"bf16", "fp16", "fp8", "int8"},
    # NVIDIA Ampere — bf16/fp16/int8, no fp8
    "sm_80": {"bf16", "fp16", "int8"},
    "sm_86": {"bf16", "fp16", "int8"},
    "sm_87": {"bf16", "fp16", "int8"},
}

CACHE_NAME = "dtype_cache.json"


def _dtypes_from_accepted(ops: list[str]) -> set[str]:
    dt: set[str] = set()
    for op in ops:
        if "fp8" in op or "bf8" in op:
            dt.add("fp8")
        if "fp4" in op:
            dt.add("fp4")
        if "bf16" in op:
            dt.add("bf16")
        if "iu8" in op:
            dt.add("int8")
        if "iu4" in op:
            dt.add("int4")
        if re.search(r"_f16", op):
            dt.add("fp16")
    return dt


def run_wmma_probe(script: str, image: str, arch: str, timeout: float = 180.0) -> set[str] | None:
    """Run the WMMA opcode probe in-container; return the native dtype set, or None."""
    rc, out, _ = run(["bash", str(script), str(image), str(arch)], timeout=timeout)
    if not out:
        return None
    accepted = []
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("ACCEPTED"):
            parts = s.split()
            if len(parts) >= 2:
                accepted.append(parts[1])
    if not accepted:
        return None
    return _dtypes_from_accepted(accepted)


def _cache_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / CACHE_NAME


def _load_cache(state_dir: Path | str | None) -> dict:
    if not state_dir:
        return {}
    p = _cache_path(state_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(state_dir: Path | str, cache: dict) -> None:
    p = _cache_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2))


def resolve_native_dtypes(
    vendor: str,
    arch: str,
    *,
    probe_script: str | None = None,
    image: str | None = None,
    state_dir: Path | str | None = None,
    refresh: bool = False,
) -> tuple[set[str], str]:
    """Return (native_dtypes, source) where source ∈ {cache, probe, table, unknown}."""
    key = f"{vendor}:{arch}:{image or ''}"
    cache = _load_cache(state_dir)
    if not refresh and key in cache:
        return set(cache[key]["dtypes"]), "cache"

    # Authoritative ISA probe (AMD only; needs the script, docker, and an image).
    if vendor == "amd" and probe_script and Path(probe_script).exists() and which("docker") and image:
        res = run_wmma_probe(probe_script, image, arch)
        if res:
            if state_dir:
                cache[key] = {"dtypes": sorted(res)}
                _save_cache(state_dir, cache)
            return res, "probe"

    table = ARCH_DTYPE_TABLE.get(arch)
    if table is not None:
        return set(table), "table"
    return set(), "unknown"
