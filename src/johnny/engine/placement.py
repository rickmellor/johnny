"""Placement math: free-GPU computation, viability, GPU + port assignment.

Per-GPU and heterogeneous-aware (§3.2): GPUs are picked from free indices,
preferring a contiguous run within one homogeneous group. CPU/pooling placements
need no GPUs.
"""

from __future__ import annotations

# quant -> the native dtype it needs accelerated
_QUANT_DTYPE = {"fp8": "fp8", "awq": "int4", "gptq": "int4", "int4": "int4", "bf16": "bf16", "fp4": "fp4"}


def occupied_gpus(seats) -> set[int]:
    occ: set[int] = set()
    for s in seats:
        occ |= set(s.gpus or [])
    return occ


def free_gpus(hardware, seats) -> list[int]:
    all_idx = {g.index for g in hardware.gpus}
    return sorted(all_idx - occupied_gpus(seats))


def viable(knobs: dict, hardware, free: list[int]) -> tuple[bool, str]:
    gc = knobs.get("gpu_count") or 0
    if gc == 0:
        return True, "cpu/pooling — no GPUs required"
    if len(free) < gc:
        return False, f"needs {gc} free GPU(s), {len(free)} available ({free})"
    q = (knobs.get("quant") or "").lower()
    dtype = _QUANT_DTYPE.get(q)
    nd = set(hardware.native_dtypes)
    if dtype and nd and dtype not in nd:
        return False, f"quant {q} -> {dtype} not natively accelerated here (have {sorted(nd)})"
    return True, "ok"


def assign_gpus(gpu_count: int, hardware, free: list[int]) -> list[int]:
    if not gpu_count:
        return []
    free_set = set(free)
    # Prefer a run within one homogeneous group (NCCL/NUMA locality).
    for g in hardware.groups:
        grp_free = sorted(i for i in g.indices if i in free_set)
        if len(grp_free) >= gpu_count:
            return grp_free[:gpu_count]
    if len(free) >= gpu_count:
        return sorted(free)[:gpu_count]
    return []


def allocate_port(cfg: dict, seats, role: str | None = None) -> int:
    net = cfg.get("network") or {}
    ports = net.get("ports") or {}
    base = ports.get("base", 8000)
    rng = ports.get("range") or [base, base + 19]
    reserved = ports.get("reserved") or {}
    used = {s.port for s in seats if s.port}

    if role == "embeddings" and reserved.get("embeddings") and reserved["embeddings"] not in used:
        return int(reserved["embeddings"])
    if role in ("orchestrator", "chat") and base not in used:
        return int(base)
    reserved_vals = set(reserved.values())
    for p in range(int(rng[0]), int(rng[1]) + 1):
        if p in used or p in reserved_vals:
            continue
        return p
    raise RuntimeError("no free port in configured range")


def pick_placement(placements: list[dict], placement_id: str | None, hardware) -> dict | None:
    if not placements:
        return None
    if placement_id:
        for p in placements:
            if p.get("id") == placement_id:
                return p
        return None
    # Prefer a placement validated on this exact hardware fingerprint.
    fp = hardware.fingerprint
    matches = [p for p in placements if (p.get("validation_key") or {}).get("hardware_fingerprint") == fp]
    return (matches or placements)[0]


def role_for(placement: dict) -> str | None:
    extra = placement.get("extra") or {}
    if extra.get("runner") == "pooling":
        return "embeddings"
    uc = placement.get("use_case")
    if uc == "context":
        return "orchestrator"
    return None
