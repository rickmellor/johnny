"""Registry schema + validation.

The registry is johnny's source of truth: per model an identity + capabilities +
a list of validated, backend-specific placements. We validate with explicit checks
(no heavyweight schema dependency — keeps johnny lean and avoids compiled wheels on
ARM64). `validate()` returns a list of human-readable errors; empty == valid.
"""

from __future__ import annotations

USE_CASES = {"throughput", "latency", "context", None}
SOURCES = {"imported", "induction", "manual"}


def validate(reg: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(reg, dict):
        return ["registry root is not a mapping"]
    if "schema_version" not in reg:
        errors.append("missing top-level schema_version (run `johnny migrate`)")
    models = reg.get("models")
    if models is None:
        errors.append("missing top-level `models`")
        return errors
    if not isinstance(models, dict):
        errors.append("`models` is not a mapping")
        return errors

    for mid, m in models.items():
        where = f"models.{mid}"
        if not isinstance(m, dict):
            errors.append(f"{where}: not a mapping")
            continue
        ident = m.get("identity") or {}
        if not (ident.get("repo_id") or ident.get("local_path")):
            errors.append(f"{where}.identity: needs repo_id or local_path")
        placements = m.get("placements")
        if not isinstance(placements, list):
            errors.append(f"{where}.placements: not a list")
            continue
        for i, p in enumerate(placements):
            pw = f"{where}.placements[{i}]"
            if not isinstance(p, dict):
                errors.append(f"{pw}: not a mapping")
                continue
            if not p.get("backend"):
                errors.append(f"{pw}: missing backend")
            if p.get("use_case") not in USE_CASES:
                errors.append(f"{pw}: use_case {p.get('use_case')!r} not in {sorted(x for x in USE_CASES if x)}")
            if p.get("source") not in SOURCES:
                errors.append(f"{pw}: source {p.get('source')!r} not in {sorted(SOURCES)}")
            vk = p.get("validation_key") or {}
            for key in ("hardware_fingerprint", "backend", "runtime_version"):
                if not vk.get(key):
                    errors.append(f"{pw}.validation_key: missing {key}")
    return errors
