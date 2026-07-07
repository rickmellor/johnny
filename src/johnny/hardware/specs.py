"""Curated GPU AI-compute spec DB — pulled once, cached, honest provenance.

There's no clean per-card spec API, and matrix 'AI TOPS' can't be computed without
fabricating architectural constants — so these are manufacturer spec-sheet numbers,
curated per gfx/sm arch with a source URL + as-of date. On first `hinfo` the bundled
seed is copied into a writable cache under the state dir (the "pull"); later runs read
the cache. `hinfo --refresh-specs` re-seeds. (A future johnny can point the pull at a
hosted URL — the cache/fallback machinery is already here; only the fetch call changes.)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_BUNDLED = Path(__file__).parent / "data" / "gpu_specs.json"


def _cache_path(state_dir: Path) -> Path:
    return Path(state_dir) / "specs" / "gpu_specs.json"


def load_specs(state_dir, refresh: bool = False) -> dict:
    """Return the spec DB, seeding the state-dir cache from the bundled DB on first use
    (or when refresh). Falls back to the bundled copy if the cache can't be written/read."""
    cache = _cache_path(Path(state_dir))
    if refresh or not cache.exists():
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_BUNDLED, cache)
        except OSError:
            pass  # unwritable state dir — read the bundled copy directly below
    for p in (cache, _BUNDLED):
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
    return {"archs": {}}


def spec_for(specs: dict, arch: str, cu_count: int | None = None) -> dict | None:
    """An arch's AI-compute spec, scaled to the detected CU count when it differs from the
    reference die (flagged `approx`). None for archs with no cached spec — never guessed."""
    entry = (specs.get("archs") or {}).get(arch)
    if not entry:
        return None
    cu_ref = entry.get("cu_ref")
    scale, approx = 1.0, False
    if cu_ref and cu_count and cu_count != cu_ref:
        scale, approx = cu_count / cu_ref, True

    def sc(v):
        return round(v * scale) if isinstance(v, (int, float)) else v

    return {
        "label": entry.get("label"),
        "int8_matrix_tops": sc(entry.get("int8_matrix_tops")),
        "int8_matrix_tops_sparse": sc(entry.get("int8_matrix_tops_sparse")),
        "fp16_matrix_tflops": sc(entry.get("fp16_matrix_tflops")),
        "fp16_matrix_tflops_sparse": sc(entry.get("fp16_matrix_tflops_sparse")),
        "source": entry.get("source"),
        "as_of": entry.get("as_of"),
        "approx": approx,
    }
