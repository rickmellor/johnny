"""Placement normalization — one canonical shape + honest status.

The registry accreted cruft as the tool evolved: manually-authored and old-imported
placements have holes (missing perf, source, validation_key), and
`validation_key.runtime_version` came to mean three different things — a vLLM image tag
(`v0.20.2`), a llama.cpp build (`5b36105b-novmm`), a GPU arch (`gfx1201`). Induction-
written placements are already consistent; the drift is in everything hand-touched.

This module gives every placement a consistent *shape* WITHOUT fabricating measured
data. A missing benchmark stays visibly missing (status `unmeasured`); an aborted run
with no provenance stays `incomplete`. We never write a tok/s number that wasn't
measured — that's `johnny tune`'s job, not normalize's.

Entry points:
- normalize_placement(p)      -> structurally-canonical placement (safe to persist)
- normalization_changes(p)    -> human-readable list of what normalize would change
- placement_status(p, current)-> validated | unmeasured | incomplete | stale | unverified
- placement_view(p, current)  -> flat dict for the fixed-column table / picker line
- retune_worklist(reg, current) -> placements that need real numbers (`johnny tune`)
- current_runtimes(cfg)       -> {backend: current launch image} for staleness checks
"""

from __future__ import annotations

# Derived, never stored blindly (see the module docstring on why status is computed on
# read rather than stamped): a stored status is exactly the kind of field that drifts.
STATUS_VALIDATED = "validated"    # has a measurement AND provenance (hardware fingerprint)
STATUS_UNMEASURED = "unmeasured"  # provenance present, but no tok/s recorded yet
STATUS_INCOMPLETE = "incomplete"  # no provenance and no numbers — an aborted/stub entry
STATUS_STALE = "stale"            # measured, but against a runtime you no longer launch
STATUS_UNVERIFIED = "unverified"  # numbers present but no provenance to trust them

# Statuses whose fix is a benchmark run, not a normalize pass.
NEEDS_RETUNE = {STATUS_UNMEASURED, STATUS_INCOMPLETE, STATUS_STALE, STATUS_UNVERIFIED}


# --------------------------------------------------------------------------- derivations
def _knobs(p: dict) -> dict:
    return p.get("knobs") or {}


def gpu_count(p: dict) -> int | None:
    """Cards the seat expects. Present cross-backend; for vLLM it equals TP, so we can
    back-fill it from tensor_parallel_size when a legacy placement omitted it."""
    k = _knobs(p)
    gc = k.get("gpu_count")
    if gc:
        return int(gc)
    tp = k.get("tensor_parallel_size")
    return int(tp) if tp else None


def is_cpu(p: dict) -> bool:
    """A CPU/pooling placement (embeddings, CPU-offload) — no GPUs at all. vLLM marks
    these with device=cpu / runner=pooling; a gpu_count of 0 says the same structurally."""
    extra = p.get("extra") or {}
    if extra.get("device") == "cpu" or extra.get("runner") == "pooling":
        return True
    return _knobs(p).get("gpu_count") == 0


def tp_label(p: dict) -> str:
    """The parallelism knob, rendered per backend. CPU/pooling placements show 'CPU'
    (they take no cards). vLLM has a real TP; llama.cpp instead tensor-splits across every
    offloaded card (no TP param), which we show as 'split'."""
    if is_cpu(p):
        return "CPU"
    k = _knobs(p)
    tp = k.get("tensor_parallel_size")
    if tp:
        return str(tp)
    if p.get("backend") == "llamacpp":
        return "split" if (gpu_count(p) or 0) > 1 else "1"
    return "—"


def perf_pair(p: dict) -> tuple[float | None, float | None]:
    perf = p.get("perf") or {}
    return perf.get("peak_tok_s"), perf.get("single_stream_tok_s")


def _has_perf(p: dict) -> bool:
    peak, single = perf_pair(p)
    return peak is not None or single is not None


def _has_provenance(p: dict) -> bool:
    vk = p.get("validation_key") or {}
    return bool(vk.get("hardware_fingerprint"))


def tool_label(p: dict) -> str:
    """Short provenance for the TOOL column: the recorded runtime_version if any, else
    the launch image's tag. This is the field whose meaning drifted, so we surface it
    verbatim for human judgement rather than pretending it's uniform."""
    vk = p.get("validation_key") or {}
    rv = vk.get("runtime_version")
    if rv:
        return str(rv)
    img = p.get("image") or ""
    return img.rsplit(":", 1)[-1] if ":" in img else "—"


def current_runtimes(cfg: dict | None) -> dict:
    """Map backend -> the image johnny would launch today (from config `docker.*`).
    Used to flag placements pinned to a runtime you've since moved off of."""
    docker = (cfg or {}).get("docker") or {}
    return {"vllm": docker.get("vllm_image"), "llamacpp": docker.get("llamacpp_image")}


def _tag(image: str | None) -> str | None:
    """The version tag of a docker image (`repo:tag` -> `tag`). We compare tags, not full
    repo paths, so a CPU vs GPU variant of the same version (vllm-openai-cpu vs -rocm,
    same `v0.20.2`) doesn't read as stale — only a genuine version change does."""
    return image.rsplit(":", 1)[-1] if image and ":" in image else image


def _is_stale(p: dict, current: dict | None) -> bool:
    if not current:
        return False
    cur = current.get(p.get("backend"))
    img = p.get("image")
    if not cur or not img:
        return False
    return _tag(img) != _tag(cur)


def placement_status(p: dict, current: dict | None = None) -> str:
    has_perf, has_prov = _has_perf(p), _has_provenance(p)
    if not has_prov and not has_perf:
        return STATUS_INCOMPLETE
    if _is_stale(p, current):
        return STATUS_STALE
    if has_perf and has_prov:
        return STATUS_VALIDATED
    if has_prov:  # provenance but no numbers
        return STATUS_UNMEASURED
    return STATUS_UNVERIFIED  # numbers but nothing to trust them against


def placement_view(p: dict, current: dict | None = None) -> dict:
    """Flat, display-ready fields — the single source the table and picker both render."""
    k = _knobs(p)
    peak, single = perf_pair(p)
    return {
        "id": p.get("id") or "",
        "backend": p.get("backend") or "?",
        "dtype": k.get("quant"),          # weights dtype/quant (placement override; may fall back to identity)
        "kv": k.get("kv_cache_dtype"),    # KV-cache dtype (vLLM); None for backends without the knob
        "gpus": gpu_count(p),
        "tp": tp_label(p),
        "priority": p.get("use_case") or "balanced",
        "mml": k.get("max_model_len"),
        "peak": peak,
        "single": single,
        "status": placement_status(p, current),
        "tool": tool_label(p),
        "source": p.get("source") or "—",
    }


# --------------------------------------------------------------------------- normalize
def normalize_placement(p: dict) -> dict:
    """Return a structurally-canonical copy: consistent field *shape*, no invented data.

    - gpu_count back-filled from TP when a legacy vLLM placement omitted it
    - perf given a stable {peak_tok_s, single_stream_tok_s} shape (null == unmeasured)
    - source defaulted to 'manual' for hand-authored entries that predate source-stamping
    - validation_key.backend filled from the placement backend (derivable, unlike the
      hardware_fingerprint / runtime_version, which we refuse to fabricate)
    - validated_at given a null placeholder so the field is always present
    """
    p = dict(p or {})

    knobs = dict(p.get("knobs") or {})
    if not knobs.get("gpu_count"):
        gc = gpu_count(p)
        if gc:
            knobs["gpu_count"] = gc
    p["knobs"] = knobs

    perf = p.get("perf") or {}
    p["perf"] = {
        "peak_tok_s": perf.get("peak_tok_s"),
        "single_stream_tok_s": perf.get("single_stream_tok_s"),
    }

    if not p.get("source"):
        p["source"] = "manual"

    vk = p.get("validation_key")
    if isinstance(vk, dict) and not vk.get("backend") and p.get("backend"):
        p["validation_key"] = {**vk, "backend": p["backend"]}

    if "validated_at" not in p:
        p["validated_at"] = None

    return p


def normalization_changes(raw: dict) -> list[str]:
    """Human-readable diff of what normalize_placement would change (for the preview).
    Empty list == already canonical."""
    canon = normalize_placement(raw)
    changes: list[str] = []

    rk, ck = raw.get("knobs") or {}, canon.get("knobs") or {}
    if rk.get("gpu_count") != ck.get("gpu_count"):
        changes.append(f"knobs.gpu_count: {rk.get('gpu_count')} → {ck.get('gpu_count')} (derived from TP)")

    if (raw.get("perf") or {}) != canon["perf"]:
        peak, single = canon["perf"]["peak_tok_s"], canon["perf"]["single_stream_tok_s"]
        shown = "unmeasured" if peak is None and single is None else f"peak={peak} single={single}"
        changes.append(f"perf → {{{shown}}} shape")

    if raw.get("source") != canon.get("source"):
        changes.append(f"source: {raw.get('source')!r} → {canon['source']!r}")

    rvk, cvk = raw.get("validation_key") or {}, canon.get("validation_key") or {}
    if rvk.get("backend") != cvk.get("backend"):
        changes.append(f"validation_key.backend → {cvk.get('backend')!r}")

    if "validated_at" not in raw:
        changes.append("validated_at → null (add placeholder)")

    return changes


def retune_worklist(reg: dict, current: dict | None = None) -> list[dict]:
    """Placements whose fix is a real benchmark, not a normalize pass. Ordered by model."""
    out: list[dict] = []
    for mid, m in sorted((reg.get("models") or {}).items()):
        for p in m.get("placements") or []:
            st = placement_status(p, current)
            if st in NEEDS_RETUNE:
                out.append({"model": mid, "placement": p.get("id"), "status": st})
    return out
