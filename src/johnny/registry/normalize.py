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
- identity_gaps(mid, m, cfg) -> derivable identity fills (params / quant)
"""

from __future__ import annotations

import re
from pathlib import Path

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


# --------------------------------------------------------------------------- identity
# '…-30B-A3B-…' ids name total and active params outright; '…-397B-…' just the total.
_MOE_PARAMS_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)B-A(\d+(?:\.\d+)?)B(?![A-Za-z0-9])", re.I)
_DENSE_PARAMS_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)B(?![A-Za-z0-9])", re.I)
# GGUF quant labels (incl. unsloth's UD- dynamic prefix) and safetensors dtype tokens.
_QUANT_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:UD-)?IQ\d_[A-Z0-9]+|(?:UD-)?Q\d(?:_[A-Z0-9]+)+"
    r"|MXFP4|NVFP4|BF16|FP16|F16|FP8|INT[48]|AWQ|GPTQ)(?![A-Za-z0-9])", re.I)


def _params_from_names(names: list[str]) -> str | None:
    """'26B-A4B' / '397B' when a name states it. MoE checked across every name before
    the dense fallback, so 'gemma-4-26b' (id) still reads 26B-A4B off its local_path."""
    for rx, fmt in ((_MOE_PARAMS_RE, lambda m: f"{m.group(1)}B-A{m.group(2)}B"),
                    (_DENSE_PARAMS_RE, lambda m: f"{m.group(1)}B")):
        for s in names:
            m = rx.search(str(s))
            if m:
                return fmt(m).upper()
    return None


def _quant_from_names(names: list[str]) -> str | None:
    for s in names:
        m = _QUANT_RE.search(str(s))
        if m:
            tok = m.group(1)
            # GGUF quant labels are conventionally upper (IQ3_XXS); dtype tokens lower (fp8).
            return tok.upper() if re.match(r"(?i)(UD-)?I?Q\d", tok) else tok.lower()
    return None


def identity_gaps(model_id: str, m: dict, cfg: dict | None = None) -> dict:
    """Derivable fills for a model's identity — {field: value} for params/quant that are
    currently empty. params: the GGUF header (general.size_label; needs the weights
    resolvable on disk — see below), else the naming conventions above. quant: the naming
    conventions first (a curated label, e.g. unsloth's UD- dynamic-quant names, wins
    over anything derived), else the actual per-tensor quant mix read from the GGUF
    tensor table across every shard (ground truth — a custom mix has no single true
    quant, so this is the same "significant types by share" label `johnny status`
    shows), else the header's file_type IF it's a string (rare; the int-enum form is
    a single whole-file stamp and provably wrong for a mixed quant — see
    backends.llamacpp.quant_mix_label). Nothing is invented beyond an explicit token
    in a name/header or a measured tensor type — an underivable field stays empty.
    Never overwrites a set value.

    `cfg` is the loaded johnny config (roots.models_dir / roots.nas_dir) — resolution
    goes through config.resolve_weights_path, same as bench._local_path and
    engine.launch.build_spec, so a `nas:`-prefixed local_path (a NAS-only model, no
    local copy) still gets its GGUF header read instead of silently finding nothing."""
    ident = m.get("identity") or {}
    names = [s for s in (ident.get("local_path"), model_id, ident.get("repo_id")) if s]
    need_params, need_quant = not ident.get("params"), not ident.get("quant")

    meta: dict = {}
    shards: list[Path] = []
    lp = ident.get("local_path")
    if (need_params or need_quant) and cfg and lp and str(lp).endswith(".gguf"):
        from .. import config as C

        resolved = C.resolve_weights_path(lp, cfg)
        p = Path(resolved.host_path) if resolved else None
        if p and p.exists():
            try:
                from ..backends.llamacpp import _gguf_metadata, gguf_shard_paths

                meta = _gguf_metadata(p) or {}
                shards = gguf_shard_paths(p)
            except Exception:
                meta = {}

    gaps: dict = {}
    if need_params:
        v = meta.get("size_label") or _params_from_names(names)
        if v:
            gaps["params"] = str(v)
    if need_quant:
        v = _quant_from_names(names)
        if not v and shards:
            try:
                from ..backends.llamacpp import quant_mix, quant_mix_label

                v = quant_mix_label(quant_mix(shards))
            except Exception:
                v = None
        if not v:
            v = meta.get("quant") if isinstance(meta.get("quant"), str) else None
        if v:
            gaps["quant"] = v
    return gaps


def retune_worklist(reg: dict, current: dict | None = None) -> list[dict]:
    """Placements whose fix is a real benchmark, not a normalize pass. Ordered by model."""
    out: list[dict] = []
    for mid, m in sorted((reg.get("models") or {}).items()):
        for p in m.get("placements") or []:
            st = placement_status(p, current)
            if st in NEEDS_RETUNE:
                out.append({"model": mid, "placement": p.get("id"), "status": st})
    return out


# A configured max_model_len at or above this is "large enough that a silent VRAM-vs-
# depth gap is worth catching before it bites" — the 2026-08-06 Ornith incident (see
# AGENTS.md § Context safety) crashed at real depths of 47K-59K tokens on placements
# nominally configured for 262144/409600, so this threshold is deliberately well below
# "huge": a 32K seat splitting 4 ways per llama.cpp's --parallel is already the kind of
# per-slot depth that mattered there.
CTXSAFE_THRESHOLD_TOKENS = 32768


def ctxsafe_worklist(reg: dict, threshold: int = CTXSAFE_THRESHOLD_TOKENS) -> list[dict]:
    """Placements with a large configured max_model_len that have never had a `ctxsafe`
    probe run (`johnny bench <target> --suite ctxsafe`) — the config's own number was
    never empirically checked against a real deep prefill. Ordered by model. This is
    additive to `retune_worklist`: a placement can have great perf/quality numbers and
    still never have had its context safety verified."""
    out: list[dict] = []
    for mid, m in sorted((reg.get("models") or {}).items()):
        for p in m.get("placements") or []:
            mml = (p.get("knobs") or {}).get("max_model_len")
            if not mml or mml < threshold:
                continue
            cs = (p.get("quality") or {}).get("ctxsafe")
            if not cs:
                out.append({"model": mid, "placement": p.get("id"), "max_model_len": mml})
    return out
