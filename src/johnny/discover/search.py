"""HF search + acquire via huggingface_hub.

search(): query the Hub, derive capability badges from tags, and attach a fit
verdict (weights size vs detected VRAM). acquire(): snapshot_download into the
models dir with the HF token; gated/not-found errors become friendly messages.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import auth, fit

# tag -> badge
_BADGES = [
    ("vision", lambda t: any(k in t for k in ("image-text-to-text", "image-to-text", "visual-question-answering"))),
    ("tool-use", lambda t: any("tool" in x for x in t)),
    ("reasoning", lambda t: any("reason" in x for x in t)),
    ("embeddings", lambda t: any(k in t for k in ("sentence-similarity", "feature-extraction"))),
]


# Ordered so compound labels win before their substrings (nvfp4 before fp4).
_QUANT_TOKENS = (
    "nvfp4", "mxfp4", "fp4", "fp8", "w8a8",
    "awq", "gptq", "w4a16", "int4", "4bit",
    "int8", "8bit", "gguf", "bf16", "fp16",
)


def _quant_from_id(repo: str) -> str | None:
    """Best-effort quant label from the repo id (quant repos name it in the id)."""
    low = repo.lower()
    for tok in _QUANT_TOKENS:
        if tok in low:
            return tok
    return None


def _gguf_variant(rfilename: str) -> str:
    """Variant key for a GGUF file: its subdirectory (bartowski-style layout), else
    the filename stem with the -NNNNN-of-NNNNN split suffix removed."""
    if "/" in rfilename:
        return rfilename.split("/", 1)[0]
    stem = rfilename.rsplit(".", 1)[0]
    return re.sub(r"-\d{5}-of-\d{5}$", "", stem)


# '…-30B-A3B-…' style ids name total and active params outright.
_AB_RE = re.compile(r"(\d+(?:\.\d+)?)b[-_]a(\d+(?:\.\d+)?)b", re.IGNORECASE)


def _params_from_id(repo: str) -> tuple[int | None, int | None]:
    m = _AB_RE.search(repo)
    if not m:
        return None, None
    return int(float(m.group(1)) * 1e9), int(float(m.group(2)) * 1e9)


def _active_params(total: int, cfg: dict) -> int | None:
    """Params live per token, for the common routed-MoE decoder shape (DeepSeek/
    GLM/Qwen style): total minus the routed experts NOT selected — attention,
    embeddings, shared experts and dense layers always run. A dense config →
    total; None when the config lacks the fields for the subtraction.
    (Checks out within ~2% on DeepSeek-V3 671B→37B and Qwen3 30B→3.3B.)"""
    if not cfg:
        return None
    experts = next((cfg[k] for k in ("n_routed_experts", "num_routed_experts",
                                     "num_local_experts", "num_experts") if cfg.get(k)), None)
    if not experts:
        return total
    top_k = cfg.get("num_experts_per_tok") or cfg.get("moe_top_k")
    layers, hidden = cfg.get("num_hidden_layers"), cfg.get("hidden_size")
    moe_int = cfg.get("moe_intermediate_size")
    if not (top_k and layers and hidden and moe_int):
        return None
    dense = cfg.get("first_k_dense_replace") or 0
    step = cfg.get("decoder_sparse_step") or 1
    moe_layers = max(0, layers - dense) // step
    # gated MLP: gate+up+down = 3 matrices per expert
    inactive = moe_layers * (experts - top_k) * 3 * hidden * moe_int
    return max(0, total - inactive) or None


def _fetch_config(repo: str, token: str | None) -> dict:
    try:
        import json

        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(repo, "config.json", token=token)) as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _repo_stats(api, repo: str, token: str | None) -> tuple[int, dict[str, int], int | None, int | None]:
    """(total weight bytes, gguf variant -> bytes, total params, active params).

    GGUF presence routes the fit verdict to the llamacpp path (VRAM+RAM), even
    when the repo id names no quant token. Multi-quant GGUF repos host every
    quant level side by side, so the fit must judge one downloadable variant —
    never the sum of all of them.

    Param counts ride the same model_info call (safetensors/gguf metadata);
    active params need config.json (MoE expert math) — the repo's own, else its
    card-declared base_model's (quant/GGUF repos rarely ship one), else an
    id-pattern ('…-30B-A3B-…'). Either may be None when nothing states it."""
    try:
        info = api.model_info(repo, files_metadata=True, token=token)
    except Exception:
        return 0, {}, None, None
    total, variants = 0, {}
    has_config = False
    for s in getattr(info, "siblings", []) or []:
        name = (s.rfilename or "").lower()
        has_config = has_config or name == "config.json"
        size = getattr(s, "size", None)
        if size and name.endswith((".safetensors", ".bin", ".gguf", ".pt")):
            total += size
            if name.endswith(".gguf"):
                key = _gguf_variant(s.rfilename)
                variants[key] = variants.get(key, 0) + size
    st = getattr(info, "safetensors", None)
    id_total, id_active = _params_from_id(repo)
    params = getattr(st, "total", None) or (getattr(info, "gguf", None) or {}).get("total") or id_total
    active = None
    if params and has_config:
        active = _active_params(params, _fetch_config(repo, token))
    if params and active is None:
        base = getattr(getattr(info, "card_data", None), "base_model", None)
        base = base[0] if isinstance(base, list) and base else base
        if isinstance(base, str) and "/" in base:
            active = _active_params(params, _fetch_config(base, token))
    if params and active is None:
        active = id_active
    return total, variants, params, active


# fits and tight are one class — both run fully on GPU, so the larger quant wins.
_VERDICT_RANK = {"fits": 0, "tight": 0, "offload": 1, "wont-fit": 2}


def _gguf_best_fit(variants: dict[str, int], hardware) -> tuple[int, dict]:
    """(size, verdict) of the best GGUF variant: best verdict class first, and within
    it the largest quant — the highest-fidelity variant this hardware can run."""
    best = None
    for name, vbytes in variants.items():
        v = fit.fit_verdict(vbytes, hardware, "gguf", gguf=True)
        key = (_VERDICT_RANK.get(v["verdict"], len(_VERDICT_RANK)), -vbytes)
        if best is None or key < best[0]:
            best = (key, name, vbytes, v)
    _, name, vbytes, v = best
    if len(variants) > 1:
        v = {**v, "variant": name,
             "detail": f"{name}: {v.get('detail', '')} · {len(variants)} quants in repo"}
    return vbytes, v


# suffixes that mark a trailing path component as a file, not the repo name
# (registry identities sometimes carry appended weight files or report junk)
_FILE_SUFFIXES = (".gguf", ".safetensors", ".bin", ".pt", ".md", ".md.")


def _registry_repo_keys(reg: dict) -> set[str]:
    """Normalized org/repo keys for every registry model, for matching HF ids.
    Identity fields are messy (bare names, appended filenames/report junk), so both
    repo_id and local_path are cleaned down to their first two path components;
    bare names without an org can't match an HF id and are skipped."""
    keys: set[str] = set()
    for m in (reg.get("models") or {}).values():
        ident = m.get("identity") or {}
        for raw in (ident.get("repo_id"), ident.get("local_path")):
            if not raw:
                continue
            parts = [p for p in str(raw).strip().split("/") if p]
            if parts and parts[-1].lower().endswith(_FILE_SUFFIXES):
                parts = parts[:-1]
            if len(parts) >= 2:
                keys.add("/".join(parts[:2]).lower())
    return keys


def _load_registry() -> dict:
    """The registry for cross-referencing search hits; never breaks a search."""
    try:
        from ..registry import store
        return store.load()
    except Exception:
        return {}


def _inducted_keys() -> set[str]:
    return _registry_repo_keys(_load_registry())


def _registry_match_rows(reg: dict, query: str, seen_ids: set[str]) -> list[dict]:
    """Rows for registry models matching the query that HF didn't return — appended
    so 'do I already have one?' is answered even when HF ranking buries the repo.
    Their fit verdict is 'inducted': validated placements beat a size heuristic."""
    terms = [t for t in query.lower().split() if t]
    if not terms:
        return []
    rows = []
    for name, m in (reg.get("models") or {}).items():
        ident = m.get("identity") or {}
        hay = " ".join(str(x) for x in (name, ident.get("repo_id"), ident.get("local_path")) if x).lower()
        if not all(t in hay for t in terms):
            continue
        keys = _registry_repo_keys({"models": {name: m}})
        if keys & seen_ids:
            continue  # HF already returned it; the REG flag on that row covers it
        pls = m.get("placements") or []
        decode = max((float((p.get("perf") or {}).get("decode_tok_s") or 0) for p in pls), default=0.0)
        backends = sorted({p.get("backend") for p in pls if p.get("backend")})
        detail = " · ".join(x for x in (
            "/".join(backends),
            f"{decode:g} tok/s decode" if decode else "",
            f"{len(pls)} placement{'s' if len(pls) != 1 else ''}",
        ) if x)
        rows.append({
            "id": min(keys) if keys else name,
            "registry_model": name,
            "downloads": None,
            "gated": False,
            "inducted": True,
            "badges": [],
            "quant": ident.get("quant"),
            "dtype": None,
            "size_gb": None,
            "fit": {"verdict": "inducted", "detail": detail or "in registry"},
        })
    return rows


def search(query: str, hardware, limit: int = 50) -> dict:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {"error": "huggingface_hub not installed"}
    token = auth.get_token()
    api = HfApi()
    try:
        models = list(api.list_models(search=query, sort="downloads", limit=limit, token=token))
    except TypeError:
        # older/newer signature variance — fall back to the minimal call
        models = list(api.list_models(search=query, limit=limit))
    except Exception as e:
        return {"error": f"HF search failed: {e}"}
    with ThreadPoolExecutor(max_workers=8) as pool:
        stats = list(pool.map(lambda m: _repo_stats(api, m.id, token), models))
    reg = _load_registry()
    inducted = _registry_repo_keys(reg)
    results = []
    for m, (total, gguf_variants, params, active) in zip(models, stats):
        tags = [str(t).lower() for t in (getattr(m, "tags", None) or [])]
        badges = [name for name, fn in _BADGES if fn(tags)]
        quant = _quant_from_id(m.id) or ("gguf" if gguf_variants else None)
        if gguf_variants:
            size, verdict = _gguf_best_fit(gguf_variants, hardware)
        else:
            size = total
            verdict = fit.fit_verdict(size, hardware, quant) if size else {"verdict": "unknown", "detail": "size n/a"}
        results.append({
            "id": m.id,
            "downloads": getattr(m, "downloads", None),
            "gated": bool(getattr(m, "gated", False)),
            "inducted": m.id.lower() in inducted,
            "badges": badges,
            "quant": quant,
            "dtype": fit.dtype_fit(quant, hardware),
            "size_gb": round(size / 1e9, 1) if size else None,
            "params": params,
            "active_params": active,
            "fit": verdict,
        })
    seen = {r["id"].lower() for r in results}
    results += _registry_match_rows(reg, query, seen)
    return {"query": query, "results": results}


def _quant_row(api, repo: str, hardware, token: str | None, base: bool = False,
               inducted: set[str] | None = None) -> dict:
    size, gguf_variants, params, active = _repo_stats(api, repo, token)
    quant = _quant_from_id(repo) or ("gguf" if gguf_variants else None)
    if gguf_variants:
        size, verdict = _gguf_best_fit(gguf_variants, hardware)
    else:
        verdict = fit.fit_verdict(size, hardware, quant) if size else {"verdict": "unknown", "detail": "size n/a"}
    return {
        "id": repo,
        "base": base,
        "inducted": repo.lower() in (inducted or set()),
        "quant": quant or ("—" if base else None),
        "dtype": fit.dtype_fit(quant, hardware),
        "size_gb": round(size / 1e9, 1) if size else None,
        "params": params,
        "active_params": active,
        "fit": verdict,
    }


def list_quantizations(base_repo: str, hardware, limit: int = 40) -> dict:
    """Enumerate a base model's quantizations + a dtype-fit verdict per variant.

    Recall is the union of HF's `base_model:quantized:` lineage tag (authoritative
    when set) and a name-based sweep (community quant repos often omit the tag).
    Each row carries whether its compute dtype is natively accelerated *here*, so
    e.g. NVFP4 shows ✗ on RDNA4 while FP8 shows ✓.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {"error": "huggingface_hub not installed"}
    token = auth.get_token()
    api = HfApi()
    found: dict = {}

    def _collect(**kw):
        try:
            for m in api.list_models(sort="downloads", limit=limit, token=token, **kw):
                found.setdefault(m.id, m)
        except Exception:
            pass

    _collect(filter=f"base_model:quantized:{base_repo}")  # lineage tag (precise)
    stem = base_repo.split("/")[-1].lower()
    pre = dict(found)  # tagged-as-quant ids before the loose name sweep
    _collect(search=stem)  # loose recall; filtered below

    inducted = _inducted_keys()
    rows = [_quant_row(api, base_repo, hardware, token, base=True, inducted=inducted)]
    for rid, _m in found.items():
        if rid == base_repo:
            continue
        # Keep loose-sweep hits only when they share the stem AND look quantized;
        # lineage-tagged ids (in `pre`) are trusted as-is.
        if rid not in pre and not (stem in rid.split("/")[-1].lower() and _quant_from_id(rid)):
            continue
        rows.append(_quant_row(api, rid, hardware, token, inducted=inducted))

    # base first, then native-dtype variants, then non-native/unknown; smaller first.
    def _key(r):
        ok = r["dtype"]["ok"]
        rank = 0 if r["base"] else (1 if ok else (3 if ok is False else 2))
        return (rank, r.get("size_gb") or 1e9)

    rows.sort(key=_key)
    return {"base": base_repo, "results": rows[: limit + 1]}


# Non-weight files worth having alongside any single-variant download.
_META_PATTERNS = ["*.md", "*.json", "LICENSE*"]

# Free space that must remain after a download completes.
_DISK_HEADROOM = 10e9


def _variant_patterns(name: str) -> list[str]:
    """Glob patterns covering one GGUF variant in either repo layout: a
    subdirectory (bartowski/unsloth-style) or split/single files named by stem."""
    return [f"{name}/*", f"{name}.gguf", f"{name}-*-of-*.gguf", *_META_PATTERNS]


def _pick_variant(variants: dict[str, int], requested: str | None, hardware) -> tuple[str | None, str | None]:
    """(variant to download, error). (None, None) means download unfiltered —
    the repo has at most one GGUF variant and none was requested."""
    if requested:
        if not variants:
            return None, "--quant only applies to GGUF repos (no .gguf files here)"
        match = next((k for k in variants if k.lower() == requested.lower()), None)
        if match is None:
            return None, f"no quant '{requested}' in repo — available: " + ", ".join(sorted(variants))
        return match, None
    if len(variants) < 2:
        return None, None
    if hardware is None:
        return None, f"{len(variants)} quants in repo — pick one with --quant, or --all for everything"
    _, v = _gguf_best_fit(variants, hardware)
    if v["verdict"] == "wont-fit":
        smallest, sbytes = min(variants.items(), key=lambda kv: kv[1])
        return None, (f"no quant fits this hardware (smallest: {smallest} at {sbytes / 1e9:.0f} GB)"
                      " — force one with --quant, or --all for everything")
    return v["variant"], None


def acquire(repo: str, models_dir: str, variant: str | None = None,
            include: list[str] | None = None, all_files: bool = False,
            hardware=None, dry_run: bool = False) -> dict:
    """Download a repo (or one GGUF variant of it) into the models dir.

    Multi-quant GGUF repos host every quant side by side (often several TB), so
    unless `all_files` or explicit `include` patterns are given, exactly one
    variant is downloaded: `variant` if named, else the best fit for `hardware`.
    Refuses up front when the remaining bytes won't fit on disk. `dry_run`
    returns the resolved plan without downloading."""
    import fnmatch
    import shutil

    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return {"error": "huggingface_hub not installed"}
    token = auth.get_token()
    local = Path(models_dir).expanduser() / repo
    try:
        info = HfApi().model_info(repo, files_metadata=True, token=token)
    except GatedRepoError:
        return {"repo": repo, "error": "gated repo — accept the license on huggingface.co and run `johnny login`"}
    except RepositoryNotFoundError:
        return {"repo": repo, "error": "repository not found (check the id, or it may be private/gated → `johnny login`)"}
    except Exception as e:
        return {"repo": repo, "error": str(e)}

    files = [(s.rfilename, getattr(s, "size", None) or 0) for s in (getattr(info, "siblings", None) or [])]
    selected = None
    if include:
        patterns = list(include)
    elif all_files:
        patterns = None
    else:
        variants: dict[str, int] = {}
        for name, size in files:
            if name.lower().endswith(".gguf"):
                key = _gguf_variant(name)
                variants[key] = variants.get(key, 0) + size
        selected, error = _pick_variant(variants, variant, hardware)
        if error:
            return {"repo": repo, "error": error}
        patterns = _variant_patterns(selected) if selected else None

    def _wanted(name: str) -> bool:
        return patterns is None or any(fnmatch.fnmatch(name, p) for p in patterns)

    need = have = 0
    for name, size in files:
        if not _wanted(name):
            continue
        need += size
        f = local / name
        if f.is_file():
            have += min(f.stat().st_size, size)
    remaining = max(0, need - have)
    free = shutil.disk_usage(local if local.exists() else Path(models_dir).expanduser()).free
    plan = {"repo": repo, "path": str(local), "variant": selected,
            "files": sum(1 for n, _ in files if _wanted(n)),
            "download_gb": round(remaining / 1e9, 1), "free_gb": round(free / 1e9, 1)}
    if remaining + _DISK_HEADROOM > free:
        return {**plan, "error": (f"not enough disk: ~{remaining / 1e9:.0f} GB to download but only "
                                  f"{free / 1e9:.0f} GB free under {models_dir}")}
    if dry_run:
        return plan
    try:
        path = snapshot_download(repo_id=repo, local_dir=str(local), token=token, allow_patterns=patterns)
        return {**plan, "path": path}
    except GatedRepoError:
        return {"repo": repo, "error": "gated repo — accept the license on huggingface.co and run `johnny login`"}
    except Exception as e:
        return {"repo": repo, "error": str(e)}
