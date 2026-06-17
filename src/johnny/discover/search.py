"""HF search + acquire via huggingface_hub.

search(): query the Hub, derive capability badges from tags, and attach a fit
verdict (weights size vs detected VRAM). acquire(): snapshot_download into the
models dir with the HF token; gated/not-found errors become friendly messages.
"""

from __future__ import annotations

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


def _repo_size_bytes(api, repo: str, token: str | None) -> int:
    try:
        info = api.model_info(repo, files_metadata=True, token=token)
    except Exception:
        return 0
    total = 0
    for s in getattr(info, "siblings", []) or []:
        name = (s.rfilename or "").lower()
        size = getattr(s, "size", None)
        if size and name.endswith((".safetensors", ".bin", ".gguf", ".pt")):
            total += size
    return total


def search(query: str, hardware, limit: int = 10) -> dict:
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
    results = []
    for m in models:
        tags = [str(t).lower() for t in (getattr(m, "tags", None) or [])]
        badges = [name for name, fn in _BADGES if fn(tags)]
        quant = _quant_from_id(m.id)
        size = _repo_size_bytes(api, m.id, token)
        verdict = fit.fit_verdict(size, hardware, quant) if size else {"verdict": "unknown", "detail": "size n/a"}
        results.append({
            "id": m.id,
            "downloads": getattr(m, "downloads", None),
            "gated": bool(getattr(m, "gated", False)),
            "badges": badges,
            "quant": quant,
            "dtype": fit.dtype_fit(quant, hardware),
            "size_gb": round(size / 1e9, 1) if size else None,
            "fit": verdict,
        })
    return {"query": query, "results": results}


def _quant_row(api, repo: str, hardware, token: str | None, base: bool = False) -> dict:
    quant = _quant_from_id(repo)
    size = _repo_size_bytes(api, repo, token)
    verdict = fit.fit_verdict(size, hardware, quant) if size else {"verdict": "unknown", "detail": "size n/a"}
    return {
        "id": repo,
        "base": base,
        "quant": quant or ("—" if base else None),
        "dtype": fit.dtype_fit(quant, hardware),
        "size_gb": round(size / 1e9, 1) if size else None,
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

    rows = [_quant_row(api, base_repo, hardware, token, base=True)]
    for rid, _m in found.items():
        if rid == base_repo:
            continue
        # Keep loose-sweep hits only when they share the stem AND look quantized;
        # lineage-tagged ids (in `pre`) are trusted as-is.
        if rid not in pre and not (stem in rid.split("/")[-1].lower() and _quant_from_id(rid)):
            continue
        rows.append(_quant_row(api, rid, hardware, token))

    # base first, then native-dtype variants, then non-native/unknown; smaller first.
    def _key(r):
        ok = r["dtype"]["ok"]
        rank = 0 if r["base"] else (1 if ok else (3 if ok is False else 2))
        return (rank, r.get("size_gb") or 1e9)

    rows.sort(key=_key)
    return {"base": base_repo, "results": rows[: limit + 1]}


def acquire(repo: str, models_dir: str) -> dict:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return {"error": "huggingface_hub not installed"}
    token = auth.get_token()
    local = Path(models_dir).expanduser() / repo
    try:
        path = snapshot_download(repo_id=repo, local_dir=str(local), token=token)
        return {"repo": repo, "path": path}
    except GatedRepoError:
        return {"repo": repo, "error": "gated repo — accept the license on huggingface.co and run `johnny login`"}
    except RepositoryNotFoundError:
        return {"repo": repo, "error": "repository not found (check the id, or it may be private/gated → `johnny login`)"}
    except Exception as e:
        return {"repo": repo, "error": str(e)}
