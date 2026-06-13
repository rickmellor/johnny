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


def _quant_from_id(repo: str) -> str | None:
    low = repo.lower()
    for tok in ("fp8", "awq", "gptq", "int4", "bf16"):
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
        size = _repo_size_bytes(api, m.id, token)
        verdict = fit.fit_verdict(size, hardware, _quant_from_id(m.id)) if size else {"verdict": "unknown", "detail": "size n/a"}
        results.append({
            "id": m.id,
            "downloads": getattr(m, "downloads", None),
            "gated": bool(getattr(m, "gated", False)),
            "badges": badges,
            "size_gb": round(size / 1e9, 1) if size else None,
            "fit": verdict,
        })
    return {"query": query, "results": results}


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
