"""Hugging Face token handling for gated models (Gemma/Llama/etc.).

Reads the standard HF locations (env or ~/.cache/huggingface/token) so johnny
shares the token with the `hf` CLI and huggingface_hub. `save_token` writes the
standard file (0600).
"""

from __future__ import annotations

import os
from pathlib import Path


def token_path() -> Path:
    base = os.environ.get("HF_HOME")
    if base:
        return Path(base) / "token"
    return Path.home() / ".cache" / "huggingface" / "token"


def get_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(var)
        if v:
            return v.strip()
    p = token_path()
    if p.exists():
        t = p.read_text().strip()
        return t or None
    return None


def has_token() -> bool:
    return bool(get_token())


def save_token(token: str) -> Path:
    p = token_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(token.strip())
    try:
        p.chmod(0o600)
    except OSError:
        pass
    return p
