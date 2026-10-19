"""What a seat can ACCEPT — modalities, per model and per live placement.

SAINT dispatches image/audio/document requests and must exclude seats that cannot
serve them, per LIVE seat. Two layers, both derivable and neither fabricated:

* **model level** — `capabilities.modalities` in the registry (written at induction
  from config.json's vision/audio towers), falling back to the legacy boolean
  `capabilities.multimodal`, which meant "has a vision or audio tower" and whose
  safe reading is vision.
* **placement level** — the flags the seat actually launches with. A vision model
  launched text-only (vLLM `--language-model-only` / `--limit-mm-per-prompt image=0`,
  a llama.cpp seat with no `--mmproj`) is a text seat for routing purposes even
  though its weights carry a tower.

`engine.service.resolve` projects this into its `capabilities` key — that is SAINT's
hot path, so every entry point here is total: a malformed registry entry degrades to
the model-level list, never to an exception.
"""

from __future__ import annotations

import json

from .registry import store

#: The vocabulary. "text" is always present for a generative model.
MODALITIES = ("text", "vision", "audio")

_VLM_FLAG = "--language-model-only"
_LIMIT_MM_FLAG = "--limit-mm-per-prompt"
# llama.cpp: the multimodal projector file is the only thing that gives a seat a tower.
_MMPROJ_FLAGS = ("--mmproj", "-mm")


def _order(found) -> list[str]:
    """De-duplicate into MODALITIES order, keeping any vocabulary the module doesn't
    know yet at the end (a future modality is passed through, not silently dropped).
    A generative seat always accepts text, so "text" is ensured."""
    names = [m for m in found if isinstance(m, str) and m]
    out = [m for m in MODALITIES if m in names]
    out += [m for m in names if m not in MODALITIES and m not in out]
    if "text" not in out:
        out.insert(0, "text")
    return out


def model_modalities(model_entry: dict) -> list[str]:
    """Modalities the model's weights can accept. An explicit `capabilities.modalities`
    list wins; otherwise derive from the legacy `capabilities.multimodal` bool
    (vision is the safe reading of "has a tower")."""
    caps = (model_entry or {}).get("capabilities") or {}
    explicit = caps.get("modalities")
    if isinstance(explicit, list):
        return _order(explicit)
    return _order(["vision"] if caps.get("multimodal") else [])


def _flag_value(flags: list[str], name: str) -> str | None:
    """Value of `name` in a raw flag list, accepting `--name value` and `--name=value`."""
    for i, tok in enumerate(flags):
        if tok == name:
            return flags[i + 1] if i + 1 < len(flags) else None
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1]
    return None


def _has_flag(flags: list[str], *names: str) -> bool:
    return any(
        tok == n or tok.startswith(n + "=") for tok in flags for n in names
    )


def _limit_mm(flags: list[str]) -> dict | None:
    """`--limit-mm-per-prompt` as {modality: limit}, or None when the flag is absent
    or unreadable (absent/unreadable ⇒ leave the model-level modalities alone).

    Three spellings show up in real placements: two tokens
    (`--limit-mm-per-prompt image=0,video=0`), one token
    (`--limit-mm-per-prompt=image=0`), and a JSON value (`{"image": 0, "audio": 0}`).
    """
    raw = _flag_value(flags, _LIMIT_MM_FLAG)
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    limits: dict = {}
    try:
        if raw.startswith("{"):
            doc = json.loads(raw)
            if not isinstance(doc, dict):
                return None
            limits = {str(k): int(v) for k, v in doc.items()}
        else:
            for part in raw.split(","):
                key, sep, val = part.partition("=")
                if not sep:
                    return None
                limits[key.strip()] = int(val.strip())
    except (AttributeError, TypeError, ValueError):
        return None
    return limits


def placement_modalities(model_entry: dict, placement: dict | None) -> list[str]:
    """Modalities this placement actually serves: the model's list, narrowed by how
    it is launched. `placement=None` (or an unknown backend) ⇒ the model-level list."""
    mods = model_modalities(model_entry)
    if not placement:
        return mods
    extra = placement.get("extra") or {}
    flags = [f for f in (extra.get("extra_flags") or []) if isinstance(f, str)]
    backend = placement.get("backend")

    if backend == "vllm":
        if _has_flag(flags, _VLM_FLAG):
            return [m for m in mods if m not in ("vision", "audio")]
        limits = _limit_mm(flags)
        if limits is None:
            return mods
        out = list(mods)
        # An image tower disabled while video is off too (or unset) means nothing in
        # the vision path runs; audio is its own tower, so its own limit.
        if limits.get("image") == 0 and limits.get("video", 0) == 0:
            out = [m for m in out if m != "vision"]
        if limits.get("audio") == 0:
            out = [m for m in out if m != "audio"]
        return _order(out)

    if backend == "llamacpp":
        # No projector file loaded ⇒ the towers are never wired up, whatever the GGUF
        # was built from — vision and audio both hinge on that one file.
        has_mmproj = bool(extra.get("mmproj")) or _has_flag(flags, *_MMPROJ_FLAGS)
        if has_mmproj:
            return mods
        return [m for m in mods if m not in ("vision", "audio")]

    # systemd / lmstudio / anything else: johnny doesn't own the launch flags, so the
    # model-level list is all there is.
    return mods


def seat_capabilities(reg: dict, model_id: str | None, placement_id: str | None) -> dict | None:
    """`{"modalities": [...]}` for one (model, placement) pair, or None when the model
    is unknown/absent — the caller (resolve) reports no capabilities rather than a guess."""
    if not model_id:
        return None
    model = store.get(reg or {}, model_id)
    if not model:
        return None
    placement = None
    if placement_id:
        placement = next(
            (p for p in (model.get("placements") or [])
             if isinstance(p, dict) and p.get("id") == placement_id),
            None,
        )
    return {"modalities": placement_modalities(model, placement)}
