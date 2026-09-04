"""GDN deep-prefill warm-up.

Qwen3.5-family models (hybrid GDN linear attention: Qwen3.6, Qwen3.8, Qwen3-Next —
archs matching Qwen3_5* / Qwen3Next*) decode at roughly HALF speed for the lifetime
of a freshly launched vLLM process *until it serves its first deep prefill*; one
~32K-token prompt flips the process to full speed permanently. Measured 2026-08-23
on gfx1201 (scratch/rdna4-kernel-tuning-and-4bit-kv-report-20260823.md §E addendum):
a TP4 Qwen3.8 seat sat at 14.2 tok/s/stream through 18 minutes of continuous
short-prompt load, then one 32K prefill (37 s) took it to its rated 40.8 tok/s
single / 142 tok/s @4 within a minute. Non-GDN models (gemma, dense qwen) are at
full speed immediately and skip this.

So: after a GDN seat reports ready, fire one long throwaway prompt before calling
it warm. `launch.up(wait=True)` and `profiles.up_profile` do this by default;
`--no-warmup` skips it.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request

log = logging.getLogger(__name__)

# Arch substrings that use the GDN / linear-attention decode path.
GDN_ARCH_MARKERS = ("Qwen3_5", "Qwen3Next", "Qwen4Exp")  # Qwen4Exp = Qwen3.8-Flash-Next (GDN + QSA)

# One paragraph of neutral filler; ~6.4 chars/token on the Qwen tokenizer.
_PARA = (
    "The fleet manager schedules inference seats across the available GPUs, balancing "
    "context length against concurrency while the router classifies each request by "
    "domain and complexity before dispatching it to the best seat. "
)
_CHARS_PER_TOKEN = 6.4
TARGET_TOKENS = 32_000          # enough to trip the fast path (validated at 32K)
MIN_TOKENS = 8_192              # below this a warm-up is unlikely to help; still try


def needs_gdn_warmup(model: dict, placement: dict | None = None) -> bool:
    """True when this model's arch uses the GDN decode path on a vLLM backend."""
    if placement is not None and (placement.get("backend") or "vllm") != "vllm":
        return False                      # llama.cpp GGUF seats don't have this behavior
    arch = str(((model or {}).get("identity") or {}).get("arch") or "")
    return any(m in arch for m in GDN_ARCH_MARKERS)


def warmup_prompt_tokens(max_model_len: int | None) -> int:
    """Deep-prefill size: 32K, capped to fit the seat's window with headroom."""
    if not max_model_len:
        return TARGET_TOKENS
    return max(MIN_TOKENS // 4, min(TARGET_TOKENS, int(max_model_len) - 4_096))


def gdn_warmup(port: int, served_model_name: str, max_model_len: int | None = None,
               timeout: float = 600.0, bind_address: str = "127.0.0.1") -> dict:
    """Send one deep throwaway prompt to a ready seat. Best-effort: never raises.

    Returns {"ok", "prompt_tokens", "seconds"} (plus "error" on failure).
    """
    tokens = warmup_prompt_tokens(max_model_len)
    n = max(1, int(tokens * _CHARS_PER_TOKEN / len(_PARA)))
    prompt = _PARA * n + "\n\nReply with the single word: ready."
    body = {
        "model": served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://{bind_address}:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.time()
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=timeout))
        seconds = round(time.time() - t0, 1)
        ptok = (resp.get("usage") or {}).get("prompt_tokens")
        return {"ok": True, "prompt_tokens": ptok, "seconds": seconds}
    except Exception as e:  # noqa: BLE001 — warm-up must never fail a launch
        seconds = round(time.time() - t0, 1)
        log.warning("GDN warm-up failed after %.1fs (%s) — seat is up but will decode at ~half speed until its first deep prompt", seconds, e)
        return {"ok": False, "seconds": seconds, "error": str(e)[:200]}
