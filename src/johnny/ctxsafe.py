"""ctxsafe — empirical context-safety probe (`johnny bench <target> --suite ctxsafe`).

**Why this exists.** 2026-08-06: two `Ornith-1.0-397B-Featherweight-v0` llama.cpp
placements were found (manual, ad hoc agent work) to reproducibly crash the whole
container — HIP "Memory access fault... page not present" on GPU0, i.e. real VRAM
exhaustion — at real prefill depths well short of their configured `max_model_len`.
One placement's nominal cap was 262144 but it crashed at n_tokens=59392; the other's
was 409600/parallel and crashed at 47104 tok/slot. Both were fixed by *empirically*
walking real depths with a live seat and real GPU monitoring until a genuinely-safe
ceiling was found — see those two placements' `extra.note` in the registry for the
full methodology and findings. `max_model_len` is a request the launcher honors by
allocating a KV-cache reservation; it is not proof the seat survives a real prompt
that actually reaches that many tokens. This module formalizes that manual process
into reusable, repeatable infrastructure.

**What it does**, per placement:

1. Builds a real long prompt: filler prose (deterministic, tiktoken-measured so the
   requested depth is accurate) with a distinctive numeric "secret code" embedded near
   the start, followed by a question asking for it back at the very end — a genuine
   needle-in-haystack recall probe, not a synthetic token-count-only request.
2. Launches the placement for real on a *dedicated* probe container/port (never the
   induction tuning seat, never a live production seat — this suite deliberately tries
   to crash things, so it must never touch traffic-serving infrastructure).
3. Walks a sweep of depths from shallow to the placement's configured `max_model_len`
   (or an explicit `--limit` ceiling, for placements too large to sweep to the full
   nominal cap in one run — e.g. a 1M-token native_context), sending one real request
   per depth/trial while a background thread polls `rocm-smi --showuse --showmeminfo
   vram --json` every ~2s and records the peak VRAM used on every GPU during that
   request (not just before/after — a crash can happen mid-prefill).
4. Detects three distinct outcomes per trial: crash (container no longer running —
   confirmed via `docker inspect`, not just a dropped connection), a clean failure (the
   seat answered but got the code wrong — a capability/lost-in-the-middle result, not a
   safety one), or a verified pass (correct recall, container still alive).
5. The last two depths in the sweep (the ones nearest the configured ceiling) get an
   extra confirmation trial each — the same "don't trust one lucky pass" rigor the
   manual work used (3-4 trials before calling a depth safe).
6. Stops at the first crash (higher depths are certain to be worse) or at the ceiling.
   `verified_safe_tokens` is the deepest depth that passed every trial; `crashed_at` is
   the depth that killed the container, or null if the whole sweep to the ceiling
   passed clean — in which case the sweep's own shape (ramping up to, and repeat-
   confirming at, the ceiling) already tested "somewhat past" every intermediate safe
   point, which is the real-margin confirmation the manual work also did.

The probe seat is always freshly launched and always torn down (`docker rm -f`,
idempotent even mid-crash) when the suite finishes — see bench.py's `_run_ctxsafe`.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time

from .util import run as _run

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken is an optional/soft dep
    _ENC = None

# Dedicated probe container/port per backend — distinct from EVERY other reserved
# port in this codebase (vLLM induction tuning 9000, llamacpp induction bench 9001,
# llamacpp bench.py tuning 9002) so ctxsafe can never collide with a concurrent
# `johnny tune`/`induct`/`bench` run. ctxsafe deliberately tries to crash its seat, so
# it must own a container name/port nothing else will ever reuse.
VLLM_CONTAINER = "vllm-johnny-ctxsafe-tuning"
VLLM_PORT = 9006
LLAMACPP_CONTAINER = "llamacpp-johnny-ctxsafe-tuning"
LLAMACPP_PORT = 9007

# Depth sweep as fractions of the ceiling (configured max_model_len, or --limit if
# smaller). Deliberately front-loaded with more points near the top: that's where
# tonight's incident actually lived (crashes at ~60% and ~23% of nominal cap on the
# two Featherweight placements) and where the VRAM-vs-depth curve accelerates.
DEPTH_FRACTIONS = (0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0)
MIN_DEPTH_TOKENS = 512
TRIALS_NEAR_CEILING = 2       # extra confirmation trials on the top 2 sweep points
_RESERVE_TAIL_TOKENS = 64     # header + question overhead subtracted from filler budget
_REQUEST_TIMEOUT_FLOOR_S = 180
_ASSUMED_PREFILL_FLOOR_TOK_S = 150   # conservative -> generous per-request timeout


def plan_depths(ceiling_tokens: int, fractions: tuple[float, ...] = DEPTH_FRACTIONS,
                min_tokens: int = MIN_DEPTH_TOKENS) -> list[int]:
    """Ascending, deduplicated real-token depths to test, capped at ceiling_tokens."""
    depths = sorted({max(min_tokens, min(ceiling_tokens, int(ceiling_tokens * f))) for f in fractions})
    return [d for d in depths if d <= ceiling_tokens]


def request_timeout_s(depth_tokens: int) -> float:
    """Generous per-request timeout — deep prefills are slow, and a timeout must never
    be mistaken for a crash (bench.py re-checks container liveness on any exception
    before calling it a crash)."""
    return max(_REQUEST_TIMEOUT_FLOOR_S, depth_tokens / _ASSUMED_PREFILL_FLOOR_TOK_S)


# --------------------------------------------------------------------------- prompt
_VOCAB = (
    "system architecture memory allocation compute node throughput latency scheduler "
    "kernel buffer cache pipeline register thread context switch queue depth window "
    "tensor parallel shard replica checkpoint restore snapshot topology fabric link "
    "bandwidth saturation utilization workload batch inference request response token "
    "sequence embedding attention projection residual normalization gradient optimizer "
    "learning rate warmup decay schedule epoch iteration dataset corpus sample split "
    "validation metric accuracy precision recall latency percentile tail distribution "
    "cluster node rack switch router firewall gateway proxy load balancer replica set "
    "container image manifest registry volume mount namespace process daemon service "
    "endpoint interface protocol handshake negotiation timeout retry backoff circuit "
    "monitor telemetry metric log trace span event alert threshold anomaly baseline"
).split()


def _est_tokens(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)  # ~4 chars/token fallback when tiktoken is unavailable


def _filler_text(n_tokens: int, seed: int) -> str:
    """Deterministic filler prose measured (via tiktoken when available) to land at
    almost exactly n_tokens — the requested depth is real, not a guess. Content is
    meaningless-but-readable technical prose (no repetition pattern a model could
    shortcut past instead of genuinely processing the context)."""
    if n_tokens <= 0:
        return ""
    rng = random.Random(seed)
    parts: list[str] = []
    text = ""
    while _est_tokens(text) < n_tokens:
        sentence = " ".join(rng.choice(_VOCAB) for _ in range(14))
        parts.append(sentence[0].upper() + sentence[1:] + ".")
        text = " ".join(parts)
    if _ENC is not None:
        toks = _ENC.encode(text)[:n_tokens]
        text = _ENC.decode(toks)
    return text


def build_probe(depth_tokens: int, trial: int) -> dict:
    """A real needle-in-haystack prompt at ~depth_tokens: a distinctive numeric code
    near the start, filler prose out to depth, then a question for the code at the
    very end — genuine long-context recall, not just a token-count-shaped request."""
    rng = random.Random(depth_tokens * 1000 + trial)
    code = str(rng.randint(100_000, 999_999))
    header = (
        "This is a long-context capacity probe.\n\n"
        f"SECRET CODE: {code}\n"
        "Remember this code carefully. After a large amount of unrelated filler text "
        "below, you will be asked to repeat it.\n\n---BEGIN FILLER---\n"
    )
    tail = (
        "\n---END FILLER---\n\n"
        "What was the SECRET CODE given at the very start of this message, right "
        "after \"SECRET CODE:\"? Reply with ONLY the 6-digit number, nothing else."
    )
    budget = max(depth_tokens - _est_tokens(header) - _est_tokens(tail) - _RESERVE_TAIL_TOKENS, 32)
    filler = _filler_text(budget, seed=depth_tokens * 1000 + trial)
    prompt = header + filler + tail
    return {"prompt": prompt, "code": code, "requested_depth": depth_tokens}


def score_response(content: str, code: str) -> bool:
    return bool(content) and re.search(rf"\b{re.escape(code)}\b", content) is not None


# --------------------------------------------------------------------------- VRAM
def _poll_vram() -> dict[int, float]:
    """One rocm-smi sample -> {gpu_index: used_gb}. Empty dict on any failure —
    monitoring is best-effort and must never be the reason a probe trial fails."""
    rc, out, _ = _run(["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"], timeout=10)
    if rc != 0 or not out.strip():
        return {}
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return {}
    per_gpu: dict[int, float] = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        m = re.match(r"card(\d+)$", k)
        if not m:
            continue
        used = v.get("VRAM Total Used Memory (B)")
        if used is not None:
            try:
                per_gpu[int(m.group(1))] = round(int(used) / (1024**3), 3)
            except (TypeError, ValueError):
                pass
    return per_gpu


class VramMonitor:
    """Background rocm-smi poller — live monitoring *during* a request, not just a
    before/after snapshot, since tonight's crashes happened mid-prefill."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[int, float]] = []

    def _loop(self) -> None:
        while not self._stop.is_set():
            s = _poll_vram()
            if s:
                self.samples.append(s)
            self._stop.wait(self.interval)

    def start(self) -> None:
        s = _poll_vram()
        if s:
            self.samples.append(s)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def peak_per_gpu(self) -> dict[int, float]:
        peak: dict[int, float] = {}
        for s in self.samples:
            for idx, v in s.items():
                peak[idx] = max(peak.get(idx, 0.0), v)
        return peak

    def overall_peak_gb(self) -> float:
        p = self.peak_per_gpu()
        return round(max(p.values()), 2) if p else 0.0


# --------------------------------------------------------------------------- trial
def run_trial(client, model_id: str, depth_tokens: int, trial: int, container: str, drv,
             timeout_s: float, poll_interval: float = 2.0, thinking: bool = False) -> dict:
    """One real request at ~depth_tokens against the live probe seat, with live VRAM
    polling for the duration. Crash is only ever confirmed via `docker inspect`
    (stages._container_exited) — a request exception with the container still alive is
    a slow/failed request, not a crash.

    `thinking` defaults off (same PLAN §3.6 convention as arc/icl/needle/humaneval):
    a reasoning model left thinking-on burns the whole (small) response budget on an
    unclosed `<think>` block and never reaches the actual answer — that reads as a
    failed recall, not the VRAM/crash finding this suite exists to catch."""
    from .induct import stages

    t0 = time.time()
    result: dict = {"requested_depth": depth_tokens, "trial": trial}
    monitor = VramMonitor(interval=poll_interval)
    extra_body = None if thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    # Everything from prompt construction onward is guarded — a bug in probe-building
    # (or anything else unexpected) must surface as a failed *trial*, not blow up the
    # whole sweep and skip the caller's teardown of the probe seat.
    try:
        probe = build_probe(depth_tokens, trial)
        monitor.start()
        kwargs = dict(model=model_id, messages=[{"role": "user", "content": probe["prompt"]}],
                     max_tokens=32, temperature=0, timeout=timeout_s)
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        content = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        actual = getattr(usage, "prompt_tokens", None) if usage is not None else None
        result.update({
            "ok": score_response(content, probe["code"]),
            "crashed": False,
            "actual_prompt_tokens": actual,
            "response_snippet": content[:120],
        })
    except Exception as e:
        crashed, exit_code = stages._container_exited(drv, container)
        result.update({"ok": False, "crashed": crashed, "exit_code": exit_code,
                       "error": str(e)[:300]})
    monitor.stop()
    result["elapsed_s"] = round(time.time() - t0, 1)
    result["vram_peak_per_gpu_gb"] = monitor.peak_per_gpu()
    result["vram_peak_gb"] = monitor.overall_peak_gb()
    return result
