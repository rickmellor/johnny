#!/usr/bin/env python3
"""
Realistic-load benchmark for any OpenAI-compatible endpoint (stdlib only).

Why this exists (vs the `perf` suite, scripts/bench.sh)
-------------------------------------------------------
`perf` fires ~5-token prompts with 100-token outputs through curl. That measures batched
decode and almost nothing else. Real serving looks different: a 2800-in / 250-out request
is ~92 % prefill by token count, so the bottlenecks are prefill compute, scheduler
behaviour, KV-cache pressure and queueing (TTFT), none of which `perf` touches. We used
to measure this ad hoc with `vllm bench serve` inside the containers; this script gives
the same numbers (req/s, prefill/output/total tok/s, TTFT, TPOT, end-to-end percentiles,
per-concurrency sweep) with no dependencies, against vLLM, llama.cpp or anything else
that speaks the OpenAI streaming API.

Method: closed-loop load
------------------------
For each concurrency level C, exactly C requests are kept in flight: C worker threads
each send a request, wait for the stream to finish, and immediately send the next, until
`--requests-per-level` requests have been issued and drained. Throughput is
(successful work) / (wall time from first send to last completion). Short runs under-read
steady-state throughput by 10-40 % (ramp-up and the draining tail are a larger share of
the run), so the default is 10 x C requests per level (minimum 20); raise it for numbers
you intend to quote. `--warmup` requests are sent first and not counted.

Every request streams (`stream: true`, `stream_options.include_usage`) with
`temperature: 0` and `ignore_eos: true` so outputs run the full `--output-tokens`
(vLLM and llama.cpp honour ignore_eos; other servers ignore the unknown field, in which
case output lengths -- and therefore output tok/s -- reflect whatever the model chose).
Per request we record TTFT (first text/content delta), end-to-end time, completion tokens
(final usage chunk, else a count of content deltas) and prompt tokens (usage, else the
calibrated estimate). TPOT = (e2e - ttft) / (completion_tokens - 1).

Token calibration (there is no tokenizer here)
----------------------------------------------
Before the sweep ONE non-streamed request with a K-word prompt and max_tokens=1 is sent;
`usage.prompt_tokens / words` gives tokens-per-word for this model + dataset (the ratio
deliberately absorbs the header line and any chat-template overhead, because K is chosen
close to the final prompt size). Prompts are then sized to land within ~3 % of
`--input-tokens`. If the server reports no usage the script falls back to 1.35
tokens/word and says so; the prompt-token figures are then estimates.

Caveat: these prompts are deliberately hostile to the prefix cache
------------------------------------------------------------------
Every prompt starts with a unique random header line and continues with randomly ordered
words, so no two prompts share a prefix and prefix caching cannot help. That is the
honest lower bound. A real workload with a shared system prompt / few-shot block /
conversation history WILL do better than the numbers reported here.

Output: a table row per level, then a final machine-readable line
    LOAD_RESULT {"input_tokens":..,"output_tokens":..,"tokens_per_word":..,"levels":[..],"best":{..}}
Exit status: 0 if at least one level completed requests, 1 if nothing completed,
2 on usage errors.

Usage:
  load_bench.py --base-url http://127.0.0.1:8002/v1 --model chat \
      [--input-tokens 2800] [--output-tokens 250] [--concurrency 8,16,32] \
      [--requests-per-level N] [--dataset words|random] [--seed N] [--timeout 900] \
      [--api-key EMPTY] [--endpoint-kind completions|chat] [--warmup 4] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator

FALLBACK_TOKENS_PER_WORD = 1.35
# First-guess tokens/word per dataset, used only to size the calibration prompt.
CALIBRATION_GUESS = {"words": 1.35, "random": 3.5}
HEADER_WORDS = 3  # "<hex> request <n>" -- see build_prompt()

WORDS = list(dict.fromkeys("""
the of and to in is that for it as was with be by on not he this are or his from at which but
have an had they you were their one all we can her has there been if more when will would who
so no out up into time than only some could them other then now do any my like over such our
man me even most made after also did many before must through back years where much your way
well down should because each just those people how too little state good very make world
still own see men work long get here between both life being under never day same another
know while last might us great old year off come since against go came right used take three
small house place again found thought part number hand high water side without head always
large night point early during however home once white told upon every does got united left
number course war until away something fact though less public put think almost enough far
took yet government system better set told nothing end why called eyes find going look asked
later knew point next city business give group toward young days let room president children
light second things face given often door social order half school early case several four
best power area hour words become sense turn mind country problem service whole among others
form important money open family local body black sure field history question keep change
river story least able land matter north south music paper week known earth letter line
moment nature human green table short voice level morning report street market evening book
simple window garden winter summer stone metal engine signal bridge forest mountain valley
island ocean cloud storm season harvest village castle kitchen silver copper marble timber
canvas ladder basket bottle candle mirror pocket ribbon saddle thread velvet wagon anchor
beacon cargo harbor vessel captain compass journey lantern meadow orchard pasture shelter
circuit voltage current sensor filter carrier module buffer packet kernel thread memory
vector matrix tensor sample batch layer weight token stream buffer cache index query schema
""".split()))


# ── small helpers ────────────────────────────────────────────────────────────────────────

def percentile(values: Iterable[float], pct: float) -> float | None:
    """Linear-interpolated percentile (numpy 'linear' method). None for an empty input."""
    data = sorted(values)
    if not data:
        return None
    if len(data) == 1:
        return float(data[0])
    rank = (len(data) - 1) * (pct / 100.0)
    lo = math.floor(rank)
    hi = min(lo + 1, len(data) - 1)
    return float(data[lo] + (data[hi] - data[lo]) * (rank - lo))


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def parse_concurrency(text: str) -> list[int]:
    levels = [int(part) for part in text.split(",") if part.strip()]
    if not levels or any(level < 1 for level in levels):
        raise ValueError("concurrency levels must be positive integers")
    return levels


# ── prompts ──────────────────────────────────────────────────────────────────────────────

def _random_word(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 8)))


def build_prompt(rng: random.Random, total_words: int, dataset: str, serial: int) -> str:
    """A prompt of `total_words` whitespace-separated words whose FIRST line is unique.

    The header leads with 64 random bits so prompts diverge from the very first token and
    the prefix cache cannot reuse anything; `serial` guarantees uniqueness within a run.
    """
    header = f"{rng.getrandbits(64):016x} request {serial}"
    n_body = max(1, total_words - HEADER_WORDS)
    if dataset == "random":
        body = [_random_word(rng) for _ in range(n_body)]
    else:
        body = rng.choices(WORDS, k=n_body)
    lines = [" ".join(body[i:i + 16]) for i in range(0, n_body, 16)]
    return header + "\n" + "\n".join(lines)


def words_for_tokens(input_tokens: int, tokens_per_word: float) -> int:
    return max(HEADER_WORDS + 1, round(input_tokens / tokens_per_word))


# ── HTTP / SSE ───────────────────────────────────────────────────────────────────────────

@dataclass
class Target:
    base_url: str
    model: str
    endpoint_kind: str  # "completions" | "chat"
    api_key: str
    timeout: float

    @property
    def url(self) -> str:
        path = "/chat/completions" if self.endpoint_kind == "chat" else "/completions"
        return self.base_url.rstrip("/") + path

    def payload(self, prompt: str, max_tokens: int, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "temperature": 0}
        if self.endpoint_kind == "chat":
            body["messages"] = [{"role": "user", "content": prompt}]
        else:
            body["prompt"] = prompt
        if stream:
            body.update(stream=True, stream_options={"include_usage": True}, ignore_eos=True)
        return body

    def open(self, body: dict[str, Any]):
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Accept": "text/event-stream, application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        return urllib.request.urlopen(request, timeout=self.timeout)


def iter_sse(lines: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of an SSE stream. Skips blanks, comments, non-data fields and
    unparsable payloads; stops at `data: [DONE]`."""
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        if not data:
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def delta_text(event: dict[str, Any]) -> str:
    """Generated text carried by one stream chunk: `choices[0].text` (completions) or
    `choices[0].delta.content` (chat; reasoning deltas count too -- they are generated tokens)."""
    choices = event.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    choice = choices[0]
    text = choice.get("text")
    if isinstance(text, str) and text:
        return text
    delta = choice.get("delta")
    if isinstance(delta, dict):
        for key in ("content", "reasoning_content", "reasoning"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _describe_error(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = exc.read(300).decode("utf-8", "replace").strip().replace("\n", " ")
        except Exception:  # noqa: BLE001 - body is best-effort
            detail = ""
        return f"HTTP {exc.code} {detail}".strip()
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "timeout"
    if isinstance(exc, urllib.error.URLError):
        return f"connection error: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"


@dataclass
class RequestResult:
    ok: bool
    ttft_s: float | None = None
    e2e_s: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_reported: bool = False
    error: str | None = None


def run_request(target: Target, prompt: str, output_tokens: int, est_prompt_tokens: int) -> RequestResult:
    """One streamed request. Never raises: every failure becomes RequestResult(ok=False)."""
    start = time.perf_counter()
    deadline = start + target.timeout
    ttft: float | None = None
    deltas = 0
    usage: dict[str, Any] | None = None
    try:
        with target.open(target.payload(prompt, output_tokens, stream=True)) as response:
            for event in iter_sse(response):
                now = time.perf_counter()
                if event.get("error"):
                    return RequestResult(ok=False, error=f"stream error: {str(event['error'])[:200]}")
                if delta_text(event):
                    deltas += 1
                    if ttft is None:
                        ttft = now - start
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                if now > deadline:
                    return RequestResult(ok=False, error="timeout")
        e2e = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - a load generator must survive any request failure
        return RequestResult(ok=False, error=_describe_error(exc))

    usage = usage or {}
    completion = usage.get("completion_tokens")
    completion = int(completion) if isinstance(completion, (int, float)) and completion > 0 else deltas
    if ttft is None or completion <= 0:
        return RequestResult(ok=False, error="no tokens generated")
    prompt_tokens = usage.get("prompt_tokens")
    reported = isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0
    return RequestResult(ok=True, ttft_s=ttft, e2e_s=e2e,
                         prompt_tokens=int(prompt_tokens) if reported else est_prompt_tokens,
                         completion_tokens=completion, usage_reported=bool(reported))


def calibrate(target: Target, rng: random.Random, input_tokens: int, dataset: str) -> tuple[float, str, bool]:
    """One non-streamed max_tokens=1 request -> (tokens_per_word, note, calibrated).

    The calibration prompt is ~75 % of the final size (by first guess) so fixed overhead
    (header, BOS, chat template) is amortised the same way it will be in the real prompts.
    """
    n_words = max(50, round(0.75 * input_tokens / CALIBRATION_GUESS.get(dataset, 1.35)))
    prompt = build_prompt(rng, n_words, dataset, serial=0)
    actual_words = len(prompt.split())
    try:
        with target.open(target.payload(prompt, 1, stream=False)) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
        prompt_tokens = (body.get("usage") or {}).get("prompt_tokens") if isinstance(body, dict) else None
    except Exception as exc:  # noqa: BLE001
        return FALLBACK_TOKENS_PER_WORD, f"calibration request failed ({_describe_error(exc)})", False
    if not isinstance(prompt_tokens, (int, float)) or prompt_tokens <= 0:
        return FALLBACK_TOKENS_PER_WORD, "server reported no usage.prompt_tokens", False
    return prompt_tokens / actual_words, f"{int(prompt_tokens)} tokens for {actual_words} words", True


# ── closed-loop level ────────────────────────────────────────────────────────────────────

def run_level(target: Target, prompts: list[str], concurrency: int, output_tokens: int,
              est_prompt_tokens: int, abort_on_failures: bool = True) -> tuple[list[RequestResult], float]:
    """Keep `concurrency` requests in flight until every prompt is issued and drained.
    With abort_on_failures, stops issuing once more than half the level has failed."""
    lock = threading.Lock()
    results: list[RequestResult] = []
    state = {"next": 0, "failed": 0}
    fail_limit = len(prompts) / 2

    def worker() -> None:
        while True:
            with lock:
                if state["next"] >= len(prompts) or (abort_on_failures and state["failed"] > fail_limit):
                    return
                index = state["next"]
                state["next"] += 1
            result = run_request(target, prompts[index], output_tokens, est_prompt_tokens)
            with lock:
                results.append(result)
                if not result.ok:
                    state["failed"] += 1

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(concurrency, len(prompts)))]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, time.perf_counter() - start


def summarize(concurrency: int, results: list[RequestResult], duration: float) -> dict[str, Any]:
    good = [r for r in results if r.ok]
    duration = max(duration, 1e-9)
    prompt_sum = sum(r.prompt_tokens for r in good)
    output_sum = sum(r.completion_tokens for r in good)
    ttft_ms = [r.ttft_s * 1000 for r in good]
    e2e_s = [r.e2e_s for r in good]
    tpot_ms = [(r.e2e_s - r.ttft_s) / (r.completion_tokens - 1) * 1000 for r in good if r.completion_tokens > 1]
    return {
        "concurrency": concurrency,
        "completed": len(good),
        "failed": len(results) - len(good),
        "duration_s": round(duration, 3),
        "req_per_s": round(len(good) / duration, 4),
        "prefill_tok_s": round(prompt_sum / duration, 1),
        "output_tok_s": round(output_sum / duration, 1),
        "total_tok_s": round((prompt_sum + output_sum) / duration, 1),
        "ttft_ms_p50": _round(percentile(ttft_ms, 50), 1),
        "ttft_ms_p90": _round(percentile(ttft_ms, 90), 1),
        "ttft_ms_p99": _round(percentile(ttft_ms, 99), 1),
        "tpot_ms_p50": _round(percentile(tpot_ms, 50), 2),
        "e2e_s_p50": _round(percentile(e2e_s, 50), 3),
        "e2e_s_p90": _round(percentile(e2e_s, 90), 3),
        "e2e_s_p99": _round(percentile(e2e_s, 99), 3),
    }


COLUMNS = [("conc", "concurrency", 5), ("done", "completed", 5), ("fail", "failed", 5), ("dur s", "duration_s", 8),
           ("req/s", "req_per_s", 8), ("prefill t/s", "prefill_tok_s", 11), ("output t/s", "output_tok_s", 10),
           ("total t/s", "total_tok_s", 10), ("TTFT p50", "ttft_ms_p50", 9), ("p90", "ttft_ms_p90", 9),
           ("p99 ms", "ttft_ms_p99", 9), ("TPOT p50 ms", "tpot_ms_p50", 11), ("E2E p50", "e2e_s_p50", 8),
           ("p90", "e2e_s_p90", 8), ("p99 s", "e2e_s_p99", 8)]


def table_header() -> str:
    return " ".join(f"{title:>{width}}" for title, _, width in COLUMNS)


def table_row(level: dict[str, Any]) -> str:
    cells = []
    for _, key, width in COLUMNS:
        value = level[key]
        if value is None:
            text = "-"
        elif isinstance(value, int):
            text = str(value)
        else:
            text = f"{value:.2f}" if key in ("req_per_s", "tpot_ms_p50", "e2e_s_p50", "e2e_s_p90", "e2e_s_p99") else f"{value:.1f}"
        cells.append(f"{text:>{width}}")
    return " ".join(cells)


# ── main ─────────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Closed-loop realistic-load benchmark for an OpenAI-compatible endpoint.")
    p.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8002/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--input-tokens", type=int, default=2800)
    p.add_argument("--output-tokens", type=int, default=250)
    p.add_argument("--concurrency", default="8,16,32", help="comma-separated levels (default 8,16,32)")
    p.add_argument("--requests-per-level", type=int, default=None, help="default: 10 x concurrency, minimum 20")
    p.add_argument("--dataset", choices=("random", "words"), default="words")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timeout", type=float, default=900.0, help="per-request seconds (default 900)")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--endpoint-kind", choices=("completions", "chat"), default="completions")
    p.add_argument("--out", default=None, help="write full JSON (incl. per-request records) here")
    p.add_argument("--warmup", type=int, default=4, help="uncounted warm-up requests (default 4)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        levels = parse_concurrency(args.concurrency)
    except ValueError as exc:
        parser.error(f"--concurrency: {exc}")
    if args.input_tokens < 8 or args.output_tokens < 1:
        parser.error("--input-tokens must be >= 8 and --output-tokens >= 1")
    if args.requests_per_level is not None and args.requests_per_level < 1:
        parser.error("--requests-per-level must be >= 1")
    if args.warmup < 0 or args.timeout <= 0:
        parser.error("--warmup must be >= 0 and --timeout > 0")

    rng = random.Random(args.seed)
    target = Target(args.base_url, args.model, args.endpoint_kind, args.api_key, args.timeout)
    print(f"load_bench: {target.url} model={args.model} in={args.input_tokens} out={args.output_tokens} "
          f"dataset={args.dataset} levels={levels}", flush=True)

    tokens_per_word, note, calibrated = calibrate(target, rng, args.input_tokens, args.dataset)
    if calibrated:
        print(f"calibration: {tokens_per_word:.3f} tokens/word ({note})", flush=True)
    else:
        print(f"calibration: {note}; FALLING BACK to {FALLBACK_TOKENS_PER_WORD} tokens/word "
              f"(prompt sizes are estimates)", flush=True)
    n_words = words_for_tokens(args.input_tokens, tokens_per_word)
    est_prompt_tokens = round(n_words * tokens_per_word)
    serial = iter(range(1, 10 ** 9))

    def make_prompts(count: int) -> list[str]:
        return [build_prompt(rng, n_words, args.dataset, next(serial)) for _ in range(count)]

    if args.warmup:
        warm, warm_s = run_level(target, make_prompts(args.warmup), min(args.warmup, levels[0]),
                                 args.output_tokens, est_prompt_tokens, abort_on_failures=False)
        print(f"warmup: {sum(r.ok for r in warm)}/{len(warm)} ok in {warm_s:.1f}s", flush=True)

    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    stopped_early = False
    print(table_header(), flush=True)
    for concurrency in levels:
        count = args.requests_per_level or max(20, 10 * concurrency)
        results, duration = run_level(target, make_prompts(count), concurrency, args.output_tokens, est_prompt_tokens)
        level = summarize(concurrency, results, duration)
        summaries.append(level)
        records.append({"concurrency": concurrency, "requests": [asdict(r) for r in results]})
        print(table_row(level), flush=True)
        if level["failed"] > count / 2:
            errors = sorted({r.error for r in results if r.error})[:3]
            print(f"more than half of the requests failed at concurrency {concurrency}; stopping the sweep. "
                  f"errors: {'; '.join(errors)}", flush=True)
            stopped_early = True
            break
        good = [r for r in results if r.ok]
        if good and not all(r.usage_reported for r in good):
            print("  note: server did not report usage on every stream; token counts are partly estimated", flush=True)

    done = [lv for lv in summaries if lv["completed"] > 0]
    best = max(done, key=lambda lv: lv["req_per_s"]) if done else None
    result = {
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "tokens_per_word": round(tokens_per_word, 4),
        "levels": summaries,
        "best": {k: best[k] for k in ("concurrency", "req_per_s", "total_tok_s")} if best else None,
    }
    if args.out:
        full = dict(result, base_url=args.base_url, model=args.model, dataset=args.dataset, seed=args.seed,
                    endpoint_kind=args.endpoint_kind, calibrated=calibrated, calibration_note=note,
                    words_per_prompt=n_words, stopped_early=stopped_early, requests=records)
        try:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(full, handle, indent=2)
        except OSError as exc:
            print(f"could not write {args.out}: {exc}", file=sys.stderr)
    if best:
        print(f"best: concurrency {best['concurrency']} -> {best['req_per_s']:.2f} req/s, "
              f"{best['total_tok_s']:.0f} total tok/s", flush=True)
    print("LOAD_RESULT " + json.dumps(result, separators=(",", ":")), flush=True)
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
