#!/usr/bin/env python3
"""
hardcode — 20 medium/hard one-shot coding tasks scored by hidden tests.

What it measures: can the model write a complete, correct function/class from a
prose spec in ONE completion (temperature 0)? Tasks (hardcode_tasks.py) cover parsing,
graphs, DP, data structures and stateful classes; each answer is executed against
hidden asserts in a subprocess, and a task passes iff that process exits 0. It sits
above HumanEval in difficulty, where current local seats all saturate (~95 %).

Reading the score: n=20, so one task is 5 points. Treat +/-2 tasks as noise —
18/20 vs 16/20 is not a ranking, 18/20 vs 11/20 is. Compare the failed-id lists.

Thinking mode: --thinking sends enable_thinking=true and raises the default budget to
20000 tokens per task. Reasoning models can burn most of that on every task (hundreds
of thousands of tokens per run, minutes per request) — check mean_completion_tokens
before comparing a thinking run with a non-thinking one.

The candidate code is model output and is executed locally, unsandboxed beyond a
temp cwd and a timeout. Run it where you would run any other model-written code.

Usage:
  hardcode_eval.py --base-url http://127.0.0.1:8124/v1 --model coder
                   [--concurrency 4] [--limit N] [--max-tokens N] [--thinking]
                   [--timeout 1800] [--test-timeout 60] [--out PATH] [--api-key KEY]
  hardcode_eval.py --self-test      # no network: references vs their hidden tests

The last stdout line is always `HARDCODE_RESULT {json}` (see summarize()).
Exit code: 0 when the run completed (failures included), 2 on usage/self-test failure.
"""
from __future__ import annotations

import argparse, json, re, subprocess, sys, tempfile, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardcode_tasks import TASKS  # noqa: E402

RESULT_PREFIX = "HARDCODE_RESULT "
PY_FENCE_RE = re.compile(r"```(?:python|py)[ \t]*\r?\n(.*?)```", re.S | re.I)
ANY_FENCE_RE = re.compile(r"```[^\n`]*\r?\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """Longest ```python block, else longest fenced block of any kind, else the raw text."""
    for pat in (PY_FENCE_RE, ANY_FENCE_RE):
        blocks = pat.findall(text)
        if blocks:
            return max(blocks, key=len)
    return text


def run_tests(candidate: str, tests: str, timeout: float = 60) -> tuple[bool, str]:
    """Run candidate + hidden tests in a subprocess. Returns (passed, last error line)."""
    program = candidate + "\n__SRC__=" + repr(candidate) + "\n" + tests
    with tempfile.TemporaryDirectory(prefix="hardcode-") as tmp:
        script = Path(tmp) / "candidate.py"
        script.write_text(program, encoding="utf-8")
        try:
            pr = subprocess.run([sys.executable, str(script)], cwd=tmp, capture_output=True,
                                text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False, f"test timeout after {timeout:g}s"
    if pr.returncode == 0:
        return True, ""
    lines = pr.stderr.strip().splitlines()
    return False, (lines[-1] if lines else f"exit code {pr.returncode}")[:200]


def chat(base_url: str, api_key: str, body: dict, timeout: float) -> tuple[str, int]:
    """One chat completion. Returns (content, completion_tokens); raises on any failure."""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace").strip().replace("\n", " ")[:200]
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None
    text = data["choices"][0]["message"].get("content") or ""
    return text, int((data.get("usage") or {}).get("completion_tokens") or 0)


def eval_task(task: dict, args: argparse.Namespace) -> dict:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": args.thinking},
    }
    t0 = time.time()
    try:
        text, tokens = chat(args.base_url, args.api_key, body, args.timeout)
    except Exception as e:  # network/HTTP/shape errors fail the task, never the run
        return {"id": task["id"], "ok": False, "secs": round(time.time() - t0, 1),
                "completion_tokens": 0, "error": f"api: {type(e).__name__}: {e}"[:200],
                "api_error": True, "answer": ""}
    ok, err = run_tests(extract_code(text), task["tests"], args.test_timeout)
    return {"id": task["id"], "ok": ok, "secs": round(time.time() - t0, 1),
            "completion_tokens": tokens, "error": err, "api_error": False, "answer": text}


def summarize(results: list[dict], thinking: bool) -> dict:
    """The HARDCODE_RESULT payload — johnny bench parses this, keep the keys stable."""
    total = len(results)
    passed = sum(r["ok"] for r in results)
    answered = [r["completion_tokens"] for r in results if not r["api_error"]]
    return {
        "passed": passed,
        "total": total,
        "pass_rate_pct": round(100.0 * passed / total, 2) if total else 0.0,
        "failed": sorted(r["id"] for r in results if not r["ok"]),
        "api_errors": sum(r["api_error"] for r in results),
        "mean_completion_tokens": round(sum(answered) / len(answered)) if answered else 0,
        "thinking": thinking,
    }


def self_test(test_timeout: float = 60) -> list[str]:
    """Run every reference solution against its hidden tests. Returns the failing ids."""
    bad = []
    for task in TASKS:
        ok, err = run_tests(task["reference"], task["tests"], test_timeout)
        print(f"{task['id']}  {'PASS' if ok else 'FAIL'}  {err}".rstrip(), flush=True)
        if not ok:
            bad.append(task["id"])
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="20 hard one-shot coding tasks, scored by hidden tests.")
    ap.add_argument("--base-url", help="OpenAI-compatible base, e.g. http://127.0.0.1:8124/v1")
    ap.add_argument("--model")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N tasks only")
    ap.add_argument("--max-tokens", type=int, default=None, help="default 2048, or 20000 with --thinking")
    ap.add_argument("--thinking", action="store_true", help="send enable_thinking=true (default false)")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds per request")
    ap.add_argument("--test-timeout", type=float, default=60, help="seconds per task test run")
    ap.add_argument("--out", default=None, help="write per-task detail JSON here")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--self-test", action="store_true", help="no network: check the reference solutions")
    args = ap.parse_args(argv)

    if args.self_test:
        bad = self_test(args.test_timeout)
        print(f"self-test: {len(TASKS) - len(bad)}/{len(TASKS)} references pass" + (f" | FAILED: {', '.join(bad)}" if bad else ""))
        return 2 if bad else 0
    if not args.base_url or not args.model:
        ap.error("--base-url and --model are required (unless --self-test)")
    if args.concurrency < 1 or (args.limit is not None and args.limit < 1):
        ap.error("--concurrency and --limit must be >= 1")
    if args.max_tokens is None:
        args.max_tokens = 20000 if args.thinking else 2048

    tasks = TASKS[: args.limit] if args.limit else TASKS
    print(f"hardcode: {len(tasks)} tasks | model={args.model} | thinking={'on' if args.thinking else 'off'} "
          f"| max_tokens={args.max_tokens} | concurrency={args.concurrency}", flush=True)
    t0 = time.time()
    by_id: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(eval_task, t, args) for t in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            by_id[r["id"]] = r
            print(f"[{n:>2}/{len(tasks)}] {r['id']}  {'PASS' if r['ok'] else 'FAIL'}  {r['secs']:>7.1f}s  "
                  f"{r['completion_tokens']:>6} tok  {r['error'][:80]}".rstrip(), flush=True)
    results = [by_id[t["id"]] for t in tasks]
    summary = summarize(results, args.thinking)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": args.model, "base_url": args.base_url,
                                   "max_tokens": args.max_tokens, "summary": summary,
                                   "tasks": {r["id"]: r for r in results}}, indent=1))

    print(f"\n{args.model}: {summary['passed']}/{summary['total']} passed ({summary['pass_rate_pct']:.2f}%) "
          f"in {time.time() - t0:.0f}s | mean {summary['mean_completion_tokens']} completion tok "
          f"| api errors: {summary['api_errors']}")
    for r in results:
        if not r["ok"]:
            print(f"  {r['id']}: {r['error'][:100]}")
    if args.out:
        print(f"detail: {args.out}")
    print(RESULT_PREFIX + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
