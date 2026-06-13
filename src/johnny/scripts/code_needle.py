#!/usr/bin/env python3
"""
Code Needle benchmark runner.

Reads a corpus.json produced by build_corpus.py, sends 16 positional-recall
queries to a vLLM chat-completions endpoint, scores each response against
ground truth via difflib alignment, and reports pass/fail per target.

Usage:
  code_needle.py --corpus corpus.json --base-url http://localhost:8000/v1 \\
    --model gemma-4-26b --out results.json [--normalized|--strict]
"""
import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("ERROR: pip install openai in ~/vllm-bench-venv")


PROMPT_TEMPLATE = (
    "{file_text}\n\n"
    "Reproduce verbatim the first 20 lines following the opening brace "
    "(or colon, for Python) of the function named `{name}`. "
    "Output ONLY those 20 lines inside a single ```code``` block. "
    "Do not write any commentary, explanation, or extra text outside the block."
)


CODE_FENCE_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)\n```", re.DOTALL)


def extract_code(response: str) -> list:
    m = CODE_FENCE_RE.search(response)
    body = m.group(1) if m else response.strip()
    return body.splitlines()


def normalize_line(line: str) -> str:
    """Strip leading/trailing whitespace and collapse internal whitespace runs.

    Use this for the `--normalized` scoring mode to be fair to chat-tuned
    models that re-indent reflexively.
    """
    return re.sub(r"\s+", " ", line.strip())


def score_target(predicted: list, expected: list, mode: str) -> dict:
    """Align predicted lines against expected via difflib, classify each line.

    Returns counts of matched / missing / hallucinated / extra_correct.
    """
    if mode == "normalized":
        pred_keys = [normalize_line(l) for l in predicted]
        exp_keys = [normalize_line(l) for l in expected]
    else:
        pred_keys = predicted
        exp_keys = expected

    sm = difflib.SequenceMatcher(a=exp_keys, b=pred_keys, autojunk=False)
    matched = 0
    missing = 0
    hallucinated = 0
    matched_expected_idx = set()
    matched_pred_idx = set()

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            matched += (i2 - i1)
            matched_expected_idx.update(range(i1, i2))
            matched_pred_idx.update(range(j1, j2))
        elif tag == "delete":
            missing += (i2 - i1)
        elif tag == "insert":
            # Predicted line not present in expected (within the 20-line window)
            hallucinated += (j2 - j1)
        elif tag == "replace":
            missing += (i2 - i1)
            hallucinated += (j2 - j1)

    extra_correct = 0  # lines beyond the 20-line window that are still correct
    if len(predicted) > 20:
        # Count tail predicted lines that look like real continuation of body.
        # We don't have ground truth for them; flag them informationally only.
        extra_correct = len(predicted) - 20

    return {
        "matched": matched,
        "missing": missing,
        "hallucinated": hallucinated,
        "extra_correct": extra_correct,
        "passed": matched >= 8,
    }


# ANSI colors
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["strict", "normalized"], default="normalized")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--disable-thinking", action="store_true",
                    help="Pass chat_template_kwargs={enable_thinking: false} to suppress "
                         "<think> blocks on Qwen3 family models. Required when launcher "
                         "uses --reasoning-parser qwen3, else content is null and the "
                         "bench scores 0/16.")
    args = ap.parse_args()

    extra_body = None
    if args.disable_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        print("Disabling thinking via chat_template_kwargs.enable_thinking=False", file=sys.stderr, flush=True)

    corpus = json.loads(Path(args.corpus).expanduser().read_text())
    file_text = corpus["file_text"]
    targets = corpus["targets"]

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)

    results = []
    print(f"Running Code Needle on {args.model} — {len(targets)} targets, "
          f"{corpus['token_count']:,} corpus tokens, mode={args.mode}\n",
          file=sys.stderr)

    for i, target in enumerate(targets, 1):
        prompt = PROMPT_TEMPLATE.format(file_text=file_text, name=target["name"])
        t0 = time.time()
        try:
            kwargs = dict(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                top_p=1,
                max_tokens=args.max_tokens,
            )
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            output = resp.choices[0].message.content or ""
            err = None
        except Exception as e:
            output = ""
            err = str(e)
        elapsed = time.time() - t0

        predicted = extract_code(output)
        score = score_target(predicted, target["expected_lines"], args.mode)
        score["name"] = target["name"]
        score["position_pct"] = target["position_pct"]
        score["start_line"] = target["start_line"]
        score["elapsed_s"] = round(elapsed, 1)
        score["error"] = err
        results.append(score)

        flag = f"{GREEN}PASS{RESET}" if score["passed"] else f"{RED}FAIL{RESET}"
        print(f"[{i:2d}/{len(targets)}] {flag}  {target['name']:30s} "
              f"pos={target['position_pct']:5.1f}%  "
              f"matched={score['matched']:2d}  miss={score['missing']:2d}  "
              f"halluc={score['hallucinated']:2d}  ({elapsed:.1f}s)"
              + (f"  ERR: {err}" if err else ""),
              file=sys.stderr)

    # Aggregate
    agg = {
        "pass": sum(1 for r in results if r["passed"]),
        "fail": sum(1 for r in results if not r["passed"]),
        "matched": sum(r["matched"] for r in results),
        "missing": sum(r["missing"] for r in results),
        "hallucinated": sum(r["hallucinated"] for r in results),
        "extra_correct": sum(r["extra_correct"] for r in results),
    }
    # Position-bias breakdown
    thirds = {"first": [], "middle": [], "last": []}
    for r in results:
        b = "first" if r["position_pct"] < 33.3 else ("middle" if r["position_pct"] < 66.6 else "last")
        thirds[b].append(r["passed"])
    position_bias = {k: f"{sum(v)}/{len(v)}" for k, v in thirds.items() if v}

    out_data = {
        "model": args.model,
        "corpus_tokens": corpus["token_count"],
        "mode": args.mode,
        "targets": results,
        "aggregate": agg,
        "position_bias": position_bias,
    }
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Code Needle — {args.model} ({corpus['token_count']:,} tokens)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"PASS: {agg['pass']} / {len(results)}", file=sys.stderr)
    print(f"Lines: {agg['matched']} matched, {agg['missing']} missing, "
          f"{agg['hallucinated']} hallucinated, {agg['extra_correct']} extra",
          file=sys.stderr)
    print(f"Position: " + "  ".join(f"{k}={v}" for k, v in position_bias.items()),
          file=sys.stderr)
    print(f"\nFull results: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
