#!/usr/bin/env python3
"""
In-Context-Learning probe: four few-shot pattern-completion tasks with a
verified closed-form rule each, checking whether the model actually induces
the rule from examples rather than pattern-matching surface tokens.

Manual run tonight (2026-08-04) against several local models found this a
good, real discriminator: small/unquantized models scored 0/8, quantized MoE
models 2-4/8, frontier cloud models 7-8/8 on the same task shape.

Categories (fixed few-shot block + closed-form rule, 4 generated query cases
each, 16 total):
  a2_cipher        — shift each letter of a 4-letter word by its 1-indexed
                      position; +1 direction if vowel, -1 if consonant.
  b2_arithmetic     — (a+b) * (c-d) over four small integers.
  c2_structural     — 5-letter word -> [c4, c3, upper(c2), c1, c5] (0-indexed
                      char positions 3,2,1,0,4).
  d2_compositional  — pick the longest of 3 comma-separated words (ties break
                      to first occurrence), reverse it, capitalize first letter.

Grading is strict: null/empty `message.content` (a reasoning model that
ignored --disable-thinking, or ran out of budget) is an automatic fail. Do
NOT fall back to reasoning_content and search it for a coincidental
substring — that produced a real false positive during manual testing (a
base-completion model got credit for a right answer that only appeared by
chance in unrelated hallucinated continuation text).

Usage:
  icl_eval.py --base-url http://localhost:8000/v1 --model NAME --out results.json
"""
import argparse
import json
import re
import random
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")


# Fixed seed: every run (any model) sees the same 16 generated cases, so
# scores are comparable across models.
SEED = 20260804

PROMPT_HEADER = (
    "Complete the pattern shown by the examples below. Reply with ONLY the "
    "missing answer that goes after the arrow — no explanation, no "
    "punctuation, just the answer.\n\n"
)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# --------------------------------------------------------------------------- rules
def a2_cipher(word: str) -> str:
    """Shift each letter by its 1-indexed position; +1 if vowel, -1 if consonant, mod 26."""
    out = []
    for i, ch in enumerate(word, start=1):
        direction = 1 if ch in "aeiou" else -1
        shift = i * direction
        out.append(chr((ord(ch) - ord("a") + shift) % 26 + ord("a")))
    return "".join(out)


def b2_arith(a: int, b: int, c: int, d: int) -> str:
    return str((a + b) * (c - d))


def c2_struct(word: str) -> str:
    return word[3] + word[2] + word[1].upper() + word[0] + word[4]


def d2_comp(words: list[str]) -> str:
    longest = max(words, key=len)  # first occurrence wins ties (Python max() is stable)
    rev = longest[::-1]
    return rev[0].upper() + rev[1:]


# --------------------------------------------------------------------------- case generation
FOUR_LETTER_WORDS = [
    "wolf", "tree", "fish", "star", "moon", "gold", "iron", "rock", "sand",
    "leaf", "rain", "snow", "fire", "wind", "salt", "milk", "corn", "barn",
    "pond", "hill", "cave", "reef", "kelp", "vine",
]

FIVE_LETTER_WORDS = [
    "table", "chalk", "stone", "brave", "cabin", "dance", "eagle", "flame",
    "hound", "mango", "north", "ocean", "piano", "quilt", "river", "sugar",
    "tiger", "unity", "vapor", "wheat",
]

D2_WORD_POOL = {
    3: ["fox", "owl", "cow", "pig", "bee", "bat"],
    4: ["wolf", "deer", "hare", "toad", "crow"],
    5: ["otter", "camel", "horse", "eagle"],
    6: ["rabbit", "monkey", "donkey", "turtle", "weasel"],
    7: ["giraffe", "dolphin", "leopard"],
}

B2_FEWSHOT_CASES = {(2, 3, 5, 2), (4, 1, 6, 2), (2, 5, 3, 1), (1, 6, 8, 3)}


def _gen_a2_cases(rng: random.Random, n: int = 4) -> list[str]:
    fewshot = {"frog", "wave", "lamp"}
    pool = [w for w in FOUR_LETTER_WORDS if w not in fewshot]
    return rng.sample(pool, n)


def _gen_b2_cases(rng: random.Random, n: int = 4) -> list[tuple]:
    seen = set(B2_FEWSHOT_CASES)
    cases = []
    while len(cases) < n:
        a, b, c, d = (rng.randint(1, 10) for _ in range(4))
        if c <= d:  # keep results unambiguously positive
            continue
        key = (a, b, c, d)
        if key in seen:
            continue
        seen.add(key)
        cases.append(key)
    return cases


def _gen_c2_cases(rng: random.Random, n: int = 4) -> list[str]:
    fewshot = {"apple", "chair", "grape"}
    pool = [w for w in FIVE_LETTER_WORDS if w not in fewshot]
    return rng.sample(pool, n)


def _gen_d2_cases(rng: random.Random, n: int = 4) -> list[list[str]]:
    all_words = [w for ws in D2_WORD_POOL.values() for w in ws]
    seen = set()
    cases = []
    while len(cases) < n - 1:
        triple = rng.sample(all_words, 3)
        key = tuple(triple)
        if key in seen:
            continue
        seen.add(key)
        cases.append(triple)
    # One deliberate, genuine length-tie case: two words from the longest
    # length bucket (so they truly tie for the max) plus a shorter filler —
    # this exercises the first-occurrence tiebreak, the trickiest part of the rule.
    tie_len = max(D2_WORD_POOL)
    w1, w2 = rng.sample(D2_WORD_POOL[tie_len], 2)
    filler_pool = [w for length, ws in D2_WORD_POOL.items() if length != tie_len for w in ws]
    w3 = rng.choice(filler_pool)
    cases.append([w1, w3, w2])
    return cases


CATEGORIES = {
    "a2_cipher": {
        "label": "A2 conditional cipher",
        "fewshot": [("frog", "eprc"), ("wave", "vcsi"), ("lamp", "kcjl")],
        "gen_cases": _gen_a2_cases,
        "format_input": lambda w: w,
        "rule": a2_cipher,
    },
    "b2_arithmetic": {
        "label": "B2 arithmetic",
        "fewshot": [("2,3,5,2", "15"), ("4,1,6,2", "20"), ("2,5,3,1", "14"), ("1,6,8,3", "35")],
        "gen_cases": _gen_b2_cases,
        "format_input": lambda t: ",".join(str(x) for x in t),
        "rule": lambda t: b2_arith(*t),
    },
    "c2_structural": {
        "label": "C2 structural",
        "fewshot": [("apple", "lpPae"), ("chair", "iaHcr"), ("grape", "paRge")],
        "gen_cases": _gen_c2_cases,
        "format_input": lambda w: w,
        "rule": c2_struct,
    },
    "d2_compositional": {
        "label": "D2 compositional",
        "fewshot": [("cat,mouse,dog", "Esuom"), ("lion,ant,zebra", "Arbez"), ("sun,moon,star", "Noom")],
        "gen_cases": _gen_d2_cases,
        "format_input": lambda ws: ",".join(ws),
        "rule": d2_comp,
    },
}


def build_cases(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    cases = []
    for key, spec in CATEGORIES.items():
        for raw in spec["gen_cases"](rng):
            cases.append({
                "category": key,
                "label": spec["label"],
                "input": spec["format_input"](raw),
                "expected": spec["rule"](raw),
                "few_shot": spec["fewshot"],
            })
    return cases


def build_prompt(case: dict) -> str:
    fewshot_block = "\n".join(f"{inp} -> {out}" for inp, out in case["few_shot"])
    return f"{PROMPT_HEADER}{fewshot_block}\n{case['input']} -> "


# --------------------------------------------------------------------------- eval
# ANSI colors
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def eval_one(client, model, case, max_tokens, extra_body=None):
    prompt = build_prompt(case)
    try:
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        err = None
    except Exception as e:
        content = None
        err = str(e)

    # Strict: null/empty content is an automatic fail. No reasoning_content
    # fallback — see module docstring for why.
    extracted = None
    if content:
        m = TOKEN_RE.search(content)
        extracted = m.group(0) if m else None

    return {
        "category": case["category"],
        "label": case["label"],
        "input": case["input"],
        "expected": case["expected"],
        "extracted": extracted,
        "passed": extracted == case["expected"],
        "error": err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None, help="Subset for testing")
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

    cases = build_cases()
    if args.limit:
        cases = cases[:args.limit]

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)

    results = []
    print(f"Running ICL probe on {args.model} — {len(cases)} cases across "
          f"{len(CATEGORIES)} categories\n", file=sys.stderr)

    for i, case in enumerate(cases, 1):
        t0 = time.time()
        r = eval_one(client, args.model, case, args.max_tokens, extra_body)
        r["elapsed_s"] = round(time.time() - t0, 1)
        results.append(r)

        flag = f"{GREEN}PASS{RESET}" if r["passed"] else f"{RED}FAIL{RESET}"
        print(f"[{i:2d}/{len(cases)}] {flag}  {r['category']:18s} {r['input']:20s} "
              f"expected={r['expected']:8s} got={r['extracted'] or '-':8s} ({r['elapsed_s']:.1f}s)"
              + (f"  ERR: {r['error']}" if r["error"] else ""),
              file=sys.stderr)

    agg = {
        "pass": sum(1 for r in results if r["passed"]),
        "fail": sum(1 for r in results if not r["passed"]),
    }
    breakdown = []
    for key, spec in CATEGORIES.items():
        sub = [r for r in results if r["category"] == key]
        if not sub:
            continue
        breakdown.append({
            "category": key,
            "label": spec["label"],
            "pass": sum(1 for r in sub if r["passed"]),
            "total": len(sub),
            "score": f"{sum(1 for r in sub if r['passed'])}/{len(sub)}",
        })

    out_data = {
        "model": args.model,
        "cases": results,
        "aggregate": agg,
        "category_breakdown": breakdown,
    }
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"ICL probe — {args.model}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"PASS: {agg['pass']} / {len(results)}", file=sys.stderr)
    print("By category: " + "  ".join(f"{b['category']}={b['score']}" for b in breakdown), file=sys.stderr)
    print(f"\nFull results: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
