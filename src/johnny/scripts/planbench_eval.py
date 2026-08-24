#!/usr/bin/env python3
"""PlanBench plan-generation eval (tasksource/planbench, task_1_plan_generation).

Why this suite exists: arc/icl/needle/humaneval are single-shot *knowledge/coding* probes and
automationbench is a 50-step *tool loop* — neither isolates PLANNING. PlanBench does: each
instance is a short, self-contained PDDL problem (Blocksworld family) presented one-shot, so a
model must produce a correct action sequence with no tools, no exploration, and ~1-2K of context.
That separates "can it plan" from "can it manage a long agentic loop / big context" — the exact
confound that made gemma-4-26B's automationbench result (0/30, but 31 tool calls/task and 13/30
hitting the step cap) ambiguous.

Scoring is EXACT ACTION-SEQUENCE MATCH against the reference plan, which is a strict lower bound:
a different-but-valid plan scores 0 (real PlanBench uses the VAL plan validator to accept those).
Report it as such — it is a comparison between models on identical instances, not an absolute
planning score. `plan_prefix_pct` (fraction of leading actions that match) is the partial-credit
companion, so a model that starts right and drifts is distinguishable from one that never lands.

Default domains are the plain-English blocksworld ones; the obfuscated/mystery domains
(deliberately renamed predicates, to defeat memorization) are available via --domains but the
NL->action normalizer here only knows the blocksworld verb forms.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Blocksworld surface forms -> canonical (action, *args). The one-shot example in every
# prompt uses these exact phrasings, so models mimic them.
_PATTERNS = [
    (re.compile(r"\bunstack\s+the\s+(\w+)\s+block\s+from\s+on\s+top\s+of\s+the\s+(\w+)\s+block", re.I), "unstack", 2),
    (re.compile(r"\bstack\s+the\s+(\w+)\s+block\s+on\s+top\s+of\s+the\s+(\w+)\s+block", re.I), "stack", 2),
    (re.compile(r"\bpick\s*[- ]?up\s+the\s+(\w+)\s+block", re.I), "pick-up", 1),
    (re.compile(r"\bput\s*[- ]?down\s+the\s+(\w+)\s+block", re.I), "put-down", 1),
]


def actions_from_text(text: str) -> list[tuple]:
    """Extract a canonical action sequence from a model's natural-language plan."""
    if "[PLAN]" in text:                       # honor the fenced form when present
        text = text.split("[PLAN]", 1)[1]
    text = text.split("[PLAN END]", 1)[0]
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for rx, name, n in _PATTERNS:
            m = rx.search(line)
            if m:
                out.append((name, *(g.lower() for g in m.groups()[:n])))
                break
    return out


def actions_from_pddl(plan: str) -> list[tuple]:
    """Reference plans are PDDL: '(pick-up red)\\n(stack red orange)'."""
    out = []
    for m in re.finditer(r"\(([^)]+)\)", plan or ""):
        parts = m.group(1).split()
        if parts:
            out.append((parts[0].lower(), *(p.lower() for p in parts[1:])))
    return out


def prefix_match(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--domains", default="blocksworld,blocksworld_3")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--disable-thinking", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("planbench: missing dep 'datasets'", file=sys.stderr)
        return 2
    try:
        from openai import OpenAI
    except ImportError:
        print("planbench: missing dep 'openai'", file=sys.stderr)
        return 2

    want = {d.strip() for d in a.domains.split(",") if d.strip()}
    ds = load_dataset("tasksource/planbench", "task_1_plan_generation", split="train")
    rows = [r for r in ds if r["domain"] in want][: a.limit]
    if not rows:
        print(f"planbench: no instances for domains={sorted(want)}", file=sys.stderr)
        return 2

    client = OpenAI(base_url=a.base_url, api_key="local", timeout=a.timeout)
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if a.disable_thinking else {}

    def one(row):
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=a.model, messages=[{"role": "user", "content": row["query"]}],
                max_tokens=a.max_tokens, temperature=0, extra_body=extra)
            text = r.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 — one bad instance must not kill the run
            return {"instance_id": row["instance_id"], "domain": row["domain"], "error": str(e)[:200],
                    "exact": False, "prefix": 0, "gt_len": 0, "s": round(time.time() - t0, 1)}
        got = actions_from_text(text)
        gt = actions_from_pddl(row["ground_truth_plan"])
        return {"instance_id": row["instance_id"], "domain": row["domain"],
                "exact": got == gt and bool(gt), "prefix": prefix_match(got, gt), "gt_len": len(gt),
                "got_len": len(got), "s": round(time.time() - t0, 1)}

    results = []
    with ThreadPoolExecutor(a.concurrency) as ex:
        for i, res in enumerate(ex.map(one, rows), 1):
            results.append(res)
            if i % 5 == 0 or i == len(rows):
                ok = sum(1 for x in results if x["exact"])
                print(f"  {i}/{len(rows)}  exact={100*ok/len(results):.1f}%", flush=True)

    n = len(results)
    exact = sum(1 for r in results if r["exact"])
    gt_tot = sum(r["gt_len"] for r in results) or 1
    pref = sum(r["prefix"] for r in results)
    errs = sum(1 for r in results if r.get("error"))
    summary = {"exact_pct": round(100 * exact / n, 2), "exact": exact, "total": n,
               "plan_prefix_pct": round(100 * pref / gt_tot, 2), "errors": errs,
               "domains": sorted(want), "task": "task_1_plan_generation"}
    if a.out:
        json.dump({"summary": summary, "results": results}, open(a.out, "w"), indent=1)
    print(f"PlanBench exact-plan {summary['exact_pct']}% ({exact}/{n}) · "
          f"prefix {summary['plan_prefix_pct']}% · errors {errs}")
    print("PLANBENCH_JSON " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
