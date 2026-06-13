#!/usr/bin/env python3
"""
ARC Challenge eval for instruction-tuned models via chat completions.

Bypasses lm-eval's broken stop-sequence truncation for CoT models.
Loads the ARC-Challenge test split directly from HuggingFace datasets,
queries the model with CoT, extracts the answer letter, and reports accuracy.

Usage:
  arc_eval.py [--base-url URL] [--model NAME] [--max-tokens N]
              [--concurrency N] [--limit N] [--out PATH]

Confirmed working: Gemma-4-26B-A4B-it FP8-Dynamic → 95.05% (1114/1172), 2026-05-27
"""
import argparse, json, re, sys, time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")

from concurrent.futures import ThreadPoolExecutor, as_completed

ANSWER_RE = re.compile(r'best answer is\s*[:\[]?\s*\**\s*([A-D])', re.IGNORECASE)
FALLBACK_RE = [
    re.compile(r'answer[:\s]+\**([A-D])\**', re.IGNORECASE),
    re.compile(r'\b([A-D])\s*[.)]\s*$', re.MULTILINE),
    re.compile(r'^\s*\**([A-D])\**\s*$', re.MULTILINE),
]

SYSTEM = (
    "You are answering a multiple-choice science exam. "
    "After reasoning through the problem, end your response with exactly: "
    "\"The best answer is X\" where X is A, B, C, or D."
)

def fmt_question(doc):
    choices = doc['choices']
    labels = choices['label']
    texts  = choices['text']
    opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return f"Question: {doc['question']}\n{opts}"

def extract_answer(text):
    m = ANSWER_RE.search(text)
    if m: return m.group(1).upper()
    for pat in FALLBACK_RE:
        m = pat.search(text)
        if m: return m.group(1).upper()
    return None

def eval_one(client, model, doc, max_tokens, extra_body=None):
    prompt = fmt_question(doc)
    key = doc['answerKey']
    target = {'1':'A','2':'B','3':'C','4':'D'}.get(key, key).upper()
    try:
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
    except Exception as e:
        return {"target": target, "extracted": None, "text": f"ERROR: {e}", "error": True}
    extracted = extract_answer(text)
    return {"target": target, "extracted": extracted, "text": text, "error": False}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url",    default="http://localhost:8000/v1")
    ap.add_argument("--model",       default="gemma-4-26b")
    ap.add_argument("--max-tokens",  type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit",       type=int, default=None, help="Subset for testing")
    ap.add_argument("--out",         default="~/vllm-bench-results/arc_scored.jsonl")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="Pass chat_template_kwargs={enable_thinking: false} to suppress "
                         "<think> blocks on Qwen3 family models (Qwen3, Qwen3.5, Qwen3.6). "
                         "Required when launcher uses --reasoning-parser qwen3, else content "
                         "is null and the bench scores 0%.")
    args = ap.parse_args()

    extra_body = None
    if args.disable_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        print("Disabling thinking via chat_template_kwargs.enable_thinking=False", flush=True)

    print("Loading ARC-Challenge test split...", flush=True)
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    docs = list(ds)
    if args.limit:
        docs = docs[:args.limit]
    print(f"Evaluating {len(docs)} questions  concurrency={args.concurrency}  max_tokens={args.max_tokens}", flush=True)

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=60)
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = [None] * len(docs)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(eval_one, client, args.model, doc, args.max_tokens, extra_body): i
                for i, doc in enumerate(docs)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if done % 100 == 0 or done == len(docs):
                correct_so_far = sum(1 for r in results[:done] if r and r['extracted'] == r['target'])
                print(f"  {done}/{len(docs)}  acc={correct_so_far/done*100:.1f}%", flush=True)

    elapsed = time.time() - t0
    correct = sum(1 for r in results if r and r['extracted'] == r['target'])
    no_ext  = sum(1 for r in results if r and r['extracted'] is None)
    errors  = sum(1 for r in results if r and r.get('error'))
    total   = len(results)
    acc     = correct / total

    print(f"\n{'='*50}")
    print(f"ARC-Challenge — {args.model}")
    print(f"{'='*50}")
    print(f"Accuracy:      {correct}/{total} = {acc*100:.2f}%")
    print(f"No extraction: {no_ext}  ({no_ext/total*100:.1f}%)")
    print(f"API errors:    {errors}")
    print(f"Elapsed:       {elapsed:.0f}s")

    with open(out_path, 'w') as f:
        for doc, r in zip(docs, results):
            f.write(json.dumps({"id": doc.get('id',''), **r}) + "\n")
    print(f"Samples saved: {out_path}")

if __name__ == "__main__":
    main()
