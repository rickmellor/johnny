#!/usr/bin/env python3
"""
Re-score lm-eval HumanEval samples for chat-completion models.

lm-eval's bundled HumanEval filters (`create_test`, `build_predictions_instruct`)
are designed for raw-completion mode where the model continues a prompt. When
used against chat completions on an instruction-tuned model the model wraps its
output in markdown fences (```python ... ```) and the bundled filter produces
garbage — silently giving pass@1 = 0 even when the model wrote correct code.

This script reads the `samples_humaneval_*.jsonl` log file produced by
`--log_samples`, extracts code from markdown fences, runs the official
HumanEval `check()` tests, and reports the real pass@1.

Usage:
  python3 humaneval_chat_score.py <samples.jsonl>

Or after a run finishes:
  python3 humaneval_chat_score.py ~/vllm-bench-results/<run>/<model>/samples_humaneval_*.jsonl

Prerequisites:
  - The lm-eval run MUST have been invoked with --log_samples.
  - --gen_kwargs "max_gen_toks=2048,until=[]" is recommended so long problems
    don't truncate mid-function. (Default 1024 + until=[\\nclass,\\ndef,...]
    causes ~2 truncation failures out of 164 on Gemma-4.)
"""
import json
import re
import subprocess
import sys
import tempfile
import os
import glob


def extract_code(raw: str) -> str:
    """Pull Python code out of a chat-completion response.

    Strategies in order:
      1. Largest ```python...``` fenced block (most complete function).
      2. Largest unlabeled ```...``` fenced block.
      3. Truncated open fence with no closer — strip the opener and use rest.
      4. Raw response as-is (no fences at all).
    """
    matches = re.findall(r"```(?:python|py)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if matches:
        return max(matches, key=len)
    m = re.match(r"\s*```(?:python|py)?\s*\n(.*)", raw, re.DOTALL)
    if m:
        return m.group(1).rstrip("`").rstrip()
    return raw.strip()


def run_test(code: str, test_code: str, entry_point: str, timeout: int = 10) -> bool:
    """Append the official HumanEval check() and return True iff all asserts pass."""
    full = code + "\n" + test_code + f"\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        r = subprocess.run(
            ["python3", path], timeout=timeout, capture_output=True, text=True
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    if "*" in path:
        path = sorted(glob.glob(path))[-1]

    with open(path) as f:
        samples = [json.loads(l) for l in f]

    passed = 0
    failures = []
    entry_point = None
    for s in samples:
        entry_point = s["doc"]["entry_point"]
        raw = s["resps"][0][0]
        code = extract_code(raw)
        # Two model output styles:
        # 1. Full function (FP8-style): model outputs the entire `def name(...):` block,
        #    possibly wrapped in markdown fences. extract_code() gives us a callable function.
        # 2. Body-only (BF16-style): model outputs just the body continuation with no
        #    signature. We must prepend the prompt to make it callable.
        has_def = any(
            line.lstrip().startswith(f"def {entry_point}")
            for line in code.splitlines()
        )
        if not has_def:
            prompt = s["doc"]["prompt"]
            # Model output body-only without the base 4-space indent.
            # Add 4 spaces to every non-empty line to put it inside the function.
            indented = "\n".join(
                "    " + line if line.strip() else line
                for line in code.splitlines()
            )
            code = prompt + indented
        ok = run_test(code, s["doc"]["test"], entry_point)
        if ok:
            passed += 1
        else:
            failures.append(s["doc"]["entry_point"])

    total = len(samples)
    print(f"HumanEval pass@1: {passed}/{total} = {100 * passed / total:.2f}%")
    if failures:
        print(f"Failed entry points: {failures[:10]}{' ...' if len(failures) > 10 else ''}")


if __name__ == "__main__":
    main()
