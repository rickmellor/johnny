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

Import restoration (default; `--strict-imports` turns it off)
-------------------------------------------------------------
HumanEval prompts begin with their own import lines (`from typing import List,
Tuple`, `import math`, ...). In raw-completion mode the model merely continues
that prompt, so those imports are always part of the tested program. A chat
model instead returns a complete function in a fenced block, and some models
leave the prompt's import out of it — an otherwise correct answer then dies
with `NameError: name 'List' is not defined` before a single assert runs. That
measures "did the model re-type the harness's import line", not code quality:
one model went 84.15% -> 91.46% once the prompt's own imports were restored,
while models that write their own imports were unchanged.

So by default every top-level `import X` / `from X import Y` line of the
task's prompt is prepended to the extracted candidate (a duplicate import is
harmless). `--strict-imports` restores the old behaviour (candidate tested
exactly as the model wrote it). The summary states which mode was used and, in
default mode, how many problems were rescued by the restored imports (they
fail strict, pass default; the strict re-run only happens for candidates that
actually lack one of the prompt's import lines, so it is cheap).

If the extracted code does not define the entry point at all it is treated as
a continuation of the prompt (body only) and the full prompt is prepended —
imports included, in either mode.

Usage:
  python3 humaneval_chat_score.py [--strict-imports] <samples.jsonl>

Or after a run finishes:
  python3 humaneval_chat_score.py ~/vllm-bench-results/<run>/<model>/samples_humaneval_*.jsonl

Prerequisites:
  - The lm-eval run MUST have been invoked with --log_samples.
  - --gen_kwargs "max_gen_toks=2048,until=[]" is recommended so long problems
    don't truncate mid-function. (Default 1024 + until=[\\nclass,\\ndef,...]
    causes ~2 truncation failures out of 164 on Gemma-4.)
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import os
import glob

_IMPORT_RE = re.compile(r"^(?:import\s+\S|from\s+\S+\s+import\s+\S)")


def prompt_imports(prompt: str) -> list[str]:
    """Top-level (unindented) import lines of a HumanEval prompt, in order."""
    return [line.rstrip() for line in prompt.splitlines() if _IMPORT_RE.match(line)]


def defines_entry_point(code: str, entry_point: str) -> bool:
    return any(
        line.lstrip().startswith(f"def {entry_point}")
        for line in code.splitlines()
    )


def _compiles(src: str) -> bool:
    try:
        compile(src, "<candidate>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def as_continuation(prompt: str, code: str) -> str:
    """Body-only answer: prepend the prompt so the body lands inside the function.

    Models write the body either without the base indent (every non-empty line
    then gets 4 spaces — the historical behaviour) or already indented, in which
    case indenting again is an IndentationError after the prompt's docstring.
    An unfenced, already-indented body additionally loses its FIRST line's
    indent to extract_code()'s strip(). The first layout that compiles wins;
    if none does, the historical one is used (and fails the test, as before).
    """
    if prompt and not prompt.endswith("\n"):
        prompt += "\n"
    lines = code.splitlines()
    indented = "\n".join("    " + l if l.strip() else l for l in lines)
    candidates = [indented, code]
    if lines:
        candidates.append("\n".join(["    " + lines[0].lstrip()] + lines[1:]))
    for body in candidates:
        if _compiles(prompt + body):
            return prompt + body
    return prompt + indented


def build_programs(doc: dict, raw: str) -> tuple[str, str | None]:
    """(default-mode program, strict-mode program or None when they are the same).

    The strict program is only returned when the candidate is a full function
    that lacks at least one of the prompt's import lines — the only case where
    the two modes can disagree.
    """
    entry_point = doc["entry_point"]
    code = extract_code(raw)
    # Two model output styles:
    # 1. Full function: the entire `def name(...):` block, possibly fenced.
    # 2. Body-only continuation with no signature: prepend the prompt.
    if not defines_entry_point(code, entry_point):
        return as_continuation(doc["prompt"], code), None
    imports = prompt_imports(doc["prompt"])
    have = {line.strip() for line in code.splitlines()}
    if all(imp in have for imp in imports):
        return code, None
    return "\n".join(imports) + "\n" + code, code


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
    ap = argparse.ArgumentParser(
        description="Re-score lm-eval HumanEval samples for chat-completion models.")
    ap.add_argument("samples", nargs="?", help="samples_humaneval_*.jsonl (a glob picks the newest)")
    ap.add_argument("--strict-imports", action="store_true",
                    help="do NOT restore the prompt's own import lines ahead of the "
                         "candidate (old behaviour)")
    args = ap.parse_args()
    if not args.samples:
        print(__doc__)
        sys.exit(1)

    path = args.samples
    if "*" in path:
        path = sorted(glob.glob(path))[-1]

    with open(path) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    passed = 0
    rescued = 0
    failures = []
    for s in samples:
        doc = s["doc"]
        entry_point = doc["entry_point"]
        raw = s["resps"][0][0]
        program, strict_program = build_programs(doc, raw)
        if args.strict_imports and strict_program is not None:
            program, strict_program = strict_program, None
        ok = run_test(program, doc["test"], entry_point)
        if ok:
            passed += 1
            if strict_program is not None and not run_test(strict_program, doc["test"], entry_point):
                rescued += 1
        else:
            failures.append(entry_point)

    total = len(samples)
    print(f"HumanEval pass@1: {passed}/{total} = {100 * passed / total:.2f}%")
    if args.strict_imports:
        print("Import mode: strict (--strict-imports; prompt imports NOT restored)")
    else:
        print(f"Import mode: prompt imports restored; rescued by restored imports: {rescued}")
    if failures:
        print(f"Failed entry points: {failures[:10]}{' ...' if len(failures) > 10 else ''}")


if __name__ == "__main__":
    main()
