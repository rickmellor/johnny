#!/usr/bin/env python3
"""
Code Needle corpus builder.

Parses a Python source file with `ast`, picks a spread of top-level function /
method definitions with a real body, and emits a corpus.json that
code_needle.py can run against: the full file text plus a set of targets
(function name, position through the file, and the literal first 20 lines of
its body) for positional-recall probing.

Defaults to bundling the package's own src/johnny/cli.py as the source — it's
the largest file in the package, self-contained, and ships with every
install, matching the "no external corpus dependency" philosophy in
bundled.py.

Usage:
  build_corpus.py [--source PATH] [--out corpus.json] [--count 16]
"""
import argparse
import ast
import json
import sys
from pathlib import Path

MIN_BODY_LINES = 20  # want a real >=20-line ground-truth window, not a stub
WINDOW = 20  # code_needle.py always asks for "the first 20 lines"


def _default_source() -> Path:
    # scripts/build_corpus.py -> scripts/ -> johnny/ -> johnny/cli.py
    return Path(__file__).resolve().parent.parent / "cli.py"


def _top_level_defs(tree: ast.Module):
    """Module-level functions and class methods — not closures nested inside
    another function's body, so `name` in the prompt is unambiguous."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield child


def collect_candidates(tree: ast.Module, total_lines: int) -> list[dict]:
    """Functions/methods with a real, substantial body: >=20 lines from the
    first statement after the `:` through the end of the def, with 20 full
    lines still available before EOF, deduped by bare name (the prompt refers
    to a function by name alone, so duplicate method names across classes
    would be ambiguous)."""
    seen: set[str] = set()
    out: list[dict] = []
    for node in _top_level_defs(tree):
        if node.name in seen or not node.body:
            continue
        start_line = node.body[0].lineno  # first line *after* the opening colon
        end_line = node.end_lineno or start_line
        body_line_count = end_line - start_line + 1
        if body_line_count < MIN_BODY_LINES:
            continue
        if start_line - 1 + WINDOW > total_lines:
            continue  # not enough lines left before EOF for a full 20-line window
        seen.add(node.name)
        out.append({"name": node.name, "start_line": start_line, "body_line_count": body_line_count})
    return out


def select_spread(candidates: list[dict], count: int) -> list[dict]:
    """Evenly spread `count` picks across the candidates sorted by file
    position, so position_pct coverage lands roughly evenly across the
    first/middle/last third (code_needle's position-bias breakdown)."""
    ordered = sorted(candidates, key=lambda c: c["start_line"])
    n = len(ordered)
    if n <= count:
        return ordered
    if count <= 1:
        return ordered[:count]
    used: set[int] = set()
    chosen_idx: list[int] = []
    for i in range(count):
        idx = round(i * (n - 1) / (count - 1))
        if idx in used:
            for delta in range(1, n):
                moved = False
                for cand in (idx - delta, idx + delta):
                    if 0 <= cand < n and cand not in used:
                        idx = cand
                        moved = True
                        break
                if moved:
                    break
        used.add(idx)
        chosen_idx.append(idx)
    chosen_idx.sort()
    return [ordered[i] for i in chosen_idx]


def build_corpus(source: Path, count: int) -> dict:
    file_text = source.read_text()
    lines = file_text.splitlines()
    total_lines = len(lines)

    tree = ast.parse(file_text, filename=str(source))
    candidates = collect_candidates(tree, total_lines)
    if not candidates:
        sys.exit(f"no candidate functions (>={MIN_BODY_LINES} body lines) found in {source}")

    chosen = select_spread(candidates, count)
    if len(chosen) < count:
        print(f"warning: only {len(chosen)} candidates available (<{count} requested) — "
              f"corpus will have fewer targets than asked", file=sys.stderr)

    targets = []
    for c in sorted(chosen, key=lambda c: c["start_line"]):
        start = c["start_line"]
        expected_lines = lines[start - 1:start - 1 + WINDOW]
        targets.append({
            "name": c["name"],
            "position_pct": round(start / total_lines * 100, 1),
            "start_line": start,
            "expected_lines": expected_lines,
        })

    return {
        "file_text": file_text,
        "token_count": len(file_text) // 4,
        "targets": targets,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(_default_source()),
                     help="Python source file to draw targets from (default: the package's own cli.py).")
    ap.add_argument("--out", default="corpus.json")
    ap.add_argument("--count", type=int, default=16)
    args = ap.parse_args()

    source = Path(args.source).expanduser()
    if not source.is_file():
        sys.exit(f"--source {source} not found")

    corpus = build_corpus(source, args.count)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(corpus, indent=2))

    total_lines = len(corpus["file_text"].splitlines())
    print(f"wrote {len(corpus['targets'])} targets from {source} ({total_lines} lines, "
          f"~{corpus['token_count']:,} tokens) -> {out_path}", file=sys.stderr)
    for t in corpus["targets"]:
        print(f"  {t['name']:30s} pos={t['position_pct']:5.1f}%  start_line={t['start_line']}", file=sys.stderr)


if __name__ == "__main__":
    main()
