"""humaneval_chat_score.py: import-tolerant re-scoring of chat-style HumanEval answers."""
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "src/johnny/scripts/humaneval_chat_score.py"

PROMPT = (
    "from typing import List\n\n\n"
    "def total(xs: List[int]) -> int:\n"
    '    """ Sum the list.\n    >>> total([1, 2])\n    3\n    """\n'
)
TEST = "def check(candidate):\n    assert candidate([1, 2, 3]) == 6\n    assert candidate([]) == 0\n"

NO_IMPORT = "Here you go:\n```python\ndef total(xs: List[int]) -> int:\n    return sum(xs)\n```\n"
OWN_IMPORT = ("```python\nfrom typing import List\n\n"
              "def total(xs: List[int]) -> int:\n    return sum(xs)\n```")
BODY_ONLY = "```python\nacc = 0\nfor x in xs:\n    acc += x\nreturn acc\n```"
BODY_INDENTED = "    acc = 0\n    for x in xs:\n        acc += x\n    return acc\n"
WRONG = ("```python\nfrom typing import List\n\n"
         "def total(xs: List[int]) -> int:\n    return 1 + sum(xs)\n```")


def _row(i, resp, entry="total", prompt=PROMPT, test=TEST):
    return {"doc": {"task_id": f"HumanEval/{i}", "prompt": prompt, "test": test,
                    "entry_point": entry}, "resps": [[resp]]}


def _score(tmp_path, resps, *flags):
    p = tmp_path / "samples_humaneval_test.jsonl"
    p.write_text("".join(json.dumps(_row(i, r)) + "\n" for i, r in enumerate(resps)))
    r = subprocess.run([sys.executable, str(SCRIPT), *flags, str(p)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _passed(out):
    m = re.search(r"^HumanEval pass@1: (\d+)/(\d+) = (\d+\.\d{2})%$", out, re.M)
    assert m, out
    return int(m.group(1)), int(m.group(2))


def test_missing_import_rescued_by_default(tmp_path):
    out = _score(tmp_path, [NO_IMPORT])
    assert _passed(out) == (1, 1)
    assert "rescued by restored imports: 1" in out
    assert "Failed entry points" not in out


def test_missing_import_fails_strict(tmp_path):
    out = _score(tmp_path, [NO_IMPORT], "--strict-imports")
    assert _passed(out) == (0, 1)
    assert "Import mode: strict" in out
    assert "Failed entry points: ['total']" in out


def test_own_imports_pass_in_both_modes(tmp_path):
    out = _score(tmp_path, [OWN_IMPORT])
    assert _passed(out) == (1, 1)
    assert "rescued by restored imports: 0" in out
    assert _passed(_score(tmp_path, [OWN_IMPORT], "--strict-imports")) == (1, 1)


def test_continuation_style_passes(tmp_path):
    # body-only, both without and with the base indent, in both modes
    assert _passed(_score(tmp_path, [BODY_ONLY, BODY_INDENTED])) == (2, 2)
    assert _passed(_score(tmp_path, [BODY_ONLY, BODY_INDENTED], "--strict-imports")) == (2, 2)


def test_wrong_answer_fails(tmp_path):
    out = _score(tmp_path, [WRONG])
    assert _passed(out) == (0, 1)
    assert "Failed entry points: ['total']" in out


def test_output_format_matches_bench_parser(tmp_path):
    out = _score(tmp_path, [NO_IMPORT, OWN_IMPORT, WRONG])
    assert out.splitlines()[0] == "HumanEval pass@1: 2/3 = 66.67%"
    from johnny.bench import parse_humaneval_score
    res = parse_humaneval_score(out)
    assert (res["passed"], res["total"], res["pass_at_1_pct"]) == (2, 3, 66.67)
    assert res["failed_entry_points_sample"] == "['total']"
