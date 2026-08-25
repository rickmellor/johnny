#!/usr/bin/env python3
"""Short-horizon tool-loop probe: can a model finish a 3-8 step delegate task?

Everything measured about gemma's agentic weakness came from 30-50 step AutomationBench loops
(0-6.7% pass, 13/30 hitting the step cap). Real coder-delegate work is much shorter: read a
file, edit it, run tests, maybe iterate. This probe simulates that regime with a tiny in-process
filesystem+test sandbox and a hard 8-step budget, so a model that thrashes long horizons can
still pass if short loops are within reach.

Usage: shorthorizon.py <base_url> <model> [label]
"""
import json, sys, time, urllib.request, copy

BASE, MODEL = sys.argv[1].rstrip("/"), sys.argv[2]
LABEL = sys.argv[3] if len(sys.argv) > 3 else MODEL
ROADMAP = "--roadmap" in sys.argv        # deliver an explicit step plan with the brief
MAX_STEPS = 8

TOOLS = [
 {"type":"function","function":{"name":"list_files","description":"List files in the project.",
   "parameters":{"type":"object","properties":{},"required":[]}}},
 {"type":"function","function":{"name":"read_file","description":"Read a file's contents.",
   "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"write_file","description":"Overwrite a file with new contents.",
   "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
 {"type":"function","function":{"name":"run_tests","description":"Run the test suite. Returns pass/fail and output.",
   "parameters":{"type":"object","properties":{},"required":[]}}},
]

TASKS = [
 {"name":"fix_off_by_one",
  "roadmap":'Plan: 1) read calc.py 2) the loop range excludes the last item — change it to cover all items 3) write the fixed file 4) run_tests to confirm.',
  "files":{"calc.py":"def total(items):\n    s = 0\n    for i in range(len(items) - 1):\n        s += items[i]\n    return s\n",
           "test_calc.py":"from calc import total\n\ndef test_total():\n    assert total([1,2,3]) == 6\n"},
  "goal":"The test suite is failing. Fix the bug in calc.py so the tests pass.",
  "check": lambda fs: _run(fs)[0]},
 {"name":"add_function",
  "roadmap":'Plan: 1) read test_util.py to see what titleize must do 2) read util.py 3) append a titleize function that title-cases each word 4) write the file 5) run_tests.',
  "files":{"util.py":"def slugify(s):\n    return s.lower().replace(' ', '-')\n",
           "test_util.py":"from util import slugify, titleize\n\ndef test_slug():\n    assert slugify('Hello World') == 'hello-world'\n\ndef test_title():\n    assert titleize('hello world') == 'Hello World'\n"},
  "goal":"Add the missing `titleize` function to util.py so all tests pass.",
  "check": lambda fs: _run(fs)[0]},
 {"name":"fix_exception",
  "roadmap":'Plan: 1) read test_parse.py to see both expectations 2) read parse.py 3) wrap the int() call in try/except ValueError returning None 4) write the file 5) run_tests.',
  "files":{"parse.py":"def to_int(s):\n    return int(s)\n",
           "test_parse.py":"from parse import to_int\n\ndef test_ok():\n    assert to_int('5') == 5\n\ndef test_bad():\n    assert to_int('abc') is None\n"},
  "goal":"Make the failing test pass by handling invalid input in parse.py.",
  "check": lambda fs: _run(fs)[0]},
 {"name":"two_file_edit",
  "roadmap":'Plan: 1) read test_ab.py — it expects doubled() == 4 2) read a.py and b.py 3) EDIT b.py ONLY: make doubled() return VALUE * 4 (leave a.py alone) 4) write b.py 5) run_tests.',
  "files":{"a.py":"VALUE = 1\n","b.py":"from a import VALUE\n\ndef doubled():\n    return VALUE\n",
           "test_ab.py":"from b import doubled\n\ndef test_doubled():\n    assert doubled() == 4\n"},
  "goal":"Make the test pass. You may edit either file.",
  "check": lambda fs: _run(fs)[0]},
 {"name":"read_then_fix",
  "roadmap":'Plan: 1) read test_app.py — budget() must equal 120 2) read app.py to see budget = retries * timeout 3) read cfg.py 4) set retries=4 and timeout=30 in cfg.py (retries must stay <= 4) 5) write cfg.py 6) run_tests.',
  "files":{"cfg.py":"SETTINGS = {'retries': 3, 'timeout': 30}\n",
           "app.py":"from cfg import SETTINGS\n\ndef budget():\n    return SETTINGS['retries'] * SETTINGS['timeout']\n",
           "test_app.py":"from app import budget\n\ndef test_budget():\n    assert budget() == 120\n"},
  "goal":"The test expects a budget of 120. Adjust the configuration in cfg.py to make it pass (keep retries at 4 or fewer).",
  "check": lambda fs: _run(fs)[0]},
]

def _run(fs):
    import subprocess, tempfile, os
    with tempfile.TemporaryDirectory() as d:
        for k, v in fs.items():
            open(os.path.join(d, k), "w").write(v)
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", d], capture_output=True, text=True, timeout=120, cwd=d)
        return p.returncode == 0, (p.stdout + p.stderr)[-600:]

def post(body, timeout=300):
    r = urllib.request.Request(BASE+"/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"content-type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))

def run_task(t):
    fs = copy.deepcopy(t["files"])
    goal = t["goal"]
    if ROADMAP and t.get("roadmap"):
        goal = f"{goal}\n\n{t['roadmap']}\n\nFollow the plan above in order. Do not re-plan."
    msgs = [{"role":"system","content":"You are a coding agent. Use the tools to inspect and edit files, then run the tests. Stop when the tests pass."},
            {"role":"user","content":goal}]
    steps = 0
    for _ in range(MAX_STEPS):
        d = post({"model":MODEL,"messages":msgs,"tools":TOOLS,"tool_choice":"auto","max_tokens":1500,
                  "temperature":0,"chat_template_kwargs":{"enable_thinking":False}})
        m = d["choices"][0]["message"]; tcs = m.get("tool_calls") or []
        msgs.append({"role":"assistant","content":m.get("content") or "","tool_calls":tcs} if tcs
                    else {"role":"assistant","content":m.get("content") or ""})
        if not tcs:
            break
        for tc in tcs:
            steps += 1
            fn = tc["function"]["name"]
            try: args = json.loads(tc["function"]["arguments"] or "{}")
            except Exception: args = {}
            if fn == "list_files": out = {"files": sorted(fs)}
            elif fn == "read_file": out = {"content": fs.get(args.get("path",""), "<no such file>")}
            elif fn == "write_file":
                fs[args.get("path","")] = args.get("content","") ; out = {"ok": True}
            elif fn == "run_tests":
                ok, log = _run(fs); out = {"passed": ok, "output": log}
            else: out = {"error": f"unknown tool {fn}"}
            msgs.append({"role":"tool","tool_call_id":tc["id"],"content":json.dumps(out)[:4000]})
    ok = t["check"](fs)
    return {"task": t["name"], "passed": bool(ok), "tool_steps": steps}

res = []
t0 = time.time()
for t in TASKS:
    try:
        r = run_task(t)
    except Exception as e:
        r = {"task": t["name"], "passed": False, "tool_steps": -1, "error": str(e)[:150]}
    res.append(r); print("  " + json.dumps(r), flush=True)
p = sum(1 for r in res if r["passed"])
print(json.dumps({"label": LABEL, "roadmap": ROADMAP, "passed": p, "total": len(TASKS),
                  "pass_pct": round(100*p/len(TASKS),1),
                  "mean_steps": round(sum(max(0,r['tool_steps']) for r in res)/len(res),1),
                  "wall_s": round(time.time()-t0,1)}))
