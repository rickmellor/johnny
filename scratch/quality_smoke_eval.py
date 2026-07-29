#!/usr/bin/env python3
"""Quick quality battery: 12B (8004) vs 26B (8000), identical prompts, objective scoring."""
import json, re, subprocess, sys, time, urllib.request

ENDPOINTS = {
    "12B": ("http://localhost:8004/v1/chat/completions", "gemma-4-12B-it-FP8-Dynamic"),
    "26B": ("http://localhost:8000/v1/chat/completions", "gemma-4-26B-A4B-it-FP8-Dynamic"),
}

def ask(ep, prompt, max_tokens=512):
    url, model = ENDPOINTS[ep]
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"] or "", time.time() - t0

# ---------- scorers ----------
def score_number(expected):
    def s(out):
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", out.replace("$", ""))
        return any(abs(float(n.replace(",", "")) - expected) < 1e-6 for n in nums[-3:])
    return s

def score_code(fn_name, tests):
    def s(out):
        m = re.search(r"```(?:python)?\n(.*?)```", out, re.S)
        code = m.group(1) if m else out
        prog = code + "\n" + tests + "\nprint('PASS')"
        try:
            r = subprocess.run([sys.executable, "-c", prog], capture_output=True, timeout=10, text=True)
            return "PASS" in r.stdout
        except Exception:
            return False
    return s

def score_json_keys(keys):
    def s(out):
        m = re.search(r"\{.*\}", out, re.S)
        if not m: return False
        try:
            d = json.loads(m.group(0))
            return set(d.keys()) == set(keys)
        except Exception:
            return False
    return s

def score_contains(*subs):
    return lambda out: all(x.lower() in out.lower() for x in subs)

def score_word_count(n):
    return lambda out: len(out.split()) == n

def score_line_count(n, prefix):
    return lambda out: sum(1 for l in out.splitlines() if l.strip().startswith(prefix)) == n

# ---------- battery ----------
SUFFIX_NUM = " Think step by step, then give the final numeric answer on the last line as just the number."
TASKS = [
    # math (GSM8K-style, unambiguous answers)
    ("math: trains", "A train leaves at 9:00 travelling 80 km/h. A second train leaves the same station at 10:30 travelling 120 km/h on a parallel track. At what time (HH:MM, 24h) does the second train catch up?" + " Give the final answer on the last line as HH:MM.", score_contains("13:30"), 400),
    ("math: discount chain", "A jacket costs $250. It is discounted 20%, then a further 15% off the reduced price, then a $10 coupon is applied. What is the final price in dollars?" + SUFFIX_NUM, score_number(160), 400),
    ("math: work rate", "Alice paints a fence in 6 hours, Bob in 4 hours. They work together for 1 hour, then Bob leaves. How many more hours does Alice need to finish? " + SUFFIX_NUM, score_number(3.5), 400),
    ("math: remainders", "What is the smallest positive integer that leaves remainder 2 when divided by 3, remainder 3 when divided by 5, and remainder 2 when divided by 7?" + SUFFIX_NUM, score_number(23), 400),
    ("math: probability", "Two fair dice are rolled. What is the probability the sum is 8, as a fraction in lowest terms? Give the final answer on the last line as a/b.", score_contains("5/36"), 400),
    ("math: percent growth", "A population grows 10% per year for 3 years from 8000. What is it after 3 years? " + SUFFIX_NUM, score_number(10648), 400),
    # code (executed)
    ("code: rle", "Write a Python function `rle(s)` that run-length encodes a string: rle('aaabccd') == 'a3b1c2d1'. Only output the function in a python code block.",
     score_code("rle", "assert rle('aaabccd')=='a3b1c2d1'\nassert rle('')==''\nassert rle('x')=='x1'"), 400),
    ("code: balanced", "Write a Python function `balanced(s)` returning True if brackets ()[]{} in s are balanced/properly nested, ignoring other chars. Only output the function in a python code block.",
     score_code("balanced", "assert balanced('a(b[c]{d})')\nassert not balanced('([)]')\nassert balanced('')\nassert not balanced('(')"), 400),
    ("code: merge intervals", "Write a Python function `merge(iv)` that merges overlapping intervals given as a list of [start,end] lists, returning them sorted. merge([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]. Only output the function in a python code block.",
     score_code("merge", "assert merge([[1,3],[2,6],[8,10]])==[[1,6],[8,10]]\nassert merge([])==[]\nassert merge([[1,4],[4,5]])==[[1,5]]"), 400),
    ("code: roman", "Write a Python function `roman(n)` converting an integer 1-3999 to a Roman numeral string. Only output the function in a python code block.",
     score_code("roman", "assert roman(1994)=='MCMXCIV'\nassert roman(3999)=='MMMCMXCIX'\nassert roman(9)=='IX'\nassert roman(40)=='XL'"), 500),
    # instruction following (mechanical checks)
    ("instr: json only", "Output ONLY a JSON object with exactly these keys: 'city' (a string), 'population' (an integer), 'landlocked' (a boolean), describing Bern, Switzerland. No prose, no code fences.", score_json_keys(["city", "population", "landlocked"]), 200),
    ("instr: exact 12 words", "Describe the ocean in exactly 12 words. Output only those 12 words, no punctuation-only tokens on their own.", score_word_count(12), 100),
    ("instr: 5 bullets", "List exactly 5 renewable energy sources as bullet lines each starting with '- '. Output nothing else.", score_line_count(5, "- "), 200),
    # factual
    ("fact: chemistry", "What is the chemical symbol for tungsten? Answer with just the symbol.", score_contains("W"), 20),
    ("fact: history", "In what year did the Berlin Wall fall? Answer with just the year.", score_contains("1989"), 20),
    ("fact: geography", "Which river flows through Baghdad? Answer with just the river name.", score_contains("Tigris"), 20),
]

# needle-in-haystack at ~4k words
import random
random.seed(7)
words = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
         "incididunt ut labore et dolore magna aliqua").split()
hay = " ".join(random.choice(words) for _ in range(4000))
needle_pos = int(len(hay) * 0.35)
hay_n = hay[:needle_pos] + " The access code for the vault is ZEBRA-7741. " + hay[needle_pos:]
TASKS.append(("needle: 4k ctx", "In the following text a vault access code is hidden. Report just the code.\n\n" + hay_n, score_contains("ZEBRA-7741"), 50))

results = {}
for ep in ENDPOINTS:
    rows = []
    for name, prompt, scorer, mt in TASKS:
        try:
            out, dt = ask(ep, prompt, mt)
            ok = bool(scorer(out))
        except Exception as e:
            out, dt, ok = f"ERROR {e}", 0, False
        rows.append((name, ok, dt, out[-120:].replace("\n", " ")))
        print(f"[{ep}] {'✓' if ok else '✗'} {name} ({dt:.1f}s)", flush=True)
    results[ep] = rows

print("\n===== SUMMARY =====")
for ep, rows in results.items():
    n_ok = sum(1 for _, ok, _, _ in rows if ok)
    print(f"{ep}: {n_ok}/{len(rows)}  (total {sum(dt for _,_,dt,_ in rows):.0f}s)")
print("\nper-task:")
for i, (name, *_ ) in enumerate(TASKS):
    marks = {ep: ("✓" if results[ep][i][1] else "✗") for ep in results}
    diff = "  <-- differs" if len(set(marks.values())) > 1 else ""
    print(f"  {marks['12B']} 12B | {marks['26B']} 26B  {name}{diff}")
json.dump({ep: [(n, ok, round(dt,1), tail) for n, ok, dt, tail in rows] for ep, rows in results.items()},
          open("/tmp/claude-1000/-home-rick/5c8b0026-3805-44d2-8d2a-35663a246d06/scratchpad/eval_results.json", "w"), indent=1)
