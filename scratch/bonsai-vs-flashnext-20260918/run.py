import json, sys, time, urllib.request, re, subprocess, textwrap
from prompts import P
name, base, model = sys.argv[1:4]
out = {}
for pid, cat, prompt, check in P:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
        msg = r["choices"][0]["message"]; text = msg.get("content") or ""; rc = msg.get("reasoning_content") or msg.get("reasoning") or ""
        usage = r.get("usage", {})
    except Exception as e:
        text = f"<<ERROR {e}>>"; rc = ""; usage = {}
    dt = time.time() - t0
    auto = None
    kind, exp = check
    if kind == "exact_num":
        m = re.findall(r"ANSWER:\s*\$?([\d,]+(?:\.\d+)?)", text); auto = bool(m) and float(m[-1].replace(",", "")) == exp
    elif kind == "contains_all":
        auto = all(e.lower() in text.lower() for e in exp)
    elif kind == "code":
        m = re.search(r"```(?:python)?\n(.*?)```", text, re.S); src = (m.group(1) if m else text)
        try:
            p = subprocess.run([sys.executable, "-c", src + "\n" + exp], capture_output=True, text=True, timeout=20); auto = p.returncode == 0
            if not auto: text += "\n<<TEST FAIL: " + (p.stderr.strip().splitlines() or [""])[-1][:200] + ">>"
        except Exception as e: auto = False; text += f"\n<<TEST ERROR {e}>>"
    out[pid] = {"cat": cat, "auto": auto, "secs": round(dt, 1), "completion_tokens": usage.get("completion_tokens"), "reasoning_chars": len(rc), "answer": text}
    print(f"{pid:3s} {cat:12s} auto={auto!s:5s} {dt:5.1f}s tok={usage.get('completion_tokens')} think={len(rc)}", flush=True)
json.dump(out, open(f"answers-{name}.json", "w"), indent=1)
