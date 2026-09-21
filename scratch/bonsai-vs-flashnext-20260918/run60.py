import json, sys, time, urllib.request, re
from prompts60 import P
name, base, model = sys.argv[1:4]
def check(kind, exp, text):
    t = text.strip()
    if kind == "exact_num":
        m = re.findall(r"ANSWER:\s*\$?([\d,]+(?:\.\d+)?)", t); return bool(m) and float(m[-1].replace(",", "")) == exp
    if kind == "contains_all": return all(e.lower() in t.lower() for e in exp)
    if kind == "contains_any": return any(e.lower() in t.lower() for e in exp)
    if kind == "regex": return re.match(exp, t, re.I) is not None
    if kind == "ordered":
        pos = [t.lower().find(w) for w in exp]; return all(p >= 0 for p in pos) and pos == sorted(pos)
    if kind == "json_keys":
        try: d = json.loads(re.sub(r"^```\w*|```$", "", t).strip())
        except Exception: return False
        return isinstance(d.get("name"), str) and isinstance(d.get("age"), int) and isinstance(d.get("tags"), list) and len(d["tags"]) == 3 and all(isinstance(x, str) for x in d["tags"])
    if kind == "json_equals":
        try: return json.loads(re.sub(r"^```\w*|```$", "", t).strip()) == exp
        except Exception: return False
    return None
out = {}
for pid, cat, prompt, (kind, exp) in P:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
        text = r["choices"][0]["message"].get("content") or ""
    except Exception as e: text = f"<<ERROR {e}>>"
    auto = check(kind, exp, text)
    out[pid] = {"cat": cat, "kind": kind, "auto": auto, "secs": round(time.time() - t0, 1), "answer": text}
    print(f"{pid} {cat:12s} auto={auto!s:5s}", flush=True)
json.dump(out, open(f"answers60-{name}.json", "w"), indent=1)
