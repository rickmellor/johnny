#!/usr/bin/env python3
"""ctxsafe-equivalent against a RUNNING seat: walk needle depths up to max_model_len (single-stream), then N concurrent
deep requests. Prints one JSON line per probe and a final summary. usage: deepprobe.py <base_url> <model> <mml> [conc=4]"""
import sys, json, time, urllib.request, concurrent.futures as cf
base, model, mml = sys.argv[1].rstrip("/"), sys.argv[2], int(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 4
para = ("The fleet manager schedules inference seats across four GPUs, balancing context length against concurrency "
        "while the router classifies each request by domain and complexity. ")
CH_PER_TOK = 6.44   # measured for this prose on the Qwen tokenizer (Gemma is similar ±10%)
def build(target_tokens, code, pos):
    n = max(1, int(target_tokens * CH_PER_TOK / len(para))); body = [para] * n
    body.insert(int(n * pos), f"IMPORTANT FACT: the maintenance code for bay seven is {code}. ")
    return "".join(body) + "\n\nQuestion: What is the maintenance code for bay seven? Reply with just the code."
def ask(prompt, tag, timeout=2400):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 40, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout)); dt = time.time() - t0
        u = r.get("usage") or {}; ans = (r["choices"][0]["message"].get("content") or "").strip()
        return {"tag": tag, "prompt_tokens": u.get("prompt_tokens"), "s": round(dt, 1), "answer": ans[:40]}
    except Exception as e:
        return {"tag": tag, "error": str(e)[:200], "s": round(time.time() - t0, 1)}
res = []
depths = sorted({d for d in (8000, 32000, 64000, 128000, 200000, mml - 6000) if 0 < d <= mml - 4000})
for d in depths:
    code = f"CODE-{d//1000:04d}"; r = ask(build(d, code, 0.62), f"single-{d//1000}K"); r["ok"] = code in r.get("answer", ""); print(json.dumps(r), flush=True); res.append(r)
    if "error" in r: break
# concurrency N at the largest depth that can fit N-way (if the seat's pool allows; otherwise they queue — still a stress test)
d = max(1000, mml - 6000); t0 = time.time()
with cf.ThreadPoolExecutor(N) as ex:
    futs = [ex.submit(ask, build(d, f"CONC-{i}{i}{i}{i}", 0.3 + 0.15 * i), f"conc{N}-{i}-{d//1000}K") for i in range(N)]
    for i, f in enumerate(cf.as_completed(futs)):
        r = f.result(); r["ok"] = any(f"CONC-{j}{j}{j}{j}" in r.get("answer", "") for j in range(N)); print(json.dumps(r), flush=True); res.append(r)
print(json.dumps({"summary": True, "single_ok": sum(1 for r in res if r["tag"].startswith("single") and r.get("ok")), "single_n": len(depths),
                  "conc_ok": sum(1 for r in res if r["tag"].startswith("conc") and r.get("ok")), "conc_n": N, "conc_wall_s": round(time.time() - t0, 1),
                  "errors": sum(1 for r in res if "error" in r)}), flush=True)
