#!/usr/bin/env python3
"""Manual perf: single-stream tok/s and N-concurrent aggregate on a running seat (johnny bench perf over-ramps seqs=4 seats)."""
import sys, json, time, urllib.request, concurrent.futures as cf
base, model = sys.argv[1].rstrip("/"), sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 4
def chat(msg, mx=400):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps({"model": model, "messages": [{"role": "user", "content": msg}],
          "max_tokens": mx, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(), headers={"content-type": "application/json"})
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=900)); return (r.get("usage") or {}).get("completion_tokens", 0), time.time() - t0
chat("hi", 8)
ct, dt = chat("Write a Python function that merges two sorted lists, with a docstring and a short example.", 400)
out = {"single_tok_s": round(ct / dt, 1)}
t0 = time.time()
with cf.ThreadPoolExecutor(N) as ex:
    res = [f.result() for f in [ex.submit(chat, f"Write a Python function #{i} that validates an IPv4 address, with tests.", 400) for i in range(N)]]
el = time.time() - t0; tot = sum(c for c, _ in res); out[f"conc{N}_agg_tok_s"] = round(tot / el, 1); out[f"conc{N}_per_stream"] = round(tot / el / N, 1)
print(json.dumps(out))
