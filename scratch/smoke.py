#!/usr/bin/env python3
"""Smoke a vLLM OpenAI seat: health, coherent chat, reasoning + tool-call round trip, single-stream tok/s,
N-concurrent coherence.  Usage: smoke.py <base_url> <model> [concurrency]"""
import sys, json, time, urllib.request, concurrent.futures as cf

base, model = sys.argv[1].rstrip("/"), sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8


def post(path, body, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r, time.time() - t0


def chat(msgs, **kw):
    think = kw.pop("think", False)             # thinking models: keep the smoke's short budgets answer-only
    body = {"model": model, "messages": msgs, "max_tokens": kw.pop("max_tokens", 256), "temperature": 0, **kw}
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return post("/v1/chat/completions", body)


ok = True
try:
    r = json.load(urllib.request.urlopen(base + "/v1/models", timeout=10))
    print("models:", [m["id"] for m in r["data"]])
except Exception as e:
    print("MODELS FAIL", e); sys.exit(2)

# 1 coherence + single-stream tok/s (second call so clocks are warm)
chat([{"role": "user", "content": "Say hi."}], max_tokens=8)
r, dt = chat([{"role": "user", "content": "In exactly three sentences, explain what tensor parallelism is."}], max_tokens=200)
txt = r["choices"][0]["message"]["content"] or ""
u = r.get("usage") or {}
ct = u.get("completion_tokens", 0)
print(f"coherence: {ct} tok in {dt:.1f}s = {ct / dt:.1f} tok/s | {txt[:160]!r}")
ok &= len(txt) > 40 and "parallel" in txt.lower()

# 2 reasoning parser — short arithmetic
r, dt = chat([{"role": "user", "content": "What is 17*23? Answer with just the number."}], max_tokens=600, think=True)
m = r["choices"][0]["message"]
rc = m.get("reasoning_content") or m.get("reasoning")
print("reasoning field:", bool(rc), "| answer:", (m.get("content") or "")[:40].strip())
ok &= "391" in (m.get("content") or "")

# 3 tool call + round trip
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
q = [{"role": "user", "content": "What's the weather in Oakland right now? Use the tool."}]
r, dt = chat(q, tools=tools, tool_choice="auto", max_tokens=300)
m = r["choices"][0]["message"]
tc = m.get("tool_calls") or []
print("tool_calls:", [(t["function"]["name"], t["function"]["arguments"]) for t in tc],
      "finish:", r["choices"][0].get("finish_reason"))
ok &= bool(tc) and tc[0]["function"]["name"] == "get_weather"
if tc:
    m2 = {"role": "assistant", "content": m.get("content") or "", "tool_calls": tc}
    r2, _ = chat(q + [m2, {"role": "tool", "tool_call_id": tc[0]["id"],
                           "content": json.dumps({"temp_f": 64, "sky": "fog"})}], tools=tools, max_tokens=120)
    print("tool round-trip:", (r2["choices"][0]["message"]["content"] or "")[:120].strip().replace("\n", " "))

# 4 concurrency
t0 = time.time()
with cf.ThreadPoolExecutor(N) as ex:
    futs = [ex.submit(chat, [{"role": "user", "content": f"Write one sentence about the number {i}."}], max_tokens=60)
            for i in range(N)]
    res = [f.result() for f in futs]
good = sum(1 for r, _ in res if ((r["choices"][0]["message"].get("content") or r["choices"][0]["message"].get("reasoning_content") or "")).strip())
tot = sum((r.get("usage") or {}).get("completion_tokens", 0) for r, _ in res)
el = time.time() - t0
print(f"concurrent {N}: {good}/{N} non-empty, {tot} tok in {el:.1f}s = {tot / el:.0f} tok/s aggregate")
ok &= good == N
print("SMOKE", "PASS" if ok else "FAIL")
