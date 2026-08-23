#!/usr/bin/env python3
"""Max-depth + concurrency-4 probe for the TP4 seat."""
import json, time, urllib.request, concurrent.futures as cf, sys
base="http://127.0.0.1:8003"; model="Qwen3.8-27B-FP8"
para=("The fleet manager schedules inference seats across four GPUs, balancing context length against concurrency "
      "while the router classifies each request by domain and complexity. ")
def build(target, code, pos=0.5):
    n=int(target*6.44/len(para)); body=[para]*n
    body.insert(int(n*pos), f"IMPORTANT FACT: the maintenance code for bay seven is {code}. ")
    return "".join(body)+"\n\nQuestion: What is the maintenance code for bay seven? Reply with just the code."
def ask(prompt, tag, timeout=1800):
    req=urllib.request.Request(base+"/v1/chat/completions", data=json.dumps({"model":model,
        "messages":[{"role":"user","content":prompt}],"max_tokens":40,"temperature":0,
        "chat_template_kwargs":{"enable_thinking":False}}).encode(), headers={"content-type":"application/json"})
    t0=time.time()
    try:
        r=json.load(urllib.request.urlopen(req, timeout=timeout)); dt=time.time()-t0
        u=r.get("usage",{}); ans=(r["choices"][0]["message"]["content"] or "").strip()
        return dict(tag=tag, ptok=u.get("prompt_tokens"), s=round(dt,1), ans=ans[:40])
    except Exception as e:
        return dict(tag=tag, err=str(e)[:150], s=round(time.time()-t0,1))
if sys.argv[1]=="single":
    for depth,code in ((160000,"ORCHID-4471"),(250000,"FALCON-9920")):
        print(ask(build(depth,code,0.62), f"single-{depth//1000}K"), flush=True)
else:
    codes=["ALPHA-1111","BRAVO-2222","CHARLIE-3333","DELTA-4444"]; depth=int(sys.argv[2])
    t0=time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        futs=[ex.submit(ask, build(depth, codes[i], 0.3+0.15*i), f"c{i}-{codes[i]}") for i in range(4)]
        for f in cf.as_completed(futs): print(f.result(), flush=True)
    print(f"wall {time.time()-t0:.1f}s", flush=True)
