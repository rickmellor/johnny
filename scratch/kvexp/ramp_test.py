#!/usr/bin/env python3
"""Sustained-load ramp test: keep 4 streams busy for N minutes, report decode tok/s per minute."""
import json, time, urllib.request, concurrent.futures as cf, sys
base, model, minutes = sys.argv[1].rstrip("/"), sys.argv[2], float(sys.argv[3]) if len(sys.argv)>3 else 12
def chat(i, mx=300):
    req=urllib.request.Request(base+"/v1/chat/completions", data=json.dumps({"model":model,"messages":[{"role":"user","content":f"Write a detailed essay #{i} about the history of aviation, at least 250 words."}],"max_tokens":mx,"temperature":0.7,"chat_template_kwargs":{"enable_thinking":False}}).encode(), headers={"content-type":"application/json"})
    t0=time.time(); n=json.load(urllib.request.urlopen(req, timeout=600)).get("usage",{}).get("completion_tokens",0); return n, time.time()-t0
t_start=time.time(); i=0; win_tok=0; win_t0=time.time()
with cf.ThreadPoolExecutor(4) as ex:
    while time.time()-t_start < minutes*60:
        res=list(ex.map(chat, range(i, i+4))); i+=4
        win_tok += sum(n for n,_ in res)
        if time.time()-win_t0 >= 60:
            el=time.time()-win_t0; print(f"min {int((time.time()-t_start)/60):>2}: {win_tok/el:6.1f} tok/s agg (4 streams) = {win_tok/el/4:5.1f}/stream", flush=True); win_tok=0; win_t0=time.time()
