import json,sys,re,subprocess,urllib.request,time,concurrent.futures as cf
sys.path.insert(0,'.'); from hard import T
name=sys.argv[1]; think=len(sys.argv)>2 and sys.argv[2]=="think"; port=sys.argv[3] if len(sys.argv)>3 else "8124"
def one(t):
    i,p,ref,tests=t
    body={"model":__import__("os").environ.get("MODEL","coder"),"messages":[{"role":"user","content":p}],"max_tokens":20000 if think else 2048,"temperature":0,"chat_template_kwargs":{"enable_thinking":think}}
    t0=time.time()
    try:
        r=json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=1800))
        text=r["choices"][0]["message"].get("content") or ""; ntok=r["usage"]["completion_tokens"]
    except Exception as e: text=f"<<ERROR {e}>>"; ntok=0
    m=re.findall(r"```(?:python)?\n(.*?)```",text,re.S); src=max(m,key=len) if m else text
    try:
        pr=subprocess.run([sys.executable,"-c",src+"\n__SRC__="+repr(src)+"\n"+tests],capture_output=True,text=True,timeout=60); ok=pr.returncode==0; err="" if ok else (pr.stderr.strip().splitlines() or ["?"])[-1][:120]
    except subprocess.TimeoutExpired: ok=False; err="timeout"
    return i,ok,round(time.time()-t0,1),ntok,err,text
with cf.ThreadPoolExecutor(4) as ex: res=list(ex.map(one,T))
json.dump({r[0]:{"ok":r[1],"secs":r[2],"tok":r[3],"err":r[4],"answer":r[5]} for r in res},open(f"hard-{name}.json","w"),indent=1)
print(f"{name}: {sum(r[1] for r in res)}/{len(res)} passed | failed: "+", ".join(f"{r[0]}({r[4][:40]})" for r in res if not r[1]))
