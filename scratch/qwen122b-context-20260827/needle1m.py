import json,time,urllib.request,sys
base="http://127.0.0.1:8003"; model="qwen-122b-awq"
para=("The fleet manager schedules inference seats across four GPUs, balancing context length against concurrency while the router classifies each request by domain and complexity. ")
CH=6.44
def build(tokens,code,pos=0.5):
    n=max(1,int(tokens*CH/len(para))); body=[para]*n; body.insert(int(n*pos), f"IMPORTANT FACT: the maintenance code for bay seven is {code}. ")
    return "".join(body)+"\n\nQuestion: What is the maintenance code for bay seven? Reply with just the code."
for tokens,code,pos in [(300000,"CODE-0300",0.5),(500000,"CODE-0500",0.3),(750000,"CODE-0750",0.5),(1040000,"CODE-1000",0.25),(1040000,"CODE-1001",0.75)]:
    b={"model":model,"messages":[{"role":"user","content":build(tokens,code,pos)}],"max_tokens":40,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    t=time.time()
    try:
        r=json.load(urllib.request.urlopen(urllib.request.Request(base+"/v1/chat/completions",data=json.dumps(b).encode(),headers={"content-type":"application/json"}),timeout=7200))
        a=(r["choices"][0]["message"].get("content") or "").strip()
        print(json.dumps({"target":tokens,"pos":pos,"prompt_tokens":r["usage"]["prompt_tokens"],"ttft_s":round(time.time()-t,1),"answer":a[:40],"ok":code in a}), flush=True)
    except Exception as e:
        print(json.dumps({"target":tokens,"pos":pos,"error":str(e)[:200],"s":round(time.time()-t,1)}), flush=True)
print("NEEDLE1M DONE", flush=True)
