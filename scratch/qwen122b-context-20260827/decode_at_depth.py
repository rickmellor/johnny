import json,time,urllib.request,sys
base="http://127.0.0.1:8003"; model="qwen-122b-awq"
para=("The fleet manager schedules inference seats across four GPUs, balancing context length against concurrency while the router classifies each request by domain and complexity. ")
def build(tokens,code,pos):
    n=max(1,int(tokens*6.44/len(para))); body=[para]*n; body.insert(int(n*pos), f"IMPORTANT FACT: the maintenance code for bay seven is {code}. ")
    return "".join(body)+"\n\nQuestion: What is the maintenance code for bay seven? Reply with just the code. Then write a 250-word essay about fleet scheduling."
tokens,code,pos,tag=int(sys.argv[1]),sys.argv[2],float(sys.argv[3]),sys.argv[4]
b={"model":model,"messages":[{"role":"user","content":build(tokens,code,pos)}],"max_tokens":300,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"stream":True,"stream_options":{"include_usage":True}}
req=urllib.request.Request(base+"/v1/chat/completions",data=json.dumps(b).encode(),headers={"content-type":"application/json"})
t0=time.time(); first=None; usage=None
with urllib.request.urlopen(req,timeout=7200) as resp:
    for line in resp:
        if line.startswith(b"data: ") and b"[DONE]" not in line:
            d=json.loads(line[6:])
            if d.get("usage"): usage=d["usage"]
            ch=d["choices"][0]["delta"].get("content") if d.get("choices") else None
            if ch and first is None: first=time.time()
t1=time.time(); ct=usage["completion_tokens"] if usage else None
print(json.dumps({"tag":tag,"prompt_tokens":usage and usage.get("prompt_tokens"),"ttft_s":round(first-t0,1),"completion_tokens":ct,"decode_tok_s":round((ct-1)/(t1-first),1) if ct else None}), flush=True)
