import json,glob,re,subprocess,sys,tempfile,os
def extract(raw):
    m=re.findall(r"```(?:python)?\n(.*?)```",raw,re.S); return max(m,key=len) if m else raw
def score(name):
    f=sorted(glob.glob(f'{name}/humaneval/**/samples_humaneval_*.jsonl',recursive=True))[-1]; ok=0; n=0; fails=[]
    for l in open(f):
        r=json.loads(l); d=r["doc"]; out=r["resps"][0][0] if isinstance(r["resps"][0],list) else r["resps"][0]
        pre="\n".join(x for x in d["prompt"].split("\n") if re.match(r"^(import |from \S+ import )",x))
        code=extract(out)
        if f"def {d['entry_point']}" not in code: code=d["prompt"]+code      # continuation-style answer
        src=pre+"\n"+code+"\n"+d["test"]+f"\ncheck({d['entry_point']})\n"
        try: p=subprocess.run([sys.executable,"-c",src],capture_output=True,text=True,timeout=15); good=p.returncode==0
        except subprocess.TimeoutExpired: good=False
        n+=1; ok+=good
        if not good: fails.append(d["entry_point"])
    print(f"{name:24s} HumanEval (imports restored): {ok}/{n} = {100*ok/n:.2f}%  | failed: {', '.join(fails[:12])}{' ...' if len(fails)>12 else ''}")
for n in sys.argv[1:]: score(n)
