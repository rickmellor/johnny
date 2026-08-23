import os, time, sys
sys.path.insert(0,"/src")
MODE = sys.argv[1]   # base_apc | base_noapc | lmc_noapc
os.environ["LMCACHE_CHUNK_SIZE"]="256"
os.environ["LMCACHE_LOCAL_CPU"]="True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"]="4.0"
os.environ["PYTHONHASHSEED"]="0"

def main():
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    kw = dict(model="/models/Qwen/Qwen3-0.6B", max_model_len=4096,
              gpu_memory_utilization=0.085, enforce_eager=True)
    if MODE == "base_apc":
        kw["enable_prefix_caching"]=True
    elif MODE == "base_noapc":
        kw["enable_prefix_caching"]=False
    elif MODE == "lmc_noapc":
        kw["enable_prefix_caching"]=False
        kw["kv_transfer_config"]=KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0, max_tokens=16)
    shared = "The quick brown fox jumps over the lazy dog. " * 300
    ts=[]
    for i,q in enumerate(["Question one:","Question two:","Question three:"]):
        t0=time.time(); llm.generate([shared+q], sp); ts.append(time.time()-t0)
    print(f"MODE={MODE} cold={ts[0]:.2f}s warm1={ts[1]:.2f}s warm2={ts[2]:.2f}s", flush=True)

if __name__=="__main__": main()
