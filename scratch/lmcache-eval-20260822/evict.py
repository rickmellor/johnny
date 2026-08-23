import os, time, sys
sys.path.insert(0,"/src")
MODE = sys.argv[1]   # apc_only | lmcache
os.environ["LMCACHE_CHUNK_SIZE"]="256"
os.environ["LMCACHE_LOCAL_CPU"]="True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"]="8.0"
os.environ["PYTHONHASHSEED"]="0"

def main():
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    kw = dict(model="/models/Qwen/Qwen3-0.6B", max_model_len=4096,
              gpu_memory_utilization=0.085, enforce_eager=True)
    if MODE == "lmcache":
        kw["kv_transfer_config"]=KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0, max_tokens=8)

    # WARMUP: absorb one-time compile/alloc cost so it can't pollute timings
    llm.generate(["warmup prompt "*10], sp)

    def timed(p):
        t0=time.time(); o=llm.generate([p], sp); return time.time()-t0

    TARGET = "ALPHA target document. " * 300      # ~2.4k tok, the prefix we care about
    # 1. prime it
    t_prime = timed(TARGET + "Q:")
    # 2. immediate repeat -> GPU APC should hit
    t_hot = timed(TARGET + "Q:")
    # 3. flood with distinct long prompts to evict TARGET from the small GPU cache
    for i in range(12):
        timed(f"FILLER-{i} unrelated content block. " * 300 + "Q:")
    # 4. come back to TARGET -> APC-only should miss & recompute; LMCache should reload from CPU
    t_after = timed(TARGET + "Q:")
    print(f"MODE={MODE} prime={t_prime:.3f}s hot={t_hot:.3f}s after_evict={t_after:.3f}s", flush=True)

if __name__=="__main__": main()
