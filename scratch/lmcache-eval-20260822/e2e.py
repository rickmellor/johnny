import os, time, sys
sys.path.insert(0,"/src")
os.environ["LMCACHE_CHUNK_SIZE"]="256"
os.environ["LMCACHE_LOCAL_CPU"]="True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"]="4.0"
os.environ["PYTHONHASHSEED"]="0"

def main():
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    ktc = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
    llm = LLM(model="/models/Qwen/Qwen3-0.6B", kv_transfer_config=ktc,
              max_model_len=4096, gpu_memory_utilization=0.085, enforce_eager=True)
    sp = SamplingParams(temperature=0, max_tokens=16)
    shared = "The quick brown fox jumps over the lazy dog. " * 300
    def run(tag, p):
        t0=time.time(); out=llm.generate([p], sp); dt=time.time()-t0
        print(f"RESULT {tag}: {dt:.2f}s text={out[0].outputs[0].text[:40]!r}", flush=True)
        return dt
    a=run("cold", shared+"Question one:")
    b=run("warm", shared+"Question two:")
    print(f"RESULT speedup: {a/b:.2f}x", flush=True)

if __name__ == "__main__":
    main()
