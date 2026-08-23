import os, sys
sys.path.insert(0,"/src")
os.environ["LMCACHE_CHUNK_SIZE"]="256"
os.environ["LMCACHE_LOCAL_CPU"]="True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"]="4.0"
os.environ["PYTHONHASHSEED"]="0"
def main():
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    ktc = KVTransferConfig(kv_connector="LMCacheMPConnector", kv_role="kv_both")
    llm = LLM(model="/models/Qwen/Qwen3-0.6B", kv_transfer_config=ktc,
              max_model_len=4096, gpu_memory_utilization=0.085, enforce_eager=True)
    sp = SamplingParams(temperature=0, max_tokens=8)
    s = "MP connector probe document. " * 300
    llm.generate([s+"Q:"], sp); llm.generate([s+"Q:"], sp)
    print("RESULT MP connector ran OK", flush=True)
if __name__=="__main__": main()
