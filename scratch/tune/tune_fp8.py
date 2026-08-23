#!/usr/bin/env python3
"""Tune vLLM's W8A8 block-FP8 Triton GEMM configs for this GPU, for the Qwen3.x-27B shape set.
Runs inside the vLLM image. One process per GPU (HIP_VISIBLE_DEVICES), shapes split by --gpu/--ngpu."""
import sys, os, time, json, argparse
sys.path.insert(0, "/app/vllm/benchmarks/kernels")
import torch
import benchmark_w8a8_block_fp8 as T
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, required=True); ap.add_argument("--ngpu", type=int, default=4)
ap.add_argument("--out", default="/out"); a = ap.parse_args()
# Qwen3.5-family 27B (hidden 5120, inter 17408, 24 q-heads x 256, 4 kv-heads): TP4 shapes seen in the seat log, TP2 = doubled
SHAPES = [(3584,5120),(4096,5120),(5120,1536),(5120,4352),(8704,5120),      # TP4
          (7168,5120),(8192,5120),(5120,3072),(5120,8704),(17408,5120)]    # TP2
BATCHES = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096]
space = [c for c in T.get_configs_compute_bound() if 128 % c["BLOCK_SIZE_K"] == 0
         and c["num_stages"] in (2, 3) and c["GROUP_SIZE_M"] in (1, 16)
         and c["BLOCK_SIZE_M"] <= 128 and c["BLOCK_SIZE_N"] <= 128]   # 256-wide tiles: minutes per compile on gfx1201, never win   # pruned for RDNA: deep pipelines/large groups never win here
mine = SHAPES[a.gpu::a.ngpu]
print(f"[gpu{a.gpu}] {torch.cuda.get_device_name(0)} shapes={mine} space={len(space)}", flush=True)
for (N,K) in mine:
    t0=time.time(); best={}
    for M in BATCHES:
        t1=time.time()
        try:
            best[M]=T.tune(M,N,K,[128,128],torch.bfloat16,space,"fp8")
        except Exception as e:
            print(f"[gpu{a.gpu}] N={N},K={K} M={M} FAILED: {str(e)[:200]}", flush=True); continue
        print(f"[gpu{a.gpu}] N={N},K={K} M={M}: {best[M]} ({time.time()-t1:.0f}s)", flush=True)
    T.save_configs(N,K,128,128,best,a.out,"fp8")
    print(f"[gpu{a.gpu}] saved N={N},K={K} in {time.time()-t0:.0f}s", flush=True)
print(f"[gpu{a.gpu}] DONE", flush=True)
