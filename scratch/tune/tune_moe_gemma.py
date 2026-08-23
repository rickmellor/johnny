#!/usr/bin/env python3
"""Ray-free fused-MoE config tuning for gemma-4-26B-A4B (E=128, top-8, moe_inter 704, hidden 2816, FP8 W8A8 per-channel).
Reuses benchmark_moe's worker class (unwrapped from Ray) and save_configs. One process per GPU handles a slice of batch
sizes and writes a partial JSON; --merge combines partials into the final config file.
usage: --gpu G --ngpu N --tp 2 --out DIR   |   --merge --tp 2 --out DIR"""
import sys, os, json, time, argparse, glob
sys.path.insert(0, "/app/vllm/benchmarks/kernels")
import torch
import benchmark_moe as B
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--ngpu", type=int, default=4)
ap.add_argument("--tp", type=int, default=2); ap.add_argument("--out", default="/out"); ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
E, TOPK, MOE_INTER, HIDDEN = 128, 8, 704, 2816
shard_inter = 2 * MOE_INTER // a.tp            # the tuner's "shard_intermediate_size" (gate+up fused, per-TP-rank)
DT = torch.bfloat16; BATCHES = [1,4,16,64,256,1024,4096]
if a.merge:
    best = {}
    for f in sorted(glob.glob(os.path.join(a.out, "partial-gpu*.json"))):
        best.update({int(k): v for k, v in json.load(open(f)).items()})
    B.save_configs(dict(sorted(best.items())), E, shard_inter, HIDDEN, TOPK, DT, True, False, False, None, a.out)
    print("merged", len(best), "batch buckets →", [f for f in os.listdir(a.out) if f.startswith("E=")]); sys.exit(0)
W = B.BenchmarkWorker.__ray_metadata__.modified_class    # plain class behind the @ray.remote wrapper
B.ray.get_gpu_ids = lambda: [0]                           # __init__ asks Ray which GPU it owns; HIP_VISIBLE_DEVICES already scopes us to one
w = W(0)
# RDNA4 prune: gemma's expert GEMMs are tiny (N=704/352 per rank) → small tiles; CDNA-style wide tiles/groups never win here.
space = [c for c in B.get_configs_compute_bound(False, None)
         if c["BLOCK_SIZE_M"] in (16, 32, 64) and c["BLOCK_SIZE_N"] in (16, 32, 64, 128) and c["BLOCK_SIZE_K"] in (32, 64, 128)
         and c["GROUP_SIZE_M"] in (1, 8) and c["num_warps"] in (2, 4, 8) and c.get("waves_per_eu", 0) in (0, 2)]
mine = BATCHES[a.gpu::a.ngpu]
print(f"[gpu{a.gpu}] E={E} shard_inter={shard_inter} hidden={HIDDEN} topk={TOPK} space={len(space)} batches={mine}", flush=True)
best = {}
for M in mine:
    t0 = time.time()
    try:
        cfg = w.tune(M, E, shard_inter, HIDDEN, TOPK, DT, True, False, False, space, None, False)
        best[M] = cfg; print(f"[gpu{a.gpu}] M={M}: {cfg} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"[gpu{a.gpu}] M={M} FAILED: {str(e)[:300]}", flush=True)
    json.dump(best, open(os.path.join(a.out, f"partial-gpu{a.gpu}.json"), "w"))
print(f"[gpu{a.gpu}] DONE", flush=True)
