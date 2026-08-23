#!/usr/bin/env python3
"""Create (or refresh) a KV-experiment placement by cloning a base placement with overrides.
usage: mkplacement.py <model> <base_placement_id> <new_id> key=value ...   (keys: image, kv, mml, seqs, gmu, bt, env.X=Y, flag+=--foo)
prints the new id."""
import sys, copy, json
from johnny.registry import store
model, base_id, new_id, *kv = sys.argv[1:4] + [sys.argv[4:]]
kv = kv[0] if kv and isinstance(kv[0], list) else sys.argv[4:]
reg = store.load(); pls = reg["models"][model]["placements"]
base = next(p for p in pls if p["id"] == base_id)
c = copy.deepcopy(base); c["id"] = new_id; c["perf"] = None; c["quality"] = {}; c["validated_at"] = None; c["source"] = "manual"
c["extra"] = dict(c.get("extra") or {}); c["env"] = dict(c.get("env") or {}); c["knobs"] = dict(c.get("knobs") or {})
for item in kv:
    k, _, v = item.partition("=")
    if k == "image": c["image"] = v
    elif k == "kv": c["knobs"]["kv_cache_dtype"] = v
    elif k == "mml": c["knobs"]["max_model_len"] = int(v)
    elif k == "seqs": c["knobs"]["max_num_seqs"] = int(v)
    elif k == "gmu": c["knobs"]["gpu_memory_util"] = float(v)
    elif k == "bt": c["knobs"]["max_num_batched_tokens"] = int(v)
    elif k == "note": c["extra"]["note"] = v
    elif k.startswith("env."): c["env"][k[4:]] = v
    elif k == "flag+": c["extra"]["extra_flags"] = list(c["extra"].get("extra_flags") or []) + v.split()
    elif k == "runtime": c.setdefault("validation_key", {})["runtime_version"] = v
    else: raise SystemExit(f"unknown key {k}")
pls[:] = [p for p in pls if p["id"] != new_id]; pls.insert(0, c); store.save(reg)
print(new_id)
