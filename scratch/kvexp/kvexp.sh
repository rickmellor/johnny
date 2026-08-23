#!/usr/bin/env bash
# KV-dtype experiment series for one seat. usage: kvexp.sh <tag> <model> <base_placement> <image> <port> <cfg1,cfg2,...> <runtime>
#   cfg: bf16 (base knobs, +johnny bench perf A/B) | int4 (int4_per_token_head, mml 262144, seqs 4) | tq (turboquant_4bit_nc, same)
set -u
TAG="$1"; MODEL="$2"; BASE="$3"; IMAGE="$4"; PORT="$5"; CFGS="$6"; RT="${7:-exp}"
S=$(cd "$(dirname "$0")" && pwd); L="$S/kvexp-$TAG.log"; J="$HOME/repos/johnny"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
mk(){ (cd "$J" && uv run --quiet python "$S/mkplacement.py" "$@"); }
seat_name(){ docker ps -a --format '{{.Names}}' | grep -E "^johnny-.*-$PORT$" | head -1; }
wait_ready(){ # $1 = container name; prints READY|ERROR|GONE and the KV line
  for i in $(seq 1 110); do
    L2=$(docker logs "$1" 2>&1 | grep -v "not documented")
    if echo "$L2" | grep -q "Application startup complete"; then echo READY; return 0; fi
    if echo "$L2" | grep -qiE "Traceback|RuntimeError|hipError|Engine core initialization failed|OutOfResources|ValueError|AssertionError"; then echo ERROR; return 1; fi
    if ! docker ps --format '{{.Names}}' | grep -q "^$1$"; then echo GONE; return 1; fi
    sleep 10; done; echo TIMEOUT; return 1; }
for CFG in ${CFGS//,/ }; do
  case $CFG in
    bf16) KV=auto; PID="kvexp-$TAG-bf16-tuned"; OV=("image=$IMAGE" "kv=auto" "runtime=$RT" "note=KV experiment baseline on tuned image (base knobs)");;
    int4) KV=int4_per_token_head; PID="kvexp-$TAG-int4-mml262144"; OV=("image=$IMAGE" "kv=$KV" "mml=262144" "seqs=4" "gmu=0.93" "bt=8192" "runtime=$RT" "note=KV experiment: int4_per_token_head, max context, concurrency 4");;
    tq)   KV=turboquant_4bit_nc; PID="kvexp-$TAG-tq4-mml262144"; OV=("image=$IMAGE" "kv=$KV" "mml=262144" "seqs=4" "gmu=0.93" "bt=8192" "runtime=$RT" "note=KV experiment: turboquant_4bit_nc, max context, concurrency 4");;
    *) log "unknown cfg $CFG"; continue;;
  esac
  log "=== $TAG/$CFG → placement $PID (kv=$KV image=$IMAGE)"
  mk "$MODEL" "$BASE" "$PID" "${OV[@]}" >>"$L" 2>&1
  MML=$(cd "$J" && uv run --quiet python -c "from johnny.registry import store; print(next(p for p in store.load()['models']['$MODEL']['placements'] if p['id']=='$PID')['knobs']['max_model_len'])")
  for attempt in 1 2; do
    old=$(seat_name); [ -n "$old" ] && { johnny down "$old" >>"$L" 2>&1; sleep 5; }
    log "launch attempt $attempt (mml=$MML)"; timeout 1500 johnny up "$MODEL" --placement "$PID" --port "$PORT" --wait >>"$L" 2>&1
    C=$(seat_name); [ -z "$C" ] && { log "no container appeared"; continue; }
    ST=$(wait_ready "$C"); log "state: $ST"
    docker logs "$C" 2>&1 | grep -v "not documented" | grep -iE "GPU KV cache size|Maximum concurrency|Using .* backend|attention backend|kv_cache_dtype|Traceback|ValueError|OutOfResources|RuntimeError" | tail -12 | cut -c1-220 >>"$L"
    if [ "$ST" = READY ]; then break; fi
    # too-large mml? vLLM says: "... max seq len (X) is larger than the maximum number of tokens that can be stored in KV cache (N)"
    N=$(docker logs "$C" 2>&1 | grep -oE "stored in KV cache \(([0-9]+)\)" | grep -oE "[0-9]+" | tail -1)
    if [ -n "$N" ] && [ "$attempt" = 1 ]; then MML=$(( (N - 2048) / 1024 * 1024 )); log "KV too small for $MML → retry with mml=$MML"; mk "$MODEL" "$BASE" "$PID" "${OV[@]}" "mml=$MML" >>"$L" 2>&1; else break; fi
  done
  if [ "$ST" != READY ]; then log "!! $TAG/$CFG failed to start — skipping evals"; echo "{\"cfg\":\"$CFG\",\"status\":\"launch-failed\"}" > "$S/result-$TAG-$CFG.json"; continue; fi
  KVLINE=$(docker logs "$C" 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens" | tail -1); CONC=$(docker logs "$C" 2>&1 | grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" | tail -1)
  BACKEND=$(docker logs "$C" 2>&1 | grep -oiE "Using [A-Z_]+ (attention )?backend|Overriding with [A-Z_]+|backend=[A-Z_]+" | tail -1)
  log "KV: $KVLINE | $CONC | $BACKEND"
  URL="http://127.0.0.1:$PORT"
  [ "$CFG" = bf16 ] && { log "johnny bench perf (A/B vs untuned)"; timeout 1800 johnny bench "$PID" --suite perf >>"$L" 2>&1; }
  log "humaneval"; timeout 3600 johnny bench "$PID" --suite humaneval >>"$L" 2>&1
  log "arc 200";   timeout 3600 johnny bench "$PID" --suite arc --limit 200 >>"$L" 2>&1
  log "needle";    timeout 3600 johnny bench "$PID" --suite needle >>"$L" 2>&1
  log "deepprobe mml=$MML"; timeout 5400 python3 "$S/deepprobe.py" "$URL" "$MODEL" "$MML" 4 > "$S/deep-$TAG-$CFG.log" 2>&1; tail -1 "$S/deep-$TAG-$CFG.log" >>"$L"
  log "perfprobe"; timeout 1200 python3 "$S/perfprobe.py" "$URL" "$MODEL" 4 > "$S/perf-$TAG-$CFG.log" 2>&1; cat "$S/perf-$TAG-$CFG.log" >>"$L"
  HE=$(grep -oE "humaneval pass@1 [0-9.]+% \([0-9/]+\)" "$L" | tail -1); ARC=$(grep -oE "arc ARC-Challenge [0-9.]+% \([0-9/]+" "$L" | tail -1); NE=$(grep -oE "✓ needle [0-9]+/[0-9]+" "$L" | tail -1); PF=$(grep -oE "perf peak [0-9.]+ · single [0-9.]+ tok/s" "$L" | tail -1)
  python3 - "$S/result-$TAG-$CFG.json" "$CFG" "$PID" "$KV" "$MML" "$KVLINE" "$CONC" "$BACKEND" "$HE" "$ARC" "$NE" "$PF" "$S/deep-$TAG-$CFG.log" "$S/perf-$TAG-$CFG.log" <<'PYX'
import sys, json
out, cfg, pid, kv, mml, kvline, conc, backend, he, arc, ne, pf, deep, perf = sys.argv[1:]
d = {"cfg": cfg, "placement": pid, "kv_cache_dtype": kv, "max_model_len": int(mml), "kv_pool": kvline, "max_concurrency": conc, "backend": backend,
     "humaneval": he, "arc200": arc, "needle": ne, "johnny_perf": pf}
try: d["deepprobe"] = json.loads(open(deep).read().strip().splitlines()[-1])
except Exception as e: d["deepprobe"] = f"n/a ({e})"
try: d["perfprobe"] = json.loads(open(perf).read().strip().splitlines()[-1])
except Exception as e: d["perfprobe"] = f"n/a ({e})"
json.dump(d, open(out, "w"), indent=1); print(json.dumps(d)[:600])
PYX
  log "result → $S/result-$TAG-$CFG.json"
  johnny down "$(seat_name)" >>"$L" 2>&1; sleep 5
done
log "SERIES_DONE $TAG"
