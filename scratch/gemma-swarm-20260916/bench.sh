#!/usr/bin/bash
# bench.sh <tag> <conc_per_seat> <in> <out> <container>...  — runs `vllm bench serve` inside each container
# in parallel against its own localhost:8000, then sums throughput across seats.
TAG=$1 C=$2 IN=$3 OUT=$4; shift 4
NP=${NP:-$((C*${MULT:-4}))}; R=results/$TAG; mkdir -p $R
for n in "$@"; do
  docker exec $n vllm bench serve --backend openai --base-url http://localhost:8000 --model ${SERVED:-gemma} \
    --tokenizer /models/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic --dataset-name random \
    --random-input-len $IN --random-output-len $OUT --num-prompts $NP --max-concurrency $C \
    --ignore-eos --seed $RANDOM > $R/$n-c$C-i$IN-o$OUT.txt 2>&1 &
done; wait
python3 - "$R" "$C" "$IN" "$OUT" "$@" <<'PY'
import re,sys
R,C,IN,OUT,*names=sys.argv[1:]
tot=dict(out=0,total=0,req=0); ttft=[];tpot=[];per=[];fail=0
for n in names:
    t=open(f"{R}/{n}-c{C}-i{IN}-o{OUT}.txt").read()
    g=lambda k:float(re.search(k+r"\s*:?\s+([\d.]+)",t).group(1)) if re.search(k+r"\s*:?\s+([\d.]+)",t) else float('nan')
    tot['out']+=g(r"Output token throughput \(tok/s\)"); tot['total']+=g(r"Total [Tt]oken throughput \(tok/s\)"); tot['req']+=g(r"Request throughput \(req/s\)")
    per.append(g(r"Request throughput \(req/s\)")); ttft.append(g(r"Median TTFT \(ms\)")); tpot.append(g(r"Median TPOT \(ms\)")); fail+=int(g(r"Failed requests") if 'Failed requests' in t else 0)
print(f"{R.split('/')[-1]:28s} seats={len(names)} c/seat={C:>4} in={IN} out={OUT} | out {tot['out']:8.0f} tok/s | total {tot['total']:8.0f} tok/s | {tot['req']:6.2f} req/s | TTFT med {max(ttft):7.0f} ms | TPOT med {max(tpot):6.1f} ms | failed {fail} | per-seat req/s {per}")
PY
