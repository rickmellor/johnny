# Gemma 4 26B max-throughput experiment — 6× R9700, 4K context (2026-09-16/17)

**Goal (Rick):** absolute peak serving throughput for standard `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`
on all six R9700s, short context (max_model_len 4096), workload **2800 tokens in / 250 out**.
Opposite direction from the daily seat (single-user speed at 262K).

## Result

**Winner: three identical seats, one per GPU pair, each `--data-parallel-size 2 --enable-expert-parallel`
(DP attention ×2 + EP MoE), vLLM v0.20.2, bf16 KV, KV pool pinned.**

| config (whole box, long runs, 2800/250) | req/s | total tok/s | out tok/s | TPOT med | TTFT med |
|---|---|---|---|---|---|
| **DP2+EP2 ×3, c=30/seat (final, pinned KV)** | **6.75** | **20 596** | **1 688** | 46 ms | 2.8 s |
| same, via `johnny profile up gemma-swarm` | 6.71 | 20 464 | 1 677 | 46 ms | 2.4 s |
| DP2+EP2 ×3, c=28/seat | 6.51 | 19 860 | 1 628 | 44 ms | 2.4 s |
| TP2 ×3, c=32/seat (previous best practice) | 4.18 | 12 740 | 1 044 | 78 ms | 3.6 s |
| TP2+EP ×3 (`-tp 2 --enable-expert-parallel`) | 3.94 | 12 022 | 985 | 81 ms | 3.6 s |
| DP2 (no EP) ×3 | 4.98 | 15 182 | 1 244 | 63 ms | 4.5 s (uneven seats) |
| DP3+EP3 ×2, c=56/seat | 3.54 | 10 792 | 885 | 106 ms | 5.8 s |
| DP4+EP4 (4 GPUs only, short run) | 2.26 | 6 890 | 565 | 203 ms | 5.1 s |
| DP6+EP6 ×1, c=96 | 2.71 | 8 278 | 679 | 109 ms | 7.7 s |
| DP2+EP2 ×3 on nightly-27a94d1 | 3.93 | 12 013 | 985 | 65 ms | 4.5 s |

+61 % over TP2×3. ≈ 583 M tokens/day processed at this mix.

### The recipe (registry placement `swarm-dp2ep2-mml4096-bt8192-blk15200-v0202`, profile `gemma-swarm`)
```
image vllm/vllm-openai-rocm:v0.20.2      env HSA_ENABLE_IPC_MODE_LEGACY=0 NCCL_PROTO=Simple
--tensor-parallel-size 1 --data-parallel-size 2 --enable-expert-parallel
--max-model-len 4096 --max-num-seqs 256 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95
--num-gpu-blocks-override 15200 --limit-mm-per-prompt '{"image":0}'
```
Ports 8002/8003/8004 (GPU pairs HIP 0,1 / 2,3 / 4,5). **The client must spread load across the three
endpoints** (SAINT only knows :8002 as `chat`); keep ≤ ~30 in flight per seat (KV ≈ 88K tokens/seat ≈ 29 full
2800+250 requests). Beyond that requests just queue — no harm, no gain.

## What was learned

1. **TP1 ×6 is not viable with FP8 weights.** Weights are 25.75–27.25 GiB of a 31.9 GiB card → KV 11–16K tokens
   (≈4 full requests/card) and only 31 tok/s single-stream / 2 650 tok/s prefill per card. A ~15 GiB 4-bit
   checkpoint would fit, but prefill here is compute-bound and FP8 is the fast GEMM on RDNA4 (222 vs 118 TFLOP/s),
   so it was not pursued.
2. **KV is expensive at 4K context:** ~236 KB/token (the sliding-window savings that give the 262K seat 32 KB/token
   don't apply when max_model_len ≈ window + batch budget). A TP2 pair holds ~113K tokens = 36 requests.
3. **The workload is ~92 % prefill; the box is compute-bound.** Per pair: prefill-only ≈ 6.0–7.3K tok/s, mixed
   ≈ 4.3K tok/s under TP2. max_num_batched_tokens (4096/8192/16384) made no difference under TP2.
4. **DP attention + EP MoE on a pair removes the per-layer TP all-reduce** and is the big win (+61 %). It only
   pays at EP2: all2all over PCIe kills EP3/4/6 (per-GPU throughput 1.1 → 0.59 → 0.57 → 0.45 req/s).
5. **Free-VRAM cliff (important for reliability).** A DP2+EP2 seat is bimodal: ~2.2 req/s or ~1.35 req/s for its
   whole life. Cause = too little free VRAM after KV allocation (EP allgather buffers aren't in vLLM's profile
   run). Pools of 39–44K tokens/rank are always fast; 47K+ always slow; gmu 0.97 always slow; unpinned gmu 0.95
   is a coin-flip (pool lands 44–49K depending on launch timing). Zero preemptions in both modes.
   **Fix = `--num-gpu-blocks-override 15200`** (44 061 tokens/rank, deterministic). 3/3 seats fast in every
   pinned launch. (Note the unit: 15 200 blocks ≈ 44K tokens because of the hybrid KV groups — 2 750 blocks is only 8K.)
6. **Dead ends, measured:** v0.28.0 + fp8 KV + AITER = 940 tok/s then `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
   (worker died); nightly + fp8 KV = 0.77 req/s/pair (Triton attention dequantizes fp8 KV); v0.28.0/nightly bf16 =
   faster pure prefill (+10–20 %) but identical mixed TP2 throughput and *much* slower DP+EP (3.9 vs 6.5);
   pipeline-parallel on v0.20.2 crashes in torch.compile (`IntermediateTensors has no attribute get`);
   BT < 8192 under DP+EP is slow; tuned fused-MoE config for E=64,N=704 (tuned-moe-ep/) = no measurable gain;
   PCIe-topology-aligned pairs (HIP 1,2 / 3,4 / 0,5 share root complexes) = no difference.
7. **Short benchmarks lie here.** 4×concurrency prompts (~50 s) under-read by 10–40 % and hid the bimodality;
   all headline numbers are 10×concurrency prompts (2–3 min) with per-seat breakdown.
8. **Incident:** killing a hung MoE-tuner kernel (M=4096 bucket, GPU at 03:00.0 = HIP 5) caused one
   `amdgpu mode1 reset … device wedged, but recovered`. Subsequent runs show that GPU's seat at full speed — no
   reboot needed, but it is the first thing to suspect if HIP 4,5 ever under-performs before the next power cycle.

## Untested / next levers
- **Prefix caching**: benchmarks used unique random prompts (worst case). If the real 2800-token inputs share a
  system prompt/template, `--enable-prefix-caching` (on in the johnny placement) will raise req/s substantially.
- A front-end balancer (or SAINT multi-endpoint role) so clients see one URL.
- CPU: load average ~20 of 24 threads during the six-rank runs (DP engines busy-poll). More GPUs will want more host CPU.
- 8× R9700: the recipe scales linearly by pairs (4 seats ≈ 9 req/s); wider EP does not.

## Files
`launch.sh` (seat launcher) · `bench.sh` (parallel `vllm bench serve`, sums seats) · `box.sh`/`box3.sh` (whole-box runs) ·
`results/summary.txt` (every measurement) · `logs/` (engine logs per config) · `tuned-moe-ep/` (E=64,N=704 Triton config).
Workload generator: `vllm bench serve --dataset-name random --random-input-len 2800 --random-output-len 250 --ignore-eos`.
HIP index → PCI: 0=63:00 1=43:00 2=46:00 3=23:00 4=26:00 5=03:00.

---

# Addendum 2026-09-17 (later): latency, a degradation incident, and MTP

## End-to-end latency of the no-MTP winner (`gemma-swarm`, 30 in flight/seat)
Random prompts: TTFT p50 2.1–2.6 s / p99 5.7–6.3 s, TPOT p50 46 ms, **E2E p50 13.5 s · p90 14.7 s · p99 16.3 s** (healthy seat).
Real-text prompts (below): 2.06–2.16 req/s/seat, E2E p50 13.6–13.8 s · p90 14.3–16.0 s · p99 16.6–18.6 s.
Isolated ceilings on the live seats: prefill-only 46.2K tok/s box (15.4K/seat, 0 % prefix-cache hits); decode-heavy (64-in/250-out) 2.7–4.4K out tok/s.

## OPEN ISSUE — seats can degrade in place
Two of the three johnny-launched `gemma-swarm` seats dropped from ~2.2 to ~1.55 req/s (TPOT 46 → 68 ms, E2E p50 19 s) some time after
launch (seen ~1.5 h in, after a prefill-only and a decode-heavy burst). Same VRAM use as the healthy seat, no GTT spill, no kernel
messages, balanced DP ranks, GPUs at ~190 W instead of ~255 W (starved, not throttled), slow even with the other seats idle.
A deliberate burst sequence on fresh seats (mixed → prefill burst → mixed → decode burst → mixed; pools 15200/13000/15200+expandable
segments) did NOT reproduce it. Cause unknown; a seat restart fixes it. **Watch TPOT: ~45 ms healthy, ~66 ms degraded.**

## MTP (Gemma-4 assistant drafter) — `google/gemma-4-26B-A4B-it-assistant`, 832 MB, local + NAS
Needs vLLM ≥ v0.28.0 (`gemma4_mtp`; v0.20.2 can't). Flag: `--speculative-config '{"model":"/models/google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4,"moe_backend":"triton"}'`.
Benchmarked on **real text** (random-token prompts give meaningless acceptance): `dataset/realtext-*.jsonl`, 864 chat-templated prompts of
2783–2793 tokens (Python stdlib source + local markdown, explain/summarize tasks), 250 forced output tokens, 288 prompts/seat/pass,
client in a separate container (`rt_bench.sh`). Acceptance length: **3.07** tokens/step at MTP 4 (2.77 at 3, 3.28 at 5).

| config (per seat unless noted) | in flight | req/s | E2E p50 | E2E p90 | E2E p99 |
|---|---|---|---|---|---|
| **no-MTP DP2+EP2 v0.20.2 (baseline, same prompts)** | 30 | 2.06–2.16 | 13.6–13.8 s | 14.3–16.0 s | 16.6–18.6 s |
| TP2 + MTP 4, v0.28.0 — all 3 seats loaded (johnny profile) | 24 | 1.90–1.95 (**box 5.77**) | 12.1–12.5 s | 14.2–14.8 s | 16.4–18.7 s |
| TP2 + MTP 4, v0.28.0 — all 3 seats loaded (gmu 0.97) | 24 | 1.88–2.21 (box 6.01) | 10.7–12.5 s | 12.3–14.6 s | 15.6–17.7 s |
| same | 32 | 1.87–2.10 (box 5.86) | 13.5–15.5 s | 27–29 s | 35–36 s (KV overcommitted) |
| same | 12 | 1.46–1.85 (box 4.80) | 6.4–8.1 s | 7.3–9.0 s | 8.6–10.7 s |
| TP2 + MTP 4, v0.28.0 — seat with lightly loaded neighbours | 24 / 32 | 2.24 / 2.39 | 10.6 / 13.2 s | 12.3 / 15.3 s | 15.4 / 20.2 s |
| TP2 + MTP 4 low-concurrency sweep | 4 / 6 / 8 / 12 / 16 / 20 | 0.69 / 0.91 / 1.11 / 1.85 / 1.91 / 1.83 | 5.7 / 6.5 / 6.8 / 6.4 / 8.0 / 10.7 s | | 6.7 / 8.0 / — / 8.4 / 12.8 / 14.0 s |
| TP2 + MTP 3 | 12 / 16 / 24 | 1.84 / 1.83 / 2.26 | 6.4 / 7.8 / 10.6 s | | |
| TP2 + MTP 5 | 12 / 16 / 24 | 1.46 / 1.55 / 1.92 | 8.1 / 9.3 / 12.3 s | | |
| TP2 + MTP 4 on nightly-27a94d1 | 12 / 16 / 20 / 24 | 1.34 / 1.44 / 1.51 / 1.54 | 8.9 / 11.0 / 13.0 / 15.3 s | | |
| DP2+EP2 + MTP 4, v0.28.0 (KV only ~27K/rank) | 12 / 14 / 16 | 1.14 / 1.74 / 1.85 | 9.6 / 7.9 / 8.6 s | | |
| TP2 + MTP 4 with `--max-num-seqs 32` | 26 | 1.27–1.41 | 18 s | | |

Findings: (1) MTP lifts TP2 from ~1.4 to ~1.9–2.4 req/s/seat and makes throughput nearly flat from 12 to 28 in flight, so latency can
be halved (13.6 → 6.4 s) for ~25 % less throughput. (2) For pure box throughput the no-MTP DP2+EP2 recipe still wins (≈6.4–6.75 vs
5.8–6.0 req/s): MTP is only available on v0.28.0, where DP+EP is slow. (3) MTP 4 ≈ MTP 3 > MTP 5; v0.28.0 > nightly. (4) Never shrink
max_num_seqs with MTP (cudagraph capture sizes fall below 5 tokens/seq × batch). (5) With three MTP seats loaded the 12-core host is
~77 % busy (31 % sys; TP workers 180–270 % each) and a seat loses ~10–15 % vs running beside idle neighbours → partly host-CPU bound.
(6) First pass after launch is always slow/long-tailed (p99 30 s): warm seats before trusting them.

johnny: placement `swarm-tp2-mtp4-mml4096-bt8192-v0280`, profile **`gemma-swarm-mtp`** (same ports/roles as `gemma-swarm`).
