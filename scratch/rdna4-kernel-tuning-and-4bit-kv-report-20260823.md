# RDNA4 (gfx1201 / R9700) — vLLM kernel tuning + 4-bit KV-cache experiments

Overnight run 2026-08-23 (≈00:40 → 07:30) on specul8-o-matic (4× R9700, vLLM-ROCm 0.27.1 default,
Gemma-4 on the 2026-08-22 nightly where 0.27.x can't load it). Everything below was measured with johnny's
own harness (`johnny bench --suite perf|humaneval|arc|needle`) plus two small probes committed under
`scratch/` (`kvexp/deepprobe.py` = ctxsafe-equivalent against a running seat, `kvexp/perfprobe.py` =
single-stream + 4-way aggregate tok/s), because `johnny bench perf` can't ramp seats capped at
`max_num_seqs=4` and `ctxsafe` needs free GPUs.

**TL;DR**
1. **Kernel tuning: marginal.** +2–4 % on the dense block-FP8 Qwen seats, ~0 on gemma's MoE. Decode on
   this card is memory-bound; vLLM's untuned Triton defaults were already close. Keep the derived images
   (`johnny-vllm-rocm:{v0.27.1,nightly0822,v0.20.2}-gfx1201`) as a free few percent — don't expect more.
2. **4-bit KV: not usable for serving on this platform today.** `int4_per_token_head` gives the capacity
   (coder **4.74×** and gemma **7.97×** concurrency at the full 262 144 window on TP2 — the 262K@4 target on
   two GPUs) with HumanEval/needle intact, but decode is **~3.7× slower**, long-prompt prefill **5–7× slower**,
   the coder's ARC collapsed (18 % at the 1 h cap vs 85 %), and both engines **died at the ~200K probe**.
   `turboquant_4bit_nc` is **rejected by the backend selector on gemma** and **crashes a TP worker on the
   first request** on the coder. Stay on bf16 KV.
3. **Incident:** the int4 prefill kernels on GPU3 — which also drives the desktop — starved its graphics
   ring: **232 `ring gfx_0.0.0 timeout` → GPU resets between 02:53 and 04:24** ("device wedged, but recovered
   through reset"). After that window every **multi-GPU** seat decodes at ~½ speed (TP4 Qwen3.8 39 → 16 tok/s,
   TP2 coder 32 → 17.5) while single-GPU, CPU, PCIe and VRAM all measure normal. **A reboot is the likely fix;
   I did not reboot your workstation unattended.** The fleet is up on the `qwen38-tp4` profile (bf16 KV, stock
   0.27.1) and will be slow until then.

---

## A. Kernel tuning

### What was tuned
vLLM's quantized GEMMs are Triton kernels with *no runtime autotuner*; tile parameters come from JSON files keyed
by `(N, K, device_name, dtype, block_shape)`. The 0.27.1 image ships **219** such files — MI300X/MI325X/H100/H200/
A100/L40S — and **none for any Radeon/gfx1201**, so every FP8 GEMM on this box ran on one generic default
(`Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!` ×20 at startup).

| target | tool | shapes / space | time |
|---|---|---|---|
| W8A8 block-FP8 GEMM (Qwen3.8-27B-FP8 **and** qwen-27b-coder = Qwen3.6-27B-FP8: identical dims, hidden 5120 / inter 17408 / 24q·4kv·hd256) | `benchmark_w8a8_block_fp8.py` (in-image), one proc per GPU | 10 shapes = TP2 + TP4 sets; 13 batch buckets; space pruned to 192 (`num_stages ∈ {2,3}`, `GROUP_SIZE_M ∈ {1,16}`, tiles ≤128 — 256-wide tiles compile for **minutes** on gfx1201 and never win) | ~35 min (first bucket compiles, later buckets reuse the Triton cache) |
| fused-MoE (gemma-4-26B-A4B: E=128, top-8, moe_inter 704, hidden 2816, FP8 per-channel) | `benchmark_moe.py` — **Ray-free wrapper** (`tune/tune_moe_gemma.py`): Ray (pip-installed in-container) crashed its core worker on both images; the stock parser also doesn't know Gemma-4's `num_experts/top_k_experts/moe_intermediate_size` keys | E=128,N=352 (TP2); 7 buckets; 432-config RDNA prune | ~25 min per image (0.20.2 + nightly) |

Packaging: block-FP8 JSONs **must live inside the image** (`fp8_utils.get_w8a8_block_fp8_configs` only looks in the
package `configs/` dir; `VLLM_TUNED_CONFIG_FOLDER` is honored by fused-MoE only) → three derived images
`johnny-vllm-rocm:<tag>-gfx1201` = base + JSONs. Registry placements can pin them via `image:`.

### Measured benefit (same knobs, `johnny bench --suite perf`)
| seat | untuned | tuned | Δ |
|---|---|---|---|
| qwen-27b-coder TP2, 0.27.1 | 900 peak / 30.4 single | **918 / 31.5** | +2 % / +3.6 % |
| gemma-4-26B TP2, 0.20.2 (+ tuned MoE) | 2045 / 94.9 | 1931 / 100.0 | −6 % / +5 % (noise-level, mixed) |
| gemma-4-26B TP2, nightly (+ tuned MoE; config confirmed loading) | 1788 / 81.0 | 1769 / 81.4 | ~0 |
| Qwen3.8-27B TP4 @262K | 39.1 single / 141 c4 (23:50) | 17 / 68 (06:00) | **not attributable** — stock image measured 16 / 59 minutes later (see Incident) |

**Verdict:** the hardware does FP8 natively (measured 222 TFLOP/s FP8 vs 118 bf16 on a 4096³ GEMM) and the
tuned tiles are a real but small win on the dense path; the MoE expert GEMMs (N=352) are too small for tiling to
matter. Decode here is bandwidth-bound, so the GEMM config is a second-order effect. Upstream contribution of
the JSONs is possible (nobody has RDNA4 configs) but they buy little.

## B. 4-bit KV experiments (TP2, target 262 144 ctx @ concurrency 4)

Images: coder on `v0.27.1-gfx1201`, gemma on `nightly0822-gfx1201`. Both seats `gmu 0.93`, `max_num_seqs 4`,
`bt 8192` for the 4-bit configs; bf16 rows are the production knobs on the tuned images.

| seat / kv dtype | mml | KV pool → conc. @262K | backend | HumanEval | ARC-200 | needle | deep probe (needle recall, single → 4-way) | decode |
|---|---|---|---|---|---|---|---|---|
| coder bf16 | 95 417 | 239 k → n/a (2.51× @95K) | ROCM_ATTN | **95.73 %** | **85.0 %** | **16/16** | 8K 3 s · 32K 17 s · 64K 46 s · 89K 67 s; 4×89K 4/4 in 299 s | 32.0 single / 118.7 agg |
| coder **int4_per_token_head** | 262 144 | **1 243 069 → 4.74×** | TRITON_ATTN | 95.12 % (−1) | **18 % at 100/200, timed out at 1 h** | 15/16 | 8K 7.5 s · 32K 79 s · 64K 315 s · 128K **1304 s** ✓; **200K → HTTP 500, engine dead** | ~31 agg w/ 4 streams (≈8/stream) |
| coder turboquant_4bit_nc | 262 144 | 1 250 882 → 4.77× | TURBOQUANT | — | — | — | — | **worker process died on first request** (`Worker proc VllmWorker-1 died unexpectedly`) |
| gemma bf16 (nightly) | 110 832 | 309 k → 2.79× @110K | TRITON_ATTN (forced for Gemma-4 hetero head dims) | 95.12 % | 95.0 % | 15/16 | 8K 1.6 s · 32K 15 s · 64K 54 s · 104K 134 s; 4×104K 4/4 in 452 s | 86.9 single / 276 agg |
| gemma **int4_per_token_head** | 262 144 | **2 089 286 → 7.97×** | TRITON_ATTN | 95.12 % (=) | 93.5 % (−1.5) | 15/16 (=) | 8K 5.5 s · 32K 63 s · 64K 237 s · 128K **908 s** ✓; **200K → HTTP 500, engine dead** | n/a (engine dead before probe) |
| gemma turboquant_4bit_nc | — | — | — | — | — | — | — | **won't start:** `No valid attention backend found for rocm … kv_cache_dtype=turboquant_4bit_nc` |

Why it behaves this way: on RDNA the attention kernels never do an FP8/INT4 dot — `chunked_prefill_paged_decode`
(ROCM_ATTN) and the int4 kernel both **dequantize K/V to bf16 in the hot loop** (`(K_load.to(fp32)*k_scale).to(Q.dtype)`,
then `tl.dot` in bf16). 4-bit adds nibble unpack + scale on every K/V load in a memory-bound phase, so prefill
(quadratic in depth) and decode both lose; CDNA gets AITER kernels with true low-bit paths, gfx1201 doesn't.
Quality: short-answer tasks survive (HumanEval, needle); the coder's long-CoT ARC collapsed under int4.

**Verdict:** bf16 KV stays. If you want 262K@4 on two cards you'd need an RDNA-aware attention kernel with a
real int4/fp8 dot — same "nobody did the RDNA4 work upstream" story as the configs. The TP4 Qwen3.8 seat
already gives 262K@4.5× with bf16 (1.18 M-token pool) because only 16 of its 64 layers carry per-token KV.

## C. Incident — GPU ring resets and the multi-GPU slowdown

- 02:53–04:24: **232 × `amdgpu 0000:03:00.0: ring gfx_0.0.0 timeout … Starting gfx_0.0.0 ring reset … device
  wedged, but recovered through reset`**, process `gnome-shell`. 0000:03:00.0 = johnny GPU3 = the card that
  drives the desktop. The gemma int4 seat was on GPUs 2,3 in that window; its long Triton prefill kernels
  starved the display ring (default ~4 s timeout). Three engines also died hard overnight (int4 200K ×2, TQ).
- After that window, every multi-GPU seat decodes at ~½ speed regardless of model, GPU pair, or image:
  TP4 Qwen3.8 39.1 → 15.7–17 tok/s (stock and tuned), TP2 Qwen3.8 (GPUs 0,1) 29.8 → 15.4, TP2 coder (GPUs 2,3)
  32.0 → 17.5. **TP1 gemma-12B: 37.3 vs 36.8 recorded — normal.** Raw per-GPU: bf16 GEMM 128 TFLOP/s, VRAM
  544–552 GB/s, PCIe h2d/d2h 28 GB/s, links x16 Gen5, clocks ramp to 3.39 GHz sclk / 1.26 GHz mclk under
  load, CPU all-core 4.19 GHz, no RAS errors, no other GPU tenants, KV pools normal. 2-GPU all-reduce on GPUs 0,1:
  68 µs @64 KiB, 148 µs @1 MiB, 1.5 ms @16 MiB (no pre-incident baseline).
- So the loss is confined to the **cross-GPU (host-bounce RCCL) path**, consistent with driver/KFD state
  left behind by the resets. **Recommendation: reboot, then re-run `kvexp/perfprobe.py` on the TP4 seat — expect
  ~39 single / ~141 c4.** Lessons: (1) don't put long-kernel experiments on the display GPU (or raise the
  gfx timeout / move the console off GPU3); (2) johnny should refuse TP placements that include the display
  GPU for "research" seats, or at least warn.

## D. State left for the morning
- Fleet: `johnny profile up qwen38-tp4` (Qwen3.8 TP4 262K@4.5× bf16, stock v0.27.1, serving chat + coder via
  `role_aliases`; nomic; classifier). Daily profile untouched. Note the speed caveat above until reboot.
- Registry: experiment placements `kvexp-*` (coder/gemma: bf16-tuned, int4, tq) carry their quality blocks;
  `ab-gemma-v0202-tunedmoe`, `tp4-c4-tuned-…`, `nightly0822-*` clones. Safe to prune the int4/tq ones.
- Images: `johnny-vllm-rocm:{v0.27.1,nightly0822,v0.20.2}-gfx1201` (tuned configs). Tuning sources + JSONs in
  `scratch/tune/` (this commit), raw logs in the session scratchpad `tune/`, `kvexp/`.
- Not changed: `docker.vllm_image` (still stock v0.27.1); gemma pin (0.20.2); no profile switched to tuned images.
