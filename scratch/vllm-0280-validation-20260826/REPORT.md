# vLLM v0.28.0 validation on specul8-o-matic (4×R9700 gfx1201) — 2026-08-26

Image: vllm/vllm-openai-rocm:v0.28.0 (transformers 5.15, torch 2.12 / triton 3.7). RCCL env (NCCL_P2P_DISABLE=1 RCCL_NET=Socket NCCL_PROTO=Simple) still required — rccl PR #2187/#2166 open.
All seats TP2, GDN seats warmed (33K prefill) before `johnny bench perf`.

| seat | baseline | v0.28.0 | v0.28.0 + AITER |
|---|---|---|---|
| gemma-4-26B-A4B TP2 (110K) | 0.20.2: 1943 / 99.9 | 1753 / 80.7 | 1745 / 80.5 |
| qwen-27b-coder TP2 (95K, bf16 KV) | 0.27.1: 900 / 30.4 (tuned 918/31.5) | 903 / 30.8 | ✗ LDS overflow at deep prefill |
| qwen-27b-coder TP2 fp8 KV | 0.20.2: 654 / 14.9 | 887 / 29.9 (33K prefill 31.9 s) | **911 / 30.8, 33K prefill 14.7 s, KV 0.54M tok** |
| Qwen3.8-27B-FP8 TP2 (95K) | 0.27.1: 577 / 29.8 | 596 / 30.3 | ✗ LDS overflow at deep prefill |

Findings
- Gemma-4 loads on 0.28.0 (PR #49797) but the nightly's regression persists (−19% single / −10% peak). Keep the per-placement v0.20.2 pin.
- Dense/GDN Qwen seats: parity or slightly better; tool calls (qwen3_xml) OK.
- PR #43615 (AITER + FP8 on gfx120x): the AITER unified-attention prefill kernel (`kernel_unified_attention_3d`) overflows gfx1201's 64 KB LDS (Triton OutOfResources 65792 > 65536) with **bf16 KV** on prompts ≥ ~8K → EngineDeadError. With **fp8 KV** it runs clean and is the fastest prefill measured.
- fp8 KV no longer carries the 0.20.2 dequant penalty: decode parity, 2× KV capacity; with AITER, prefill 15% faster than bf16.
- johnny bug: `bench perf` on a dead seat recorded 304233 tok/s from HTTP 500s — should fail the run.

Crash logs: qwen38-aiter-crash.log, coder-aiter-bf16kv-crash.log. Registry placements: v0280-* on the three models.
Recommendation: move johnny `docker.vllm_image` default to v0.28.0 (gemma keeps its v0.20.2 pin); adopt `kv_cache_dtype: fp8` + `VLLM_ROCM_USE_AITER=1` for Qwen seats that want context (validated on coder TP2; re-run on Qwen3.8 + TP4 before promoting).

## Adoption (2026-08-26 evening, per Rick)
- johnny `docker.vllm_image` → `vllm/vllm-openai-rocm:v0.28.0` (config.yaml + code default, johnny commit 2fd132e). `cpu_image` left at 0.27.1 (nomic vectors not re-validated).
- Profiles: `daily`/`standard` coder → `v0280-aiter-kvfp8-tp2-gmu0.92-seqs64-bt16384-mml95417` (911/30.8, fp8 KV + AITER);
  `qwen38-tp4` chat → `effort-low-tp4-kvfp8-aiter` (TP4 262K: 43.1 single / 130 agg; bf16 twin 43.0 / 150; needle clean to 256K + 4×256K; tool calls OK).
- Gemma stays `gemma-tp4-c4-mml262144-v0202` on v0.20.2 (default profile gemma-tp4 unchanged).
- Gotcha logged: a TP4 seat read 16.6 t/s after 4×256K probes + a timed-out `bench perf`; fresh launch = 43.1. `bench perf` times out on 262K TP4 seats — use scratch/kvexp/perfprobe.py.
