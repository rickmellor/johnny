# Qwen3.8-Flash-Next on specul8-o-matic — 2026-08-26 (speed-focused; quality benches abandoned per Rick)

## Model
Qwen/Qwen3.8-Flash-Next (arch `qwen4_exp`, Qwen4 preview): 125B MoE (48 layers, 512 experts × 640-wide, top-10+1 shared, hidden 2560), 6B active,
+ 51B n-gram "engram" PLE table (20M hashes, layer 2) + 4B MTP + ViT. 3-of-4 layers Gated DeltaNet, 4th = Qwen Sparse Attention (indexer top-2048).
262K native. Official FP8 = 172.8 GiB (174.5B fp8 + 5.5B bf16).

## Fit / engine verdict
- vLLM: only FP8 loadable; PLE offloadable to host (VLLM_PLE_CPU_OFFLOAD=1) but experts alone = 112 GiB fp8 → ~119 GiB GPU-resident vs 119 GiB physical. ~20 GiB short.
  Mainline v0.28.0 has no qwen4_exp; `vllm/vllm-openai-rocm:qwen38-flash-next` side-branch image pulled (95 GB) — AITER/MI355X recipe; gfx1201 QSA path unknown.
  Bridge candidates: `--cpu-offload-gb ~6/GPU` (UVA, ~300 MB/token PCIe → ≤60 t/s ceiling) — untested. Clean fix = INT4 expert quant (none published).
- llama.cpp: unsloth PR ggml-org#27742 (@035e227) implements PLE (host get_rows), QSA, vision; MTP WIP. Images built: johnny-llamacpp-qwen4exp:{gfx1201 (HIP, NO_VMM), vulkan}.
  Dockerfile fix needed (multi-source COPY trailing slash) — branch johnny-gfx1201 in ~/repos/llama.cpp-qwen4exp.
- Quant: unsloth UD-Q4_K_XL 103.7 GiB (experts Q4_K/Q5_K/Q5_1 ≈4.75 bpw = 71.7 GiB; PLE one 26.8 GiB IQ4_NL tensor on host; attn Q8_0). ~20-22 GB/GPU resident.

## Speed (4×R9700, HIP layer split, ctx 64K/16K)
| config | pp512 | tg (single) |
|---|---|---|
| 4 GPU, fa on | 1022 | 25.4 |
| 4 GPU, fa off | 686 | 25.5 |
| 3 GPU | 1133 | 25.5 |
| HIP graphs disabled | 1118 | 25.7 |
| -t 4 / -t 24 | 1116/1119 | 25.6/25.5 |
| Vulkan image | — | 20.7 |
| 4 slots concurrent (server) | — | 76.4 aggregate (19.1/stream) |
| + Qwen3.5-0.8B draft spec (n-max 8) | — | 30–37 (acceptance 0.77–0.82, mean len 6–7) |
- `-sm row/tensor`: not implemented for this arch (hybrid). PLE on ROCm0 (-ot per_layer_token_embd=ROCm0): load fails.
- Diagnosis: not launch-overhead (graphs/threads/GPU-count invariant) — 8-token verify costs ~5× a 1-token step, so cost scales with tokens: the expert
  `mul_mat_id` path reads 3.75 GB/token at ~94 GB/s effective (~15% of card bandwidth). 512 tiny 640-wide experts are a bad shape for llama.cpp's kernels on RDNA4.
  Same ~25 t/s floor Ornith-397B (17B active) showed on this stack.
- Reference: Qwen3.8-27B-FP8 vLLM TP4 = 39 t/s single / 141 peak; gemma-4-26B TP4 ≈ 100 t/s. Rick's bar: ~100 t/s. Verdict: llama.cpp cannot get there; parity at best with draft spec.

## Paths to ≥2× (not done)
1. vLLM TP4 with an INT4 expert quant (wait for AWQ/GPTQ/W4A16 upload, or llm-compressor from the 335 GB bf16 — hours, day-0 arch risk).
2. vLLM FP8 + PLE offload + --cpu-offload-gb on the qwen38-flash-next ROCm image (30–40% odds it runs on gfx1201; ~1–2 h).
3. llama.cpp: PR's MTP (draft-mtp) when it lands; upstream MoE kernel work for tiny experts.
