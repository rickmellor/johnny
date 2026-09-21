# Best coder for the Intel Arc Pro B50 (16 GB, 224 GB/s, 70 W) — 2026-09-20/21

**Pick: Qwen3.6-35B-A3B, pure K-quant Q2_K (bartowski, 12.6 GiB), llama.cpp SYCL, flash attention ON, fp16 KV, 131K context.**
Launch: `~/projects/infrastructure/tools/b50-coder.sh up` (port 8124, model alias `coder`, thinking off by default).
Needs `sudo tools/b50-rebar.sh` once after each boot.

## Why
| candidate (backend) | decode @0 | decode @~20K | decode @~35–40K | decode @~99K | max ctx | HumanEval* | hard-20 |
|---|---|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B Q2_K (SYCL, FA on, f16 KV)** | **44** | **36** | **37** | **28** | **131K (262K w/ q8 KV: 34 / – / 19)** | 91.5 % | 14–15 |
| Qwen3.6-35B-A3B UD-IQ3_S (Vulkan, Mesa 26) | 41 | 29–32 @16K | 21–27 @32K | 9 @120K (q8, FA) | 120K fits, slow | – | 13 |
| Qwen3-Coder-30B-A3B UD-Q3_K_XL (Vulkan, Mesa 26) | 49 | 13 @16K | 8 @30K | – | ~32K (q8 KV) | **95.1 %** | **15** |
| Qwen3-Coder-30B-A3B Q3_K_M (Vulkan, Mesa 26) | 53 | – | – | – | ~16K | – | 12 |
| Qwen3-Coder-30B-A3B Q3_K_S (SYCL, q8 KV) | 29–32 | 17 | 12 | – | ~40K | – | – |
| gpt-oss-20b MXFP4 | 17 (Vulkan) / 21 (SYCL) | | | | | | |
*HumanEval rescored with the prompt's own `from typing import …` lines restored (rescore.py): the 35B omits them and the stock
scorer counted 12 correct answers as NameErrors (raw 84.1 %).

Prefill: SYCL 35B ≈ 490–505 tok/s flat to 40K, 455 at 99K. Vulkan 35B: 590 @0 → 320 @32K (FA off) / 109 (FA on).
The 35B wins because it keeps only **10 KV layers of 40** (20 KiB/token vs 96 KiB for Qwen3-Coder's 48), so context is cheap in
both memory and attention time; Qwen3-Coder is the slightly better pure-function coder but collapses with depth on this card.

## Stack findings
1. **Mesa matters on Vulkan**: container with kisak Mesa 26.2.3 vs Ubuntu's 25.2.8 = +25–75 % (Qwen3-Coder tg 37 → 46–53; 35B 27 → 46). Image `prism-vk-mesa-new`.
2. **Vulkan flash attention is slow at depth on ANV**; FA off + f16 KV doubles prefill at 16–32K but its scratch buffer OOMs by 64K.
3. **SYCL (`ghcr.io/ggml-org/llama.cpp:server-intel`) is flat at depth with FA on**, but only has fast kernels for Q4_0/Q8_0/K-quants → pure K files only (check the tensor histogram; Unsloth "UD" files are mostly IQ types).
4. q8_0 KV on SYCL costs decode at depth (19 vs 28 tok/s at 99K) — use f16 unless 262K is needed.
5. Thinking mode is impractical here: 9–17K reasoning tokens per hard task with no answer inside a 20K budget.
6. Needs the 16 GB BAR fix (`tools/b50-rebar.sh`) for SYCL to see the card at all.
Hard set = hard.py (20 tasks, hidden tests validated against reference solutions); ±2 tasks is noise.

## Four-way matrix (2026-09-21) — same 20-task hard set, same HumanEval scorer (rescore.py), thinking off, temperature 0
| | Bonsai 2 27B PQ2_0 (RTX 4080) | Qwen3.6-35B-A3B Q2_K (B50, SYCL) | Qwen3.8-27B-FP8 (2× R9700, vLLM TP2) | Flash-Next AWQ (4× R9700, vLLM TP4+EP) |
|---|---|---|---|---|
| hard-20 | 13 | 14–15 | **17** | **18** |
| ARC-Challenge 200 | 96.0 % | **97.0 %** | 95.0 % | **97.0 %** |
| HumanEval | 92.1 % | 91.5 % | **96.3 %** | 95.1 % |
| decode short | 67–72 tok/s | 44 | 27–30 (TP4+MTP: 52–59) | 40–49 |
| weights | 7.2 GB | 12.6 GiB | 31 GB | 129 GB |
Same base model at 8 bits vs ternary: hard-20 17 → 13, HumanEval 96.3 → 92.1. The 27B at 8 bits is within noise of Flash-Next on code.
