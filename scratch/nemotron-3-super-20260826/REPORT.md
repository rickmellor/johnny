# Nemotron-3-Super-120B-A12B (cyankiwi AWQ-4bit) on specul8-o-matic — 2026-08-27

Registry `nemotron-3-super-awq`; weights /mnt/data/models/cyankiwi/… (75 GiB, mirrored to /mnt/ug-models/cyankiwi/). vLLM v0.28.0, TP4, RCCL env.
Arch nemotron_h: 88 layers = 40 Mamba-2 + 8 attention (2 KV heads) + 40 latent MoE (512 experts, top-22). KV = 8 KB/token bf16; Mamba state ~170 MB/seq.

| placement | result |
|---|---|
| manual-tp4-awq-mml131072-kvfp8 (no MTP) | loads (Mamba2 SSD Triton, triton SSU backend); KV pool 2.13M tokens (16x @128K); **40.1 t/s single / 124.3 agg@4**; reasoning split nemotron_v3 + qwen3_xml tools OK; needle 8.6K/34K/69K/123K single (TTFT 7.6/12.5/23/49 s) + 4x123K concurrent, 0 errors; perf unchanged after deep probes |
| manual-tp4-awq-mml131072-kvfp8-mtp2 | serves but **MTP broken: 4140 drafts / 0 accepted → 16.0 single / 47 agg**. AWQ quantized the MTP head (weight_packed) despite the card; NemotronH MTP has known acceptance bugs (sglang #21138). Don't use. |

Quality (reasoning on — harness `enable_thinking:false` is Qwen-specific; treat as lower bounds): **HumanEval 82.32% (135/164)**, **ARC-200 91.5% (183/200, 12 no-extraction)**.
Reference qwen-122b-awq-mtp: ARC 87.0 (limit 100), HumanEval 93.29, 82.6 t/s single with MTP.

Verdict: runs cleanly on RDNA4/v0.28.0 — the first Mamba-2 hybrid on the box. Speed = the Qwen3.8-27B seat (40), half the 122B-with-MTP; context effectively free. Reasoning bench above the 122B, coding below (likely partly reasoning/max_tokens truncation — a `/no_think`-style rerun or larger gen budget would firm it up). Role candidate: reasoning/agentic escalation seat with huge context; not a speed upgrade.
Gotchas: deepprobe.py CH_PER_TOK is Qwen-calibrated (Nemotron ~7.5% denser → pass mml*0.93); `--mamba-ssm-cache-dtype float32` used per NVIDIA recipe.
