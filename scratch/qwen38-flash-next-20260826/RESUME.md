# Qwen3.8-Flash-Next eval — CLOSED 2026-08-26 (see REPORT.md); resume notes below are historical

## Done
- Weights: /home/rick/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL (103.7 GiB, sizes verified vs HF).
- NAS mirror (re-run if interrupted): `rsync -a --exclude .cache /home/rick/models/unsloth/Qwen3.8-Flash-Next-GGUF/ /mnt/ug-models/unsloth/Qwen3.8-Flash-Next-GGUF/`
- Engine source: ~/repos/llama.cpp-qwen4exp @ 035e227 (unslothai/llama.cpp branch qwen4exp/qwen3.8-flash-next, PR ggml-org#27742).
  Dockerfiles patched (COPY trailing slash): .devops/rocm-gfx1201.Dockerfile (adds -DGGML_HIP_NO_VMM=ON), .devops/vulkan.Dockerfile.
- Images: johnny-llamacpp-qwen4exp:gfx1201 (HIP, primary) / :vulkan (fallback). Rebuild if missing:
  `docker build -f .devops/rocm-gfx1201.Dockerfile --target server --build-arg ROCM_DOCKER_ARCH=gfx1201 -t johnny-llamacpp-qwen4exp:gfx1201 .`
  `docker build -f .devops/vulkan.Dockerfile --target server -t johnny-llamacpp-qwen4exp:vulkan .`
- johnny registry: model `Qwen3.8-Flash-Next-UD-Q4_K_XL`, placements manual-llama-{64kx1,262kx1,64kx4}-effort-low (backup: ~/.config/johnny/registry.yaml.bak-*-pre-flash).
- Baseline: Qwen3.8-27B-FP8 / effort-low-tp4 — HumanEval 93.29, ARC-200 79.5, needle 15/16, ICL 0/16, AutomationBench 40% (30 sales), PlanBench 57/71.4; perf 39.1 single / 141 peak (tp4-c4 placement).
- vllm/vllm-openai-rocm:qwen38-flash-next pulled (parked: FP8 can't fit 128 GB VRAM).

## Next
1. Free GPUs: `johnny down johnny-gemma-4-26B-A4B-it-FP8-Dynamic-8002` (or whatever profile came up at boot; keep nomic 8001 + classifier 8000).
2. `johnny up Qwen3.8-Flash-Next-UD-Q4_K_XL --placement manual-llama-64kx1-effort-low --port 8003 --wait`
   (or raw: ./flash-run.sh johnny-llamacpp-qwen4exp:gfx1201 65536 1 — container flash-test on :8003)
3. Check `docker logs`: PLE tensor (per_layer_token_embd, 26.8 GiB IQ4_NL) on CPU buffer, per-GPU VRAM, no -fa/QSA asserts. KV must stay f16.
4. Smoke: chat + tool call; `python3 ~/repos/johnny/scratch/kvexp/perfprobe.py http://127.0.0.1:8003 Qwen3.8-Flash-Next 4`
5. Bench: `johnny bench manual-llama-64kx1-effort-low --suite perf,humaneval,arc,needle --limit 200 --concurrency 1 --yes` (thinking off, like the baseline); then icl/planbench/automationbench (sales, --limit 30) if time.
6. Write-up in this dir, memory note, infra workstation.md + commit/push, restore gemma seat.
