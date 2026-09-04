"""multi_gpu_env: box-correctness env for multi-GPU vLLM launches, and the
P2P/IPC env that replaced the old RCCL P2P-off workaround (gfx1201, 2026-09-03).
derive_launch_extra: arch-keyed load-bearing launch flags (gfx1201 GDN/PIECEWISE)."""

from johnny.induct.report import derive_launch_extra
from johnny.induct.stages import multi_gpu_env


def test_qwen35_dense_gets_piecewise_graphs():
    extra = derive_launch_extra("Qwen3_5ForConditionalGeneration")
    assert extra["compilation_config"] == {"cudagraph_mode": "PIECEWISE"}


def test_other_archs_get_no_launch_extra():
    assert derive_launch_extra("Qwen3_5MoeForCausalLM") == {}
    assert derive_launch_extra("Gemma4UnifiedForConditionalGeneration") == {}
    assert derive_launch_extra(None) == {}


def test_single_gpu_gets_no_env():
    assert multi_gpu_env(1, "vllm/vllm-openai-rocm:v0.25.1") == {}
    assert multi_gpu_env(0, None) == {}


def test_multi_gpu_env_enables_p2p_on_every_image():
    # 2026-09-03: the "gfx1201 RCCL bug" was HSA_ENABLE_IPC_MODE_LEGACY=1 baked into
    # >=0.21 images; forcing it off restores P2P/IPC. No image gets the old P2P-off pair.
    for image in ("vllm/vllm-openai-rocm:v0.20.2", "vllm/vllm-openai-rocm:v0.25.1",
                  "vllm/vllm-openai-rocm:v0.28.0", "vllm/vllm-openai-rocm:nightly-27a94d1c",
                  "my-custom-vllm:latest", "vllm/vllm-openai-rocm:qwen38", None):
        env = multi_gpu_env(2, image)
        assert env["HSA_ENABLE_IPC_MODE_LEGACY"] == "0"
        assert env["NCCL_PROTO"] == "Simple"
        assert "NCCL_P2P_DISABLE" not in env
        assert "RCCL_NET" not in env
