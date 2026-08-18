"""multi_gpu_env: box-correctness env for multi-GPU vLLM launches, and the
version gate for the RCCL P2P workaround (gfx1201, vLLM-ROCm > v0.20.x).
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


def test_old_image_keeps_p2p_enabled():
    env = multi_gpu_env(2, "vllm/vllm-openai-rocm:v0.20.2")
    assert env["NCCL_PROTO"] == "Simple"
    assert "NCCL_P2P_DISABLE" not in env
    assert "RCCL_NET" not in env


def test_new_image_gets_rccl_workaround():
    env = multi_gpu_env(2, "vllm/vllm-openai-rocm:v0.25.1")
    assert env["NCCL_P2P_DISABLE"] == "1"
    assert env["RCCL_NET"] == "Socket"
    assert env["NCCL_PROTO"] == "Simple"


def test_unparseable_tag_gets_rccl_workaround():
    for image in ("my-custom-vllm:latest", "vllm/vllm-openai-rocm:qwen38", None):
        env = multi_gpu_env(4, image)
        assert env["NCCL_P2P_DISABLE"] == "1"
        assert env["RCCL_NET"] == "Socket"
