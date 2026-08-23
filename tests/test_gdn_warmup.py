"""GDN warm-up: arch detection, prompt sizing, and launch/profile wiring."""
from johnny.engine import warmup


def test_needs_gdn_warmup_by_arch():
    gdn = {"identity": {"arch": "Qwen3_5ForConditionalGeneration"}}
    nxt = {"identity": {"arch": "Qwen3NextForCausalLM"}}
    gemma = {"identity": {"arch": "Gemma4ForConditionalGeneration"}}
    dense = {"identity": {"arch": "Qwen2ForCausalLM"}}
    assert warmup.needs_gdn_warmup(gdn)
    assert warmup.needs_gdn_warmup(nxt)
    assert not warmup.needs_gdn_warmup(gemma)
    assert not warmup.needs_gdn_warmup(dense)
    assert not warmup.needs_gdn_warmup({})


def test_needs_gdn_warmup_skips_non_vllm_backends():
    gdn = {"identity": {"arch": "Qwen3_5ForConditionalGeneration"}}
    assert not warmup.needs_gdn_warmup(gdn, {"backend": "llamacpp"})
    assert warmup.needs_gdn_warmup(gdn, {"backend": "vllm"})
    assert warmup.needs_gdn_warmup(gdn, {})           # backend defaults to vllm


def test_warmup_prompt_tokens_caps_to_window():
    assert warmup.warmup_prompt_tokens(None) == warmup.TARGET_TOKENS
    assert warmup.warmup_prompt_tokens(262_144) == warmup.TARGET_TOKENS
    assert warmup.warmup_prompt_tokens(16_384) == 16_384 - 4_096
    assert warmup.warmup_prompt_tokens(4_096) == warmup.MIN_TOKENS // 4   # floor, never zero/negative


def test_gdn_warmup_failure_is_soft(monkeypatch):
    import urllib.request

    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    out = warmup.gdn_warmup(9, "m", 32_768, timeout=0.1)
    assert out["ok"] is False and "error" in out


def test_launch_up_signature_has_warmup():
    import inspect
    from johnny.engine import launch, profiles
    assert "warmup" in inspect.signature(launch.up).parameters
    assert inspect.signature(launch.up).parameters["warmup"].default is True
    assert "warmup" in inspect.signature(profiles.up_profile).parameters
    assert inspect.signature(profiles.up_profile).parameters["warmup"].default is True
