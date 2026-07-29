"""Multi-quant GGUF repo sizing — fit one downloadable variant, not the sum.

The motivating case: bartowski/deepreinforce-ai_Ornith-1.0-397B-GGUF holds every
quant level (146 files, ~5.2 TB summed). The old size logic summed them all and
called the repo wont-fit, while a single IQ2 variant (~120 GB) runs fine.
"""

from johnny.discover import search
from test_fit import _box


def test_variant_key_from_subdir():
    assert search._gguf_variant("Ornith-IQ2_M/Ornith-IQ2_M-00002-of-00004.gguf") == "Ornith-IQ2_M"


def test_variant_key_from_split_suffix():
    assert search._gguf_variant("Ornith-Q4_K_M-00001-of-00008.gguf") == "Ornith-Q4_K_M"


def test_variant_key_single_file():
    assert search._gguf_variant("Ornith-1.0-397B-Featherweight-v0.gguf") == "Ornith-1.0-397B-Featherweight-v0"


def test_best_fit_picks_largest_all_gpu_variant_not_sum():
    variants = {
        "IQ1_S": int(80e9),    # fits comfortably
        "IQ2_M": int(100e9),   # tight, but still all-GPU — largest that runs, wins
        "Q4_K_M": int(230e9),  # offload
        "Q8_0": int(420e9),    # wont-fit
    }
    size, v = search._gguf_best_fit(variants, _box())
    assert v["verdict"] == "tight"
    assert size == int(100e9)
    assert v["variant"] == "IQ2_M"
    assert "IQ2_M" in v["detail"] and "4 quants" in v["detail"]


def test_best_fit_degrades_to_offload_when_nothing_fits_outright():
    variants = {"Q4_K_M": int(230e9), "Q8_0": int(420e9)}
    size, v = search._gguf_best_fit(variants, _box())
    assert v["verdict"] == "offload"
    assert size == int(230e9)


def test_best_fit_single_variant_keeps_plain_detail():
    size, v = search._gguf_best_fit({"Featherweight-v0": int(119.5e9)}, _box())
    assert v["verdict"] == "offload"
    assert "variant" not in v


_GLM_VARIANTS = {
    "BF16": int(737e9),
    "Q8_0": int(420e9),
    "UD-IQ3_XXS": int(150e9),
    "UD-IQ1_S": int(90e9),
}


def test_pick_variant_honors_explicit_request_case_insensitive():
    name, error = search._pick_variant(_GLM_VARIANTS, "ud-iq1_s", None)
    assert (name, error) == ("UD-IQ1_S", None)


def test_pick_variant_unknown_request_lists_available():
    name, error = search._pick_variant(_GLM_VARIANTS, "Q4_K_M", None)
    assert name is None
    assert "UD-IQ3_XXS" in error


def test_pick_variant_request_on_non_gguf_repo_errors():
    name, error = search._pick_variant({}, "Q8_0", None)
    assert name is None and "--quant only applies" in error


def test_pick_variant_defaults_to_best_fit():
    name, error = search._pick_variant(_GLM_VARIANTS, None, _box())
    assert error is None
    assert name == "UD-IQ1_S"  # the only all-GPU variant beats the offload ones


def test_pick_variant_multi_quant_without_hardware_errors():
    name, error = search._pick_variant(_GLM_VARIANTS, None, None)
    assert name is None and "--quant" in error


def test_pick_variant_single_variant_downloads_unfiltered():
    assert search._pick_variant({"Featherweight-v0": int(120e9)}, None, _box()) == (None, None)
    assert search._pick_variant({}, None, _box()) == (None, None)


def test_variant_patterns_match_both_layouts():
    import fnmatch

    pats = search._variant_patterns("GLM-5.2-UD-IQ3_XXS")
    keep = ["UD-IQ3_XXS/GLM-5.2-UD-IQ3_XXS-00001-of-00004.gguf",
            "GLM-5.2-UD-IQ3_XXS-00001-of-00004.gguf",
            "GLM-5.2-UD-IQ3_XXS.gguf", "README.md", "config.json"]
    drop = ["BF16/GLM-5.2-BF16-00001-of-00033.gguf", "GLM-5.2-Q8_0-00001-of-00017.gguf"]
    # subdir layout uses the bare variant key as the pattern root
    pats_dir = search._variant_patterns("UD-IQ3_XXS")
    for f in keep:
        assert any(fnmatch.fnmatch(f, p) for p in pats + pats_dir), f
    for f in drop:
        assert not any(fnmatch.fnmatch(f, p) for p in pats + pats_dir), f


def test_params_from_id_ab_notation():
    assert search._params_from_id("unsloth/Qwen3-30B-A3B-GGUF") == (int(30e9), int(3e9))
    assert search._params_from_id("google/gemma-4-26B-A4B-it") == (int(26e9), int(4e9))
    assert search._params_from_id("zai-org/GLM-5.2") == (None, None)


def test_active_params_deepseek_v3_shape():
    cfg = {"n_routed_experts": 256, "num_experts_per_tok": 8, "num_hidden_layers": 61,
           "hidden_size": 7168, "moe_intermediate_size": 2048, "first_k_dense_replace": 3}
    active = search._active_params(int(671e9), cfg)
    assert abs(active - 37e9) < 2e9  # real answer: 37B


def test_active_params_qwen3_moe_shape():
    cfg = {"num_experts": 128, "num_experts_per_tok": 8, "num_hidden_layers": 48,
           "hidden_size": 2048, "moe_intermediate_size": 768}
    active = search._active_params(int(30.5e9), cfg)
    assert abs(active - 3.3e9) < 0.5e9  # 30B-A3B


def test_active_params_dense_config_is_total():
    assert search._active_params(int(27e9), {"num_hidden_layers": 62, "hidden_size": 5120}) == int(27e9)


def test_active_params_unknown_on_empty_or_partial_config():
    assert search._active_params(int(100e9), {}) is None
    assert search._active_params(int(100e9), {"n_routed_experts": 64}) is None


def _reg(*identities):
    return {"models": {f"m{i}": {"identity": ident} for i, ident in enumerate(identities)}}


def test_registry_keys_from_clean_org_repo():
    keys = search._registry_repo_keys(_reg({"repo_id": "Qwen/Qwen3.6-27B-FP8", "local_path": "Qwen/Qwen3.6-27B-FP8"}))
    assert keys == {"qwen/qwen3.6-27b-fp8"}


def test_registry_keys_strip_weight_file_and_report_junk():
    keys = search._registry_repo_keys(_reg(
        {"repo_id": "Ornith-1.0-397B-Featherweight-v0",  # bare name: unusable alone
         "local_path": "SEBK4C/Ornith-1.0-397B-Featherweight/Ornith-1.0-397B-Featherweight-v0.gguf"},
        {"repo_id": "cyankiwi/gemma-4-31B-it-AWQ-4bit/TUNING_REPORT.md.",
         "local_path": "cyankiwi/gemma-4-31B-it-AWQ-4bit/TUNING_REPORT.md."},
    ))
    assert keys == {"sebk4c/ornith-1.0-397b-featherweight", "cyankiwi/gemma-4-31b-it-awq-4bit"}


def test_registry_keys_skip_bare_names_and_empty():
    keys = search._registry_repo_keys(_reg(
        {"repo_id": "gemma-3-4b-it", "local_path": None},
        {},
    ))
    assert keys == set()


_ORNITH_REG = {"models": {"Ornith-1.0-397B-Featherweight-v0": {
    "identity": {"repo_id": "Ornith-1.0-397B-Featherweight-v0",
                 "local_path": "SEBK4C/Ornith-1.0-397B-Featherweight/Ornith-1.0-397B-Featherweight-v0.gguf"},
    "placements": [
        {"backend": "llamacpp", "perf": {"decode_tok_s": 24.8}},
        {"backend": "llamacpp", "perf": {"decode_tok_s": 22.4}},
    ],
}}}


def test_registry_injection_appends_missing_match():
    rows = search._registry_match_rows(_ORNITH_REG, "ornith", seen_ids=set())
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == "sebk4c/ornith-1.0-397b-featherweight"
    assert r["inducted"] is True
    assert r["fit"]["verdict"] == "inducted"
    assert "llamacpp" in r["fit"]["detail"] and "24.8 tok/s" in r["fit"]["detail"]
    assert "2 placements" in r["fit"]["detail"]


def test_registry_injection_skips_repos_hf_already_returned():
    rows = search._registry_match_rows(_ORNITH_REG, "ornith",
                                       seen_ids={"sebk4c/ornith-1.0-397b-featherweight"})
    assert rows == []


def test_registry_injection_requires_all_query_terms():
    assert search._registry_match_rows(_ORNITH_REG, "ornith 397b", set())
    assert search._registry_match_rows(_ORNITH_REG, "ornith 9b", set()) == []
    assert search._registry_match_rows(_ORNITH_REG, "   ", set()) == []
