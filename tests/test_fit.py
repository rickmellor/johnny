"""Fit-verdict tests — the GGUF/llamacpp path (VRAM+RAM) vs the vLLM TP path.

The motivating case: Ornith-1.0-397B-Featherweight (119.5 GB GGUF) on 4×32 GB
R9700 + 251 GB RAM. The old VRAM-only verdict said "won't fit at TP4"; the
llamacpp backend serves it fine with ~12 GB of experts offloaded to RAM.
"""

from johnny.discover import fit
from johnny.hardware.detect import GPU, GpuGroup, Hardware


def _box(ngpu=4, vram=32.0, ram=251.0):
    gpus = [GPU(i, "R9700", vram, "gfx1201", "amd") for i in range(ngpu)]
    groups = [GpuGroup("amd", "gfx1201", vram, ngpu, list(range(ngpu)),
                       ["fp8", "bf16", "fp16"], f"amd-gfx1201-{int(vram)}g")]
    return Hardware("amd", gpus, groups, True, False, vram * ngpu, ram,
                    ["fp8", "bf16", "fp16"], "probe", f"{ngpu}xamd-gfx1201-{int(vram)}g")


ORNITH_BYTES = 119_517_476_064


def test_gguf_bigger_than_vram_is_offload_not_wontfit():
    v = fit.fit_verdict(ORNITH_BYTES, _box(), None, gguf=True)
    assert v["verdict"] == "offload"
    assert v["backend"] == "llamacpp"
    assert 5 < v["offload_gb"] < 30


def test_gguf_quant_label_routes_to_llamacpp_path():
    v = fit.fit_verdict(ORNITH_BYTES, _box(), "gguf")
    assert v["verdict"] == "offload"


def test_safetensors_path_unchanged_wont_fit():
    v = fit.fit_verdict(ORNITH_BYTES, _box(), None)
    assert v["verdict"] == "wont-fit"
    assert v["best_tp"] is None


def test_gguf_small_fits_all_gpu():
    v = fit.fit_verdict(int(20e9), _box(), "gguf")
    assert v["verdict"] == "fits"
    assert v["gpu_count"] == 4


def test_gguf_beyond_vram_plus_ram_wont_fit():
    v = fit.fit_verdict(int(300e9), _box(), "gguf")
    assert v["verdict"] == "wont-fit"


def test_gguf_cpu_only_box_fits_in_ram():
    v = fit.fit_verdict(int(100e9), _box(ngpu=0, ram=251.0), "gguf")
    assert v["verdict"] == "fits"
    assert v["device"] == "cpu"


def test_gguf_dtype_fit_is_agnostic_not_red():
    d = fit.dtype_fit("gguf", _box())
    assert d["ok"] is None
