"""GGUF ref resolution for directory refs — the layout `johnny download` produces.

Motivating case: `johnny download unsloth/GLM-5.2-GGUF` lands one variant at
models/unsloth/GLM-5.2-GGUF/UD-IQ3_XXS/*.gguf, but `johnny induct
unsloth/GLM-5.2-GGUF` failed "not found on disk" — gguf_ref only accepted
direct .gguf paths, so the repo dir fell through to the config.json path.
"""

import pytest

from johnny.induct import llamacpp


def _cfg(models_dir):
    return {"roots": {"models_dir": str(models_dir)}}


def _mk_shards(d, stem, n=2):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (d / f"{stem}-{i:05d}-of-{n:05d}.gguf").write_bytes(b"x")


def test_repo_dir_with_one_variant_subdir_resolves_to_first_shard(tmp_path):
    repo = tmp_path / "unsloth" / "GLM-5.2-GGUF"
    _mk_shards(repo / "UD-IQ3_XXS", "GLM-5.2-UD-IQ3_XXS", n=4)
    (repo / "README.md").write_text("readme")
    mid, path = llamacpp.gguf_ref("unsloth/GLM-5.2-GGUF", _cfg(tmp_path))
    assert mid == "GLM-5.2-UD-IQ3_XXS"
    assert path.endswith("UD-IQ3_XXS/GLM-5.2-UD-IQ3_XXS-00001-of-00004.gguf")


def test_variant_subdir_ref_resolves(tmp_path):
    _mk_shards(tmp_path / "r" / "Q4_K_M", "m-Q4_K_M")
    mid, path = llamacpp.gguf_ref("r/Q4_K_M", _cfg(tmp_path))
    assert mid == "m-Q4_K_M"
    assert path.endswith("m-Q4_K_M-00001-of-00002.gguf")


def test_single_file_variant_in_repo_root(tmp_path):
    repo = tmp_path / "org" / "repo"
    repo.mkdir(parents=True)
    (repo / "model-Q8_0.gguf").write_bytes(b"x")
    mid, path = llamacpp.gguf_ref("org/repo", _cfg(tmp_path))
    assert mid == "model-Q8_0"
    assert path.endswith("model-Q8_0.gguf")


def test_multiple_variants_error_names_them_when_not_a_tty(tmp_path):
    repo = tmp_path / "org" / "repo"
    _mk_shards(repo / "Q4_K_M", "m-Q4_K_M")
    _mk_shards(repo / "Q8_0", "m-Q8_0")
    with pytest.raises(FileNotFoundError, match="m-Q4_K_M.*m-Q8_0"):
        llamacpp.gguf_ref("org/repo", _cfg(tmp_path))


def test_multiple_variants_prompt_when_interactive(tmp_path, monkeypatch):
    from johnny.external import picker

    repo = tmp_path / "org" / "repo"
    _mk_shards(repo / "Q4_K_M", "m-Q4_K_M", n=3)
    (repo / "Q8_0").mkdir()
    (repo / "Q8_0" / "m-Q8_0.gguf").write_bytes(b"xxxx")  # bigger → sorted last
    monkeypatch.setattr(picker, "_interactive_capable", lambda: True)
    seen = {}

    def fake_select(items, render, title=""):
        seen["items"] = list(items)
        return 1

    monkeypatch.setattr(picker, "select", fake_select)
    mid, path = llamacpp.gguf_ref("org/repo", _cfg(tmp_path))
    assert seen["items"] == ["m-Q4_K_M", "m-Q8_0"]  # size-sorted, smallest first
    assert mid == "m-Q8_0"
    assert path.endswith("Q8_0/m-Q8_0.gguf")


def test_multiple_variants_cancelled_prompt_errors(tmp_path, monkeypatch):
    from johnny.external import picker

    repo = tmp_path / "org" / "repo"
    _mk_shards(repo / "Q4_K_M", "m-Q4_K_M")
    _mk_shards(repo / "Q8_0", "m-Q8_0")
    monkeypatch.setattr(picker, "_interactive_capable", lambda: True)
    monkeypatch.setattr(picker, "select", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError, match="pass the variant"):
        llamacpp.gguf_ref("org/repo", _cfg(tmp_path))


def test_non_gguf_dir_falls_through_to_none(tmp_path):
    repo = tmp_path / "org" / "safetensors-repo"
    repo.mkdir(parents=True)
    (repo / "config.json").write_text("{}")
    (repo / "model.safetensors").write_bytes(b"x")
    assert llamacpp.gguf_ref("org/safetensors-repo", _cfg(tmp_path)) is None


def test_direct_gguf_path_still_works(tmp_path):
    f = tmp_path / "solo-IQ2_M.gguf"
    f.write_bytes(b"x")
    mid, path = llamacpp.gguf_ref(str(f), _cfg(tmp_path))
    assert mid == "solo-IQ2_M"
    assert path == str(f.resolve())


# --- weight split (GPU vs CPU-RAM offload) -----------------------------------

def test_split_tensors_override_regex_sends_experts_to_ram():
    tensors = [
        ("blk.0.attn_q.weight", 100),
        ("blk.0.ffn_gate_exps.weight", 1000),
        ("blk.1.ffn_gate_exps.weight", 1000),
        ("output.weight", 50),
    ]
    vram, ram = llamacpp._split_tensors(tensors, r"blk\.(0)\.ffn_(gate|up|down)_exps\.weight=CPU", None)
    assert (vram, ram) == (1150, 1000)


def test_split_tensors_ncmoe_first_n_layers():
    tensors = [(f"blk.{i}.ffn_up_exps.weight", 10) for i in range(4)] + [("blk.0.attn_q.weight", 5)]
    vram, ram = llamacpp._split_tensors(tensors, None, 2)
    assert (vram, ram) == (25, 20)


def test_split_tensors_no_offload_all_vram():
    assert llamacpp._split_tensors([("a", 7), ("b", 3)], None, None) == (10, 0)


def test_gguf_tensor_sizes_from_offset_deltas(tmp_path):
    import struct

    from johnny.backends.llamacpp import gguf_tensors

    # minimal GGUF v3: no KV, two tensors, 32-byte alignment
    hdr = b"GGUF" + struct.pack("<IQQ", 3, 2, 0)
    def tinfo(name, off):
        b = struct.pack("<Q", len(name)) + name.encode()
        return b + struct.pack("<IQIQ", 1, 4, 0, off)  # 1 dim, dim=4, f32, offset
    body = tinfo("t1", 0) + tinfo("t2", 64)
    data_start = len(hdr) + len(body)
    pad = (-data_start) % 32
    f = tmp_path / "mini.gguf"
    f.write_bytes(hdr + body + b"\0" * (pad + 64 + 16))  # t1: 64 bytes, t2: 16
    assert gguf_tensors(f) == [("t1", 64), ("t2", 16)]
