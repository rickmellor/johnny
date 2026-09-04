"""VllmDriver.compose: placement-level bind mounts (extra.mounts) for patched-source seats."""
from pathlib import Path

from johnny.backends.vllm import VllmDriver


def _spec(**extra):
    return {
        "container_name": "johnny-x-8002", "port": 8002, "image": "img:tag",
        "model_path": "/models/x", "served_model_name": "x", "gpus": [0, 1],
        "knobs": {"tensor_parallel_size": 2}, "extra": extra, "models_dir": "/mnt/data/models",
    }


def test_mounts_become_volume_args_in_order():
    argv = VllmDriver().compose(_spec(mounts=[
        "/a/ple_cpu.py:/usr/lib/vllm/ple_cpu.py:ro",
        "~/scratch/fnrepo/moe_wna16.py:/usr/lib/vllm/moe_wna16.py",
    ]))
    vols = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert vols[0] == "/mnt/data/models:/models"
    assert "/a/ple_cpu.py:/usr/lib/vllm/ple_cpu.py:ro" in vols
    assert f"{Path.home()}/scratch/fnrepo/moe_wna16.py:/usr/lib/vllm/moe_wna16.py" in vols
    assert vols.index("/a/ple_cpu.py:/usr/lib/vllm/ple_cpu.py:ro") < vols.index(
        f"{Path.home()}/scratch/fnrepo/moe_wna16.py:/usr/lib/vllm/moe_wna16.py")
    # mounts come before the image so docker treats them as run options
    assert argv.index("-v") < argv.index("img:tag")


def test_no_mounts_key_is_a_noop():
    argv = VllmDriver().compose(_spec())
    vols = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert vols == ["/mnt/data/models:/models"]
