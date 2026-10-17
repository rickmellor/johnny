"""VllmDriver.compose: placement-level docker ulimits (extra.ulimits) and shm override (extra.shm_size)."""
from johnny.backends.vllm import VllmDriver


def _spec(**extra):
    return {
        "container_name": "johnny-x-8002", "port": 8002, "image": "img:tag",
        "model_path": "/models/x", "served_model_name": "x", "gpus": [0, 1],
        "knobs": {"tensor_parallel_size": 2}, "extra": extra, "models_dir": "/mnt/data/models",
        "shm_size": "16g",
    }


def test_ulimits_become_docker_flags_before_the_image():
    argv = VllmDriver().compose(_spec(ulimits=["memlock=-1:-1", "nofile=65536:65536"]))
    lims = [argv[i + 1] for i, a in enumerate(argv) if a == "--ulimit"]
    assert lims == ["memlock=-1:-1", "nofile=65536:65536"]
    assert argv.index("--ulimit") < argv.index("img:tag")


def test_shm_size_override_and_defaults():
    argv = VllmDriver().compose(_spec(shm_size="32g"))
    assert argv[argv.index("--shm-size") + 1] == "32g"
    argv = VllmDriver().compose(_spec())
    assert argv[argv.index("--shm-size") + 1] == "16g"
    assert "--ulimit" not in argv


def test_absolute_weights_dir_is_mounted_read_only_at_weights():
    spec = _spec(); spec["weights_dir"] = "/home/rick/models/tcclaviger"; spec["model_path"] = "/weights/ckpt"
    argv = VllmDriver().compose(spec)
    vols = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert "/home/rick/models/tcclaviger:/weights:ro" in vols
    assert argv.index("-v") < argv.index("img:tag")
