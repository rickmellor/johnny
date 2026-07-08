"""Config + roots discovery.

johnny's code is the package; the user's *data* lives outside it under XDG dirs
(so a package upgrade never clobbers it):

  $XDG_CONFIG_HOME/johnny/  -> config.yaml, registry.yaml, profiles.yaml
  $XDG_STATE_HOME/johnny/   -> ingest/, runs/, telemetry.db

Every owned file carries a `schema_version` (see migrate.py). All roots are
config-driven with env overrides + autodiscovery; nothing about the host is
hardwired. Autodiscovery records a path only if it exists, which keeps the
starter config portable across boxes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import platformdirs
import yaml

from .util import which

APP_NAME = "johnny"

# Schema versions for the files johnny owns. Bump + add a migration when a
# format changes (migrate.py). v1 = the P0 baseline.
CONFIG_SCHEMA_VERSION = 1
REGISTRY_SCHEMA_VERSION = 1
PROFILES_SCHEMA_VERSION = 1

# Vendor-appropriate default vLLM images (config can override).
DEFAULT_VLLM_IMAGE_AMD = "vllm/vllm-openai-rocm:v0.20.2"
DEFAULT_VLLM_IMAGE_NVIDIA = "vllm/vllm-openai:latest"
DEFAULT_VLLM_CPU_IMAGE = "vllm/vllm-openai-cpu:v0.20.2"

# llama.cpp server image (GGUF backend). Self-contained builds ship llama-server as
# the ENTRYPOINT; no vendor split (the same image targets the box's GPU arch).
DEFAULT_LLAMACPP_IMAGE = "johnny-llamacpp-dsv4:gfx1201"


@dataclass
class Paths:
    config_dir: Path
    state_dir: Path

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.yaml"

    @property
    def registry_file(self) -> Path:
        return self.config_dir / "registry.yaml"

    @property
    def profiles_file(self) -> Path:
        return self.config_dir / "profiles.yaml"

    @property
    def ingest_dir(self) -> Path:
        return self.state_dir / "ingest"

    @property
    def runs_dir(self) -> Path:
        return self.state_dir / "runs"

    @property
    def db_file(self) -> Path:
        return self.state_dir / "telemetry.db"


def get_paths() -> Paths:
    """Resolve config/state dirs. JOHNNY_CONFIG_DIR/JOHNNY_STATE_DIR override XDG."""
    cfg = os.environ.get("JOHNNY_CONFIG_DIR")
    st = os.environ.get("JOHNNY_STATE_DIR")
    config_dir = Path(cfg).expanduser() if cfg else Path(platformdirs.user_config_dir(APP_NAME))
    state_dir = Path(st).expanduser() if st else Path(platformdirs.user_state_dir(APP_NAME))
    return Paths(config_dir, state_dir)


def detect_gpu_vendor() -> str | None:
    """Light probe for the starter config; full detection is P1."""
    if Path("/dev/kfd").exists():
        return "amd"
    if which("nvidia-smi"):
        return "nvidia"
    return None


def _first_existing(candidates: list[Path]) -> str | None:
    for p in candidates:
        if p and Path(p).expanduser().exists():
            return str(Path(p).expanduser())
    return None


def autodiscover() -> dict:
    """Probe the box for roots, reusable scripts, and available backends.

    Records a path only if it exists, so the resulting config is portable.
    """
    home = Path.home()
    paths = get_paths()
    vendor = detect_gpu_vendor()

    roots: dict = {}
    roots["models_dir"] = (
        os.environ.get("JOHNNY_MODELS_DIR")
        or _first_existing([home / "models", Path.cwd() / "models"])
        or str(home / "models")
    )
    vc = _first_existing([home / "vllm" / "vllm-cache"])
    if vc:
        roots["vllm_cache"] = vc
    roots["results_dir"] = str(paths.runs_dir)
    ld = _first_existing([home / "vllm" / "launchers"])
    if ld:
        roots["launchers_dir"] = ld  # consumed by the P2 registry importer

    # The mlops scripts ship bundled in the package (johnny/bundled.py); we no longer
    # hunt for machine-specific copies. `config.scripts.<key>` remains an optional
    # override for users who want to point at their own copy.
    backends = {
        "vllm": bool(which("docker")) and os.name == "posix",
        "llamacpp": bool(which("docker")) and os.name == "posix",
        "lmstudio": bool(which("lms")),
        "ollama": bool(which("ollama")),
    }

    return {
        "vendor": vendor,
        "roots": roots,
        "backends": backends,
    }


def _default_vllm_image(vendor: str | None) -> str:
    if vendor == "nvidia":
        return DEFAULT_VLLM_IMAGE_NVIDIA
    return DEFAULT_VLLM_IMAGE_AMD


def resolve_image(cfg: dict, *, device: str = "gpu", backend: str = "vllm") -> str | None:
    """Effective docker image for a launch. Defaults the vLLM CPU image when the config
    omits `docker.cpu_image` (configs from before it existed), so `--device cpu` just works
    instead of launching with a null image."""
    docker = (cfg or {}).get("docker") or {}
    if backend == "llamacpp":
        return docker.get("llamacpp_image")
    if device == "cpu":
        return docker.get("cpu_image") or DEFAULT_VLLM_CPU_IMAGE
    return docker.get("vllm_image")


def build_default_config(disc: dict | None = None) -> dict:
    disc = disc or autodiscover()
    cfg: dict = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "roots": disc["roots"],
        "docker": {
            "vllm_image": _default_vllm_image(disc.get("vendor")),
            "cpu_image": DEFAULT_VLLM_CPU_IMAGE,
            "llamacpp_image": DEFAULT_LLAMACPP_IMAGE,
            "shm_size": "16g",
        },
        "network": {
            # Security default: localhost only. A seat is an unauthenticated
            # OpenAI endpoint; LAN exposure is an explicit, deliberate opt-in.
            "bind_address": "127.0.0.1",
            "advertise_host": "auto",
            "ports": {"base": 8000, "reserved": {"embeddings": 8001}, "range": [8000, 8019]},
        },
        "backends": {"enabled": [k for k, v in disc["backends"].items() if v]},
        # Optional per-script overrides (key -> path); empty = use the bundled copies.
        "scripts": {},
        # Chat-tool handoff for `alive` / `provider sync` (generic by default).
        "external": {"provider": "johnny", "adapter": "hermes", "tmux_session": "johnny"},
    }
    return cfg


def registry_stub() -> dict:
    return {"schema_version": REGISTRY_SCHEMA_VERSION, "models": {}, "fingerprints": []}


def profiles_stub() -> dict:
    return {"schema_version": PROFILES_SCHEMA_VERSION, "profiles": {}}


def load_yaml(path: Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: dict, header: str | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    with open(p, "w") as f:
        if header:
            f.write(header.rstrip() + "\n\n")
        f.write(body)
