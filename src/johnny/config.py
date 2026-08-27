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
from typing import NamedTuple

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
DEFAULT_VLLM_IMAGE_AMD = "vllm/vllm-openai-rocm:v0.28.0"
DEFAULT_VLLM_IMAGE_NVIDIA = "vllm/vllm-openai:latest"
DEFAULT_VLLM_CPU_IMAGE = "vllm/vllm-openai-cpu:v0.27.1"

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
    # The permanent NAS store (read-only mount into a launched container, see
    # resolve_weights_path below) — optional like vllm_cache/launchers_dir: only
    # recorded if something is actually mounted there on this box.
    nas = os.environ.get("JOHNNY_NAS_DIR") or _first_existing([Path("/mnt/ug-models")])
    if nas:
        roots["nas_dir"] = nas
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


class ResolvedWeights(NamedTuple):
    """A registry-relative weights path resolved against the box's configured
    roots. `container_path` is the path as it will actually appear *inside* the
    launched container (mount prefix baked in) — the thing callers building a
    launch spec actually need, not just a host-side absolute path. `root` is
    which `roots.*` key served it ("models_dir" or "nas_dir"), so a caller can
    tell whether the NAS mount needs to be attached to this launch at all."""

    host_path: str
    container_path: str
    root: str


# roots.models_dir mounts read-write at /models (existing behavior); roots.nas_dir
# mounts read-only at /nas (this feature) — see backends/llamacpp.py, backends/vllm.py.
_ROOT_MOUNTS = (("models_dir", "/models"), ("nas_dir", "/nas"))


def resolve_weights_path(rel_path: str, cfg: dict) -> ResolvedWeights | None:
    """Resolve a registry path (`identity.local_path` or a llama.cpp placement's
    `extra.gguf_file`) against `roots.models_dir` (local, disposable) and
    `roots.nas_dir` (permanent NAS store, mounted read-only).

    Convention: a leading `nas:` prefix (e.g. `nas:vendor/Model/file.gguf`) is an
    explicit, unambiguous request to resolve under `roots.nas_dir` — use this for
    any registry entry that intentionally references a NAS-only file, so the
    registry.yaml itself documents *why* a plain `johnny.models_dir` copy doesn't
    exist. Without the prefix, resolution is existence-based (mirrors the cascade
    `bench._local_path` already used before this): try `models_dir` first (the
    common case — most weights are the local copy), then fall back to `nas_dir` so
    a file that quietly only exists on the NAS still loads instead of failing with
    a confusing "not found" against a path that's technically valid, just on the
    other root.

    Returns None if `rel_path` can't be found under either configured root (or
    the indicated one, when `nas:`-prefixed) — callers should fall back to their
    own pre-existing default behavior rather than treat this as fatal, since a
    root can be transiently unmounted (e.g. the NAS autofs mount hasn't fired
    yet) without the file actually being gone.
    """
    roots = cfg.get("roots") or {}
    forced_nas = rel_path.startswith("nas:")
    clean = rel_path[len("nas:") :] if forced_nas else rel_path

    order = [("nas_dir", "/nas")] if forced_nas else list(_ROOT_MOUNTS)
    for root_key, prefix in order:
        root = roots.get(root_key)
        if not root:
            continue
        p = Path(root).expanduser() / clean
        if p.exists():
            return ResolvedWeights(str(p), f"{prefix}/{clean}", root_key)
    return None


def _default_vllm_image(vendor: str | None) -> str:
    if vendor == "nvidia":
        return DEFAULT_VLLM_IMAGE_NVIDIA
    return DEFAULT_VLLM_IMAGE_AMD


def resolve_image(cfg: dict, *, device: str = "gpu", backend: str = "vllm",
                  model_id: str | None = None) -> str | None:
    """Effective docker image for a launch. Defaults the vLLM CPU image when the config
    omits `docker.cpu_image` (configs from before it existed), so `--device cpu` just works
    instead of launching with a null image.

    model_id: when given, an already-registered placement's own `image` pin wins over
    the global default — the same per-placement override `johnny up` already honors
    per-seat (engine/launch.py: `placement.get("image") or resolve_image(...)`). Without
    this, tune/bench/induct always launched against the global default even when
    re-tuning a model whose arch that image doesn't register (e.g. it needs a newer/
    older build than the box is pinned to) — the sweep just failed instead of reusing
    the image that's already known to work. Global default when the model is new or
    has no placement carrying its own pin."""
    docker = (cfg or {}).get("docker") or {}
    if model_id:
        try:
            from .registry import store

            m = (store.load().get("models") or {}).get(model_id) or {}
            for p in m.get("placements") or []:
                if p.get("backend") == backend and p.get("image"):
                    return p["image"]
        except Exception:
            pass
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
