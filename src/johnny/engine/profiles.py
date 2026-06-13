"""Profiles — named fleets (human-authored). Minimal at P3: lookup + role→model."""

from __future__ import annotations

from .. import config as C


def load() -> dict:
    return C.load_yaml(C.get_paths().profiles_file) or C.profiles_stub()


def get_profile(name: str) -> dict | None:
    return (load().get("profiles") or {}).get(name)


def role_to_model(role: str) -> str | None:
    for prof in (load().get("profiles") or {}).values():
        for seat in prof.get("seats") or []:
            if seat.get("role") == role:
                return seat.get("model")
    return None
