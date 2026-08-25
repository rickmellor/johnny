"""Unit tests for the profiles engine (validate/capture) and the boot unit generator.

Pure-logic tests only — no docker, no GPUs. capture() is exercised with mocked
seats; validate() against an in-memory registry.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from johnny import boot
from johnny.engine import profiles


REG = {
    "schema_version": 1,
    "models": {
        "qwen-27b-coder": {"identity": {"repo_id": "x"}, "placements": [
            {"id": "qwen-tp2", "backend": "vllm", "knobs": {"gpu_count": 2}},
        ]},
        "nomic-embed": {"identity": {"repo_id": "y"}, "placements": [
            {"id": "nomic-embed-cpu", "backend": "vllm", "knobs": {"gpu_count": 0},
             "extra": {"runner": "pooling"}},
        ]},
    },
    "fingerprints": [],
}


def _seat(model, placement, port, role, pinned=True):
    return {"model": model, "placement": placement, "port": port, "role": role,
            **({"pinned": True} if pinned else {})}


GOOD = {"seats": [
    _seat("qwen-27b-coder", "qwen-tp2", 8003, "coder"),
    _seat("nomic-embed", "nomic-embed-cpu", 8001, "embed"),
]}


def _validate(profile, hardware=None):
    # cross-profile role warnings read the profiles file; isolate from the real one
    with mock.patch.object(profiles, "all_profiles", return_value={}):
        return profiles.validate(profile, REG, hardware)


def test_validate_good_profile():
    errors, warnings = _validate(GOOD)
    assert errors == []
    assert warnings == []


def test_validate_missing_required_fields():
    errors, _ = _validate({"seats": [{"model": "qwen-27b-coder"}]})
    joined = " ".join(errors)
    for field in ("placement", "port", "role"):
        assert f"missing required '{field}'" in joined


def test_validate_unknown_model_and_placement():
    errors, _ = _validate({"seats": [
        _seat("nope", "x", 8000, "chat"),
        _seat("qwen-27b-coder", "wrong-id", 8001, "coder"),
    ]})
    assert any("not in the registry" in e for e in errors)
    assert any("placement 'wrong-id' not found" in e for e in errors)


def test_validate_duplicate_port_is_the_only_hard_collision():
    errors, _ = _validate({"seats": [
        _seat("qwen-27b-coder", "qwen-tp2", 8003, "coder"),
        _seat("qwen-27b-coder", "qwen-tp2", 8003, "coder"),
    ]})
    assert any("duplicate port '8003'" in e for e in errors)
    assert not any("duplicate model" in e or "duplicate role" in e for e in errors)


def test_validate_scaleout_profile_is_clean():
    # N seats of one model+role on distinct ports (e.g. 12b-quad) is legitimate
    errors, warnings = _validate({"seats": [
        _seat("qwen-27b-coder", "qwen-tp2", 8000 + i, "coder") for i in range(4)
    ]})
    assert errors == []
    assert warnings == []


def test_validate_role_across_models_warns():
    errors, warnings = _validate({"seats": [
        _seat("qwen-27b-coder", "qwen-tp2", 8003, "chat"),
        _seat("nomic-embed", "nomic-embed-cpu", 8001, "chat"),
    ]})
    assert errors == []
    assert any("spans multiple models" in w for w in warnings)


def test_validate_gpu_oversubscription_warns():
    hw = SimpleNamespace(gpus=[0])  # one GPU, placement wants 2
    _, warnings = _validate(GOOD, hardware=hw)
    assert any("GPUs" in w for w in warnings)


def test_validate_cross_profile_role_warns():
    other = {"other": {"seats": [_seat("m", "p", 9000, "coder")]}}
    with mock.patch.object(profiles, "all_profiles", return_value=other):
        _, warnings = profiles.validate(GOOD, REG, None)
    w = [x for x in warnings if "role 'coder'" in x]
    assert len(w) == 1 and "also defined in other" in w[0]
    # the message must not claim plain first-match: resolve prefers the LIVE seat
    assert "live seat" in w[0]


def test_validate_cross_profile_role_warns_once_per_role():
    """A profile with several seats on the same role (scale-out, e.g. 12b-quad's four
    chat seats) must produce ONE warning naming the profiles, not one per seat."""
    other = {"other": {"seats": [_seat("m", "p", 9000, "coder"),
                                 _seat("m", "p", 9001, "coder"),
                                 _seat("m", "p", 9002, "coder")]},
             "third": {"seats": [_seat("m", "p", 9003, "coder")]}}
    with mock.patch.object(profiles, "all_profiles", return_value=other):
        _, warnings = profiles.validate(GOOD, REG, None)
    w = [x for x in warnings if "role 'coder'" in x]
    assert len(w) == 1, w
    assert "other" in w[0] and "third" in w[0]


def _live(name, model, placement, port, managed=True):
    labels = {"johnny.managed": "1", "johnny.model": model,
              "johnny.placement": placement} if managed else {}
    return SimpleNamespace(name=name, model=model, port=port,
                           extra={"labels": labels})


def test_capture_maps_seats_and_skips_unmanaged():
    seats = [
        _live("johnny-qwen-27b-coder-8003", "qwen-27b-coder", "qwen-tp2", 8003),
        _live("johnny-nomic-embed-8001", "nomic-embed", "nomic-embed-cpu", 8001),
        _live("vllm-nomic-cpu", "nomic-embed", "", 8009, managed=False),
    ]
    with mock.patch.object(profiles, "_infer_role", side_effect=lambda m, p, r: "embed" if m == "nomic-embed" else None), \
         mock.patch("johnny.engine.all_seats", return_value=seats), \
         mock.patch("johnny.engine.load_config", return_value={}), \
         mock.patch("johnny.registry.store.load", return_value=REG), \
         mock.patch("johnny.telemetry.collect.is_pinned", side_effect=lambda n: "qwen" in n):
        cap = profiles.capture(cfg={}, roles={"qwen-27b-coder": "coder"})
    assert cap["skipped"] == ["vllm-nomic-cpu"]
    by_model = {s["model"]: s for s in cap["seats"]}
    assert by_model["qwen-27b-coder"] == {"model": "qwen-27b-coder", "placement": "qwen-tp2",
                                          "port": 8003, "role": "coder", "pinned": True}
    # role inferred for the pooling seat; unpinned seat carries no pinned key
    assert by_model["nomic-embed"]["role"] == "embed"
    assert "pinned" not in by_model["nomic-embed"]


def test_capture_flags_missing_role():
    seats = [_live("johnny-x-8000", "qwen-27b-coder", "qwen-tp2", 8000)]
    with mock.patch("johnny.engine.all_seats", return_value=seats), \
         mock.patch("johnny.engine.load_config", return_value={}), \
         mock.patch("johnny.registry.store.load", return_value=REG), \
         mock.patch("johnny.telemetry.collect.is_pinned", return_value=False):
        cap = profiles.capture(cfg={})
    assert "role" not in cap["seats"][0]  # caller (save) turns this into an error


def test_infer_role_pooling():
    assert profiles._infer_role("nomic-embed", "nomic-embed-cpu", REG) == "embed"
    assert profiles._infer_role("qwen-27b-coder", "qwen-tp2", REG) is None


def test_boot_unit_text():
    text = boot.unit_text(johnny="/home/rick/.local/bin/johnny")
    assert "ExecStart=/home/rick/.local/bin/johnny profile up %i --wait" in text
    assert "ExecStop=/home/rick/.local/bin/johnny profile down %i" in text
    assert "TimeoutStartSec=1800" in text
    assert "WantedBy=default.target" in text
    assert "docker info" in text  # waits for the docker socket


def test_boot_disable_never_stops_the_fleet():
    # plain disable only — `--now` would run ExecStop (profile down) and stop seats
    for cmd in boot.disable_commands("standard"):
        assert "--now" not in cmd
    assert any("--now" in c for c in boot.enable_commands("standard"))


def test_role_to_models_prefers_definition_order_and_dedups(monkeypatch):
    from johnny.engine import profiles

    doc = {"profiles": {
        "standard": {"seats": [{"role": "chat", "model": "gemma"}]},
        "knowledge": {"seats": [{"role": "chat", "model": "ornith"},
                                {"role": "chat", "model": "gemma"}]},
    }}
    monkeypatch.setattr(profiles, "load", lambda: doc)
    assert profiles.role_to_models("chat") == ["gemma", "ornith"]
    assert profiles.role_to_model("chat") == "gemma"
    assert profiles.role_to_models("nope") == []


def test_role_aliases_resolve_within_own_profile(monkeypatch):
    from johnny.engine import profiles

    doc = {"profiles": {
        "standard": {"seats": [{"role": "coder", "model": "qwen"},
                               {"role": "chat", "model": "gemma"}]},
        "knowledge": {"seats": [{"role": "chat", "model": "ornith"}],
                      "role_aliases": {"coder": "chat"}},
    }}
    monkeypatch.setattr(profiles, "load", lambda: doc)
    # real coder seat first (file order), then knowledge's aliased chat seat
    assert profiles.role_to_models("coder") == ["qwen", "ornith"]
    # alias resolves against its OWN profile's seats, not standard's chat
    assert "gemma" not in profiles.role_to_models("coder")
