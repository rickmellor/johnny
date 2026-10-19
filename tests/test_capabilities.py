"""Modalities per model and per placement — pure-logic, no docker, no real registry.

`resolve`'s `capabilities` key is what SAINT uses to exclude a seat that cannot take
an image/audio request, so the two things that matter are the derivation rules
(legacy bool, explicit list, launch flags) and that resolve always carries the key
without ever failing on registry trouble.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from johnny import capabilities
from johnny.capabilities import MODALITIES, model_modalities, placement_modalities, seat_capabilities
from johnny.engine import service


def _model(caps: dict, placements: list | None = None) -> dict:
    return {"identity": {"repo_id": "m"}, "capabilities": caps,
            "placements": placements if placements is not None else []}


def _placement(pid: str, backend: str, extra: dict | None = None) -> dict:
    return {"id": pid, "backend": backend, "knobs": {}, "extra": extra or {}}


def _vllm(*flags: str) -> dict:
    return _placement("p-vllm", "vllm", {"extra_flags": list(flags)} if flags else {})


# --- model level -------------------------------------------------------------

def test_legacy_multimodal_bool_reads_as_vision():
    assert model_modalities(_model({"multimodal": True})) == ["text", "vision"]
    assert model_modalities(_model({"multimodal": False})) == ["text"]
    assert model_modalities(_model({})) == ["text"]
    assert model_modalities({}) == ["text"]


def test_explicit_list_wins_over_the_bool():
    # An audio-only model: the bool would have (safely) implied vision.
    assert model_modalities(_model({"multimodal": True, "modalities": ["text", "audio"]})) == ["text", "audio"]
    # Explicitly text-only despite the bool (e.g. a corrected registry entry).
    assert model_modalities(_model({"multimodal": True, "modalities": ["text"]})) == ["text"]
    # An explicit null (an induction that couldn't tell) falls back to the bool.
    assert model_modalities(_model({"multimodal": True, "modalities": None})) == ["text", "vision"]


def test_explicit_list_is_deduped_and_ordered():
    assert model_modalities(_model({"modalities": ["audio", "vision", "vision", "text"]})) == [
        "text", "vision", "audio"]
    # text is implied for a generative model even when the list omits it.
    assert model_modalities(_model({"modalities": ["vision"]})) == ["text", "vision"]
    # An unknown modality is passed through, not dropped.
    assert model_modalities(_model({"modalities": ["vision", "depth"]})) == ["text", "vision", "depth"]
    for m in model_modalities(_model({"modalities": ["audio", "vision"]}))[:2]:
        assert m in MODALITIES


# --- vllm placement level ----------------------------------------------------

def test_vllm_language_model_only_drops_vision_and_audio():
    m = _model({"modalities": ["text", "vision", "audio"]})
    assert placement_modalities(m, _vllm("--language-model-only")) == ["text"]
    assert placement_modalities(m, _vllm("--language-model-only", "--enforce-eager")) == ["text"]
    assert placement_modalities(m, _vllm()) == ["text", "vision", "audio"]


def test_vllm_limit_mm_per_prompt_all_spellings():
    m = _model({"modalities": ["text", "vision", "audio"]})
    # two tokens
    assert placement_modalities(m, _vllm("--limit-mm-per-prompt", "image=0,video=0")) == ["text", "audio"]
    # one token
    assert placement_modalities(m, _vllm("--limit-mm-per-prompt=image=0")) == ["text", "audio"]
    # JSON value
    assert placement_modalities(m, _vllm("--limit-mm-per-prompt", '{"image": 0, "audio": 0}')) == ["text"]
    # a nonzero limit keeps the modality; video still counts as vision
    assert placement_modalities(m, _vllm("--limit-mm-per-prompt=image=4")) == ["text", "vision", "audio"]
    assert placement_modalities(m, _vllm("--limit-mm-per-prompt", "image=0,video=1")) == ["text", "vision", "audio"]
    assert placement_modalities(m, _vllm("--limit-mm-per-prompt", "audio=0")) == ["text", "vision"]


def test_vllm_unparseable_limit_mm_leaves_modalities_untouched():
    m = _model({"modalities": ["text", "vision", "audio"]})
    for bad in (["--limit-mm-per-prompt", "garbage"],
                ["--limit-mm-per-prompt", "image=lots"],
                ["--limit-mm-per-prompt", "[0]"],
                ["--limit-mm-per-prompt"],
                ["--limit-mm-per-prompt", ""]):
        assert placement_modalities(m, _vllm(*bad)) == ["text", "vision", "audio"], bad
    # a non-string entry in extra_flags doesn't break the scan
    assert placement_modalities(m, _placement("p", "vllm", {"extra_flags": [None, "--language-model-only"]})) == ["text"]


# --- llamacpp placement level ------------------------------------------------

def test_llamacpp_needs_an_mmproj_for_vision():
    m = _model({"multimodal": True})
    assert placement_modalities(m, _placement("p", "llamacpp")) == ["text"]
    assert placement_modalities(m, _placement("p", "llamacpp", {"extra_flags": ["-ngl", "999"]})) == ["text"]
    for flags in (["--mmproj", "mmproj-F16.gguf"], ["--mmproj=mm.gguf"], ["-mm", "mm.gguf"]):
        assert placement_modalities(m, _placement("p", "llamacpp", {"extra_flags": flags})) == ["text", "vision"], flags
    # the extra.mmproj key works the same way as the flag
    assert placement_modalities(m, _placement("p", "llamacpp", {"mmproj": "mm.gguf"})) == ["text", "vision"]
    # audio hinges on that same projector file
    audio = _model({"modalities": ["text", "vision", "audio"]})
    assert placement_modalities(audio, _placement("p", "llamacpp")) == ["text"]
    assert placement_modalities(audio, _placement("p", "llamacpp", {"mmproj": "mm.gguf"})) == [
        "text", "vision", "audio"]


# --- backends johnny doesn't own the flags for -------------------------------

def test_systemd_and_unknown_backends_pass_the_model_list_through():
    m = _model({"modalities": ["text", "vision"]})
    for backend in ("systemd", "lmstudio", "something-new"):
        got = placement_modalities(m, _placement("p", backend, {"extra_flags": ["--language-model-only"]}))
        assert got == ["text", "vision"], backend
    assert placement_modalities(m, None) == ["text", "vision"]


# --- seat_capabilities ------------------------------------------------------

REG = {
    "models": {
        "gemma-vision": {"identity": {}, "capabilities": {"multimodal": True}, "placements": [
            _placement("gemma-tp4", "vllm"),
            _placement("gemma-tp4-textonly", "vllm", {"extra_flags": ["--language-model-only"]}),
        ]},
        "qwen-text": {"identity": {}, "capabilities": {"modalities": ["text"]}, "placements": [
            _placement("qwen-tp2", "vllm"),
        ]},
        "saint-features": {"identity": {}, "capabilities": {}, "placements": [
            _placement("rtx4080-systemd", "systemd", {"unit": "saint-features.service"}),
        ]},
    }
}


def test_seat_capabilities_is_placement_aware():
    assert seat_capabilities(REG, "gemma-vision", "gemma-tp4") == {"modalities": ["text", "vision"]}
    assert seat_capabilities(REG, "gemma-vision", "gemma-tp4-textonly") == {"modalities": ["text"]}
    assert seat_capabilities(REG, "qwen-text", "qwen-tp2") == {"modalities": ["text"]}
    assert seat_capabilities(REG, "saint-features", "rtx4080-systemd") == {"modalities": ["text"]}
    # an unknown placement id still reports the model-level list
    assert seat_capabilities(REG, "gemma-vision", "nope") == {"modalities": ["text", "vision"]}
    assert seat_capabilities(REG, "gemma-vision", None) == {"modalities": ["text", "vision"]}


def test_seat_capabilities_unknown_or_missing_model_is_none():
    assert seat_capabilities(REG, "no-such-model", "gemma-tp4") is None
    assert seat_capabilities(REG, None, None) is None
    assert seat_capabilities({}, "gemma-vision", None) is None
    assert seat_capabilities(None, "gemma-vision", None) is None


# --- resolve ----------------------------------------------------------------

def _live_seat(name="johnny-gemma-tp4-8000", model="gemma-vision", placement="gemma-tp4",
               port=8000, state="ready"):
    return SimpleNamespace(name=name, model=model, port=port, state=state, gpus=[0, 1, 2, 3],
                           extra={"labels": {"johnny.managed": "1", "johnny.model": model,
                                             "johnny.placement": placement}})


def _resolve(target, seat):
    """resolve() with the fleet, the role→seat search, the registry and the telemetry
    probes all stubbed — the same seams tests/test_profiles.py patches."""
    with mock.patch.object(service, "all_seats", return_value=[] if seat is None else [seat]), \
         mock.patch.object(service, "_find", return_value=(seat, None if seat else "gemma-vision")), \
         mock.patch("johnny.registry.store.load", return_value=REG), \
         mock.patch.object(service, "seat_guidance", return_value=None), \
         mock.patch("johnny.telemetry.sources.metrics_for_port", return_value={}), \
         mock.patch("johnny.telemetry.collect.cold_start_estimate", return_value=None), \
         mock.patch("johnny.engine.profiles.all_profiles", return_value={}):
        return service.resolve(target, cfg={})


def test_resolve_ready_seat_carries_placement_modalities():
    res = _resolve("chat", _live_seat())
    assert res["state"] == "ready"
    assert res["capabilities"] == {"modalities": ["text", "vision"]}


def test_resolve_ready_seat_reflects_a_text_only_placement():
    res = _resolve("chat", _live_seat(placement="gemma-tp4-textonly"))
    assert res["capabilities"] == {"modalities": ["text"]}


def test_resolve_absent_seat_still_carries_the_key():
    res = _resolve("chat", None)
    assert res["state"] == "absent"
    assert res["model"] == "gemma-vision"
    assert res["capabilities"] == {"modalities": ["text", "vision"]}  # model level: no placement to narrow by


def test_resolve_absent_unknown_model_capabilities_is_none():
    with mock.patch.object(service, "all_seats", return_value=[]), \
         mock.patch.object(service, "_find", return_value=(None, "not-in-the-registry")), \
         mock.patch("johnny.registry.store.load", return_value=REG), \
         mock.patch.object(service, "seat_guidance", return_value=None), \
         mock.patch("johnny.telemetry.collect.cold_start_estimate", return_value=None), \
         mock.patch("johnny.engine.profiles.all_profiles", return_value={}):
        res = service.resolve("not-in-the-registry", cfg={})
    assert res["state"] == "absent"
    assert res["capabilities"] is None


def test_resolve_survives_a_broken_registry():
    """SAINT's hot path: an unreadable registry costs the capabilities key, nothing else."""
    with mock.patch.object(service, "all_seats", return_value=[_live_seat()]), \
         mock.patch.object(service, "_find", return_value=(_live_seat(), None)), \
         mock.patch.object(service, "_REG_CACHE", None), \
         mock.patch("johnny.registry.store.load", side_effect=RuntimeError("yaml boom")), \
         mock.patch.object(service, "seat_guidance", return_value=None), \
         mock.patch("johnny.telemetry.sources.metrics_for_port", return_value={}):
        res = service.resolve("chat", cfg={})
    assert res["state"] == "ready"
    assert res["capabilities"] is None


def test_seat_capabilities_never_raises_on_junk():
    junk = {"models": {"m": {"capabilities": {"modalities": "vision"}, "placements": "nope"}}}
    assert seat_capabilities(junk, "m", "p") == {"modalities": ["text"]}
    assert capabilities.placement_modalities({"capabilities": None}, {"backend": "vllm", "extra": None}) == ["text"]


# --- induction writes the list ----------------------------------------------

_VISION_CFG = {"image_size": 448, "patch_size": 14}
_AUDIO_CFG = {"feature_size": 128, "num_mel_bins": 128}


def _audit_stubs(monkeypatch, config: dict, arch: str | None):
    """stages.audit with the driver + GGUF/config probes stubbed: only the tower
    detection is under test here."""
    from johnny.induct import stages

    class _Drv:
        def probe_model(self, path):
            return {"arch": arch}

    monkeypatch.setattr(stages, "VllmDriver", _Drv)
    monkeypatch.setattr(stages.grid, "read_config", lambda path: config)
    monkeypatch.setattr(stages.grid, "dims", lambda config: {})
    monkeypatch.setattr(stages.grid, "model_size_bytes", lambda path: 0)
    return stages.audit("some/path")


def test_audit_derives_modalities_from_config_towers(monkeypatch):
    assert _audit_stubs(monkeypatch, {}, "Qwen3ForCausalLM") == {
        "arch": "Qwen3ForCausalLM", "dims": {}, "size_bytes": 0,
        "multimodal": False, "modalities": ["text"]}
    assert _audit_stubs(monkeypatch, {"vision_config": _VISION_CFG},
                        "Qwen3VLForConditionalGeneration")["modalities"] == ["text", "vision"]
    assert _audit_stubs(monkeypatch, {"audio_config": _AUDIO_CFG}, "Qwen3ForCausalLM")["modalities"] == [
        "text", "audio"]
    both = {"vision_config": _VISION_CFG, "audio_config": _AUDIO_CFG}
    assert _audit_stubs(monkeypatch, both, "Qwen2_5OmniForConditionalGeneration")["modalities"] == [
        "text", "vision", "audio"]
    # the arch suffix alone means a vision encoder bolted on (and multimodal for the CPU warning)
    arch_only = _audit_stubs(monkeypatch, {}, "Cohere2VisionForConditionalGeneration")
    assert arch_only["multimodal"] is True and arch_only["modalities"] == ["text", "vision"]


def test_write_placement_records_modalities_without_overwriting(monkeypatch):
    from johnny.induct import report

    saved = {}
    reg = {"models": {"m": {"identity": {}, "capabilities": {"modalities": ["text"]},
                            "placements": [], "lifecycle": {}}}}
    monkeypatch.setattr(report.store, "load", lambda *a, **k: reg)
    monkeypatch.setattr(report.store, "save", lambda r, *a, **k: saved.update(r))
    monkeypatch.setattr(report.normalize, "identity_gaps", lambda mid, m: {})
    report.write_placement("m", {"arch": "A", "modalities": ["text", "vision"]},
                           {"id": "p1", "backend": "vllm"}, SimpleNamespace(fingerprint="fp"))
    assert saved["models"]["m"]["capabilities"]["modalities"] == ["text"]  # fill-if-absent

    reg2 = {"models": {"n": {"identity": {}, "capabilities": {}, "placements": [], "lifecycle": {}}}}
    monkeypatch.setattr(report.store, "load", lambda *a, **k: reg2)
    report.write_placement("n", {"arch": "A", "multimodal": True, "modalities": ["text", "vision"]},
                           {"id": "p1", "backend": "vllm"}, SimpleNamespace(fingerprint="fp"))
    assert reg2["models"]["n"]["capabilities"]["modalities"] == ["text", "vision"]



def test_resolve_prefers_registry_id_label_over_served_name(monkeypatch):
    """A systemd seat serves under an alias (gemma31) while the launch label carries the
    registry id — capabilities must come from the registry entry, not the alias."""
    from johnny.engine import service

    class _Seat:
        backend, name, model, port, gpus, state = "systemd", "gemma31.service", "gemma31", 8121, [], "ready"
        extra = {"labels": {"johnny.model": "gemma-4-31B-it", "johnny.placement": "rtx-pair-systemd"}}

    reg = {"models": {"gemma-4-31B-it": {"capabilities": {"multimodal": False},
                                         "placements": [{"id": "rtx-pair-systemd", "backend": "systemd"}]}}}
    monkeypatch.setattr(service, "all_seats", lambda cfg: [_Seat()])
    monkeypatch.setattr(service, "_find", lambda seats, target: (seats[0], None))
    monkeypatch.setattr(service, "_endpoint", lambda cfg, seat: "http://127.0.0.1:8121/v1")
    monkeypatch.setattr(service, "_registry_safe", lambda: reg)
    monkeypatch.setattr(service, "seat_guidance", lambda *a, **k: None)
    monkeypatch.setattr(service.sources, "metrics_for_port", lambda *a, **k: {})
    res = service.resolve("chat", cfg={})
    assert res["capabilities"] == {"modalities": ["text"]}
