"""Profiles — named fleets (human-authored, `profiles.yaml`).

A profile declares the seats that make up a working fleet (model + placement +
port + role), so the whole set can be brought up with one command and re-created
after a reboot. Roles are the contract SAINT resolves against (`johnny resolve
<role>` → live endpoint); explicit ports keep SAINT's static fallbacks valid.
"""

from __future__ import annotations

from .. import config as C

_HEADER = f"# johnny profiles — schema v{C.PROFILES_SCHEMA_VERSION} (human-authored fleets)"

# Seats a profile requires: model+placement say WHAT to run, port+role say WHERE
# and AS-WHOM (SAINT's johnny_role binding). GPUs are deliberately not stored —
# they're assigned at launch from what's free.
_REQUIRED = ("model", "placement", "port", "role")


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


def all_profiles() -> dict:
    return load().get("profiles") or {}


def save(name: str, profile: dict) -> None:
    doc = load()
    doc.setdefault("schema_version", C.PROFILES_SCHEMA_VERSION)
    doc.setdefault("profiles", {})[name] = profile
    C.write_yaml(C.get_paths().profiles_file, doc, header=_HEADER)


def remove(name: str) -> bool:
    doc = load()
    if name not in (doc.get("profiles") or {}):
        return False
    del doc["profiles"][name]
    C.write_yaml(C.get_paths().profiles_file, doc, header=_HEADER)
    return True


def _seat_labels(seat) -> dict:
    return (seat.extra or {}).get("labels") or {}


def _infer_role(model_id: str, placement_id: str | None, reg: dict) -> str | None:
    """Only pooling (embeddings) placements have a derivable role; chat/coder/etc.
    are user intent and must come from --role."""
    from ..registry import store

    m = store.get(reg, model_id) or {}
    for p in m.get("placements") or []:
        if placement_id and p.get("id") != placement_id:
            continue
        if (p.get("extra") or {}).get("runner") == "pooling":
            return "embed"
    return None


def capture(cfg: dict | None = None, roles: dict | None = None) -> dict:
    """Snapshot the live johnny-managed fleet into profile seats.

    Containers without `johnny.*` labels are invisible to johnny and are reported
    in `skipped` (e.g. a hand-started server that predates adoption). Seats whose
    role can't be inferred and isn't in `roles` come back without one — the caller
    decides whether that's an error (save requires roles).
    """
    from ..registry import store
    from ..telemetry import collect
    from . import all_seats, load_config

    cfg = cfg if cfg is not None else load_config()
    reg = store.load()
    roles = roles or {}
    seats, skipped = [], []
    for s in all_seats(cfg):
        labels = _seat_labels(s)
        model = labels.get("johnny.model") or s.model
        if not labels.get("johnny.managed") and not labels.get("johnny.model"):
            skipped.append(s.name)
            continue
        seat: dict = {"model": model}
        if labels.get("johnny.placement"):
            seat["placement"] = labels["johnny.placement"]
        if s.port:
            seat["port"] = s.port
        role = roles.get(model) or _infer_role(model, seat.get("placement"), reg)
        if role:
            seat["role"] = role
        if collect.is_pinned(s.name):
            seat["pinned"] = True
        seats.append(seat)
    return {"seats": seats, "skipped": skipped}


def validate(profile: dict, reg: dict, hardware=None, name: str | None = None) -> tuple[list[str], list[str]]:
    """Structural checks → (errors, warnings). Mirrors registry.schema style:
    errors block a save; warnings inform. `name` identifies this profile so the
    cross-profile role check skips its own saved copy (e.g. on a --force re-save)."""
    from ..registry import store

    errors: list[str] = []
    warnings: list[str] = []
    seats = profile.get("seats") or []
    if not seats:
        errors.append("profile has no seats")
        return errors, warnings

    seen: dict[str, set] = {"model": set(), "port": set(), "role": set()}
    gpu_need = 0
    for i, seat in enumerate(seats):
        who = seat.get("model") or f"seat[{i}]"
        for field in _REQUIRED:
            if not seat.get(field):
                errors.append(f"{who}: missing required '{field}'")
        m = store.get(reg, seat.get("model") or "")
        placement = None
        if seat.get("model") and not m:
            errors.append(f"{who}: model not in the registry")
        elif m and seat.get("placement"):
            placement = next((p for p in m.get("placements") or []
                              if p.get("id") == seat["placement"]), None)
            if not placement:
                errors.append(f"{who}: placement '{seat['placement']}' not found on model")
        if placement:
            gpu_need += int((placement.get("knobs") or {}).get("gpu_count") or 0)
        # duplicates: model collides with launch.up's model-keyed idempotency;
        # port/role must be unambiguous by design.
        for field in ("model", "port", "role"):
            v = seat.get(field)
            if v is None:
                continue
            if v in seen[field]:
                errors.append(f"{who}: duplicate {field} '{v}' in profile")
            seen[field].add(v)

    if hardware is not None and gpu_need > len(getattr(hardware, "gpus", []) or []):
        warnings.append(f"placements want {gpu_need} GPUs but this box has "
                        f"{len(hardware.gpus)} — bring-up will be partial")
    # A role defined by another profile too makes `resolve <role>` first-match
    # ambiguous (role_to_model scans all profiles).
    my_roles = {s.get("role") for s in seats if s.get("role")}
    for other_name, other in all_profiles().items():
        if other is profile or other_name == name:
            continue
        for s in other.get("seats") or []:
            if s.get("role") in my_roles:
                warnings.append(f"role '{s['role']}' is also defined in profile "
                                f"'{other_name}' (resolve uses first match)")
    return errors, warnings


def up_profile(name: str, wait: bool = False, cfg: dict | None = None) -> dict:
    """Bring up every seat in a profile, best-effort: one seat's failure is
    recorded and the rest still launch. Idempotent — already-running models
    return action=exists (launch.up is model-keyed). Pinned seats get an
    indefinite pin so the reaper keeps them warm."""
    from ..telemetry import collect
    from . import all_seats, launch, load_config

    cfg = cfg if cfg is not None else load_config()
    prof = get_profile(name)
    if prof is None:
        raise launch.PlacementError(f"no profile '{name}' (see `johnny profile list`)")

    results = []
    for seat in prof.get("seats") or []:
        model = seat.get("model")
        entry = {"model": model, "role": seat.get("role"), "port": seat.get("port")}
        # launch.up doesn't collision-check an explicit port (it only allocates
        # around live seats when port is None) — a taken port would surface as a
        # raw docker bind failure. Pre-check for a clear error instead.
        holder = next((s for s in all_seats(cfg)
                       if s.port == seat.get("port")
                       and (_seat_labels(s).get("johnny.model") or s.model) != model), None)
        if holder:
            entry.update({"action": "error", "error": f"port {seat.get('port')} held by {holder.name}"})
            results.append(entry)
            continue
        try:
            r = launch.up(model, placement_id=seat.get("placement"),
                          port=seat.get("port"), wait=wait)
        except launch.PlacementError as e:
            entry.update({"action": "error", "error": str(e)})
            results.append(entry)
            continue
        if seat.get("pinned") and r.get("seat"):
            collect.add_pin(r["seat"])  # indefinite: profile seats stay warm
        entry.update(r)
        results.append(entry)
    return {"profile": name, "seats": results}


def down_profile(name: str, drain: bool = False, cfg: dict | None = None) -> dict:
    """Stop every running seat of a profile (launch.down also clears its pin)."""
    from . import all_seats, launch, load_config

    cfg = cfg if cfg is not None else load_config()
    prof = get_profile(name)
    if prof is None:
        raise launch.PlacementError(f"no profile '{name}' (see `johnny profile list`)")

    results = []
    for seat in prof.get("seats") or []:
        model = seat.get("model")
        entry = {"model": model, "role": seat.get("role")}
        live = launch._find_seat(all_seats(cfg), model)
        if not live:
            entry["action"] = "absent"
            results.append(entry)
            continue
        try:
            r = launch.down(live.name, drain=drain)
            entry.update(r if isinstance(r, dict) else {})
            entry.setdefault("action", "stopped")
            entry["seat"] = live.name
        except launch.PlacementError as e:
            entry.update({"action": "error", "error": str(e)})
        results.append(entry)
    return {"profile": name, "seats": results}
