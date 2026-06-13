# Handoff — johnny v2 + SAINT integration (final, approved)

Two repos, one contract. The layout in this bundle mirrors where files land.

## johnny/  → the johnny repo (new)
- `PLAN.md`   — the approved design & phased roadmap. Build starts at **P0**
                (package skeleton, config, doctor/init, `schema_version` + `migrate`).
- `README.md` — repo front page. Names decided: PyPI **`johnny-fleet`**, CLI `johnny`.

Suggested first Claude Code prompt: "Read PLAN.md. Implement Phase 0 exactly as
specified, including the P0 verify criteria."

## saint/  → the SAINT repo (formerly goorouter)
- `openspec/changes/add-johnny-integration/` — a complete OpenSpec change in the
  repo's existing idiom (proposal, design, tasks skeleton, three delta specs:
  config / routing / observability). Drop into the repo as-is.
- The repo-wide rename goorouter → SAINT rides along with this change's
  implementation: PyPI **`saint-router`**, CLI `saint`, virtual models
  `saint-auto` / `saint-explain` / `saint-<backend>`, config dir `~/.config/saint/`.
  The `[johnny]` config block keeps its name — it references the other tool.

## Cross-repo contract (versioned JSON; defined in johnny PLAN §3.13)
- `johnny resolve <role|seat> --json` → `{seat, endpoint, model,
  state: ready|loading|absent|failed, eta_s, queue_depth}` — hot path, ~1 s cache;
  `eta_s` is best-effort from recorded historical cold-start durations.
- `johnny up <role> --wait --json` → ready endpoint, or immediate
  `{state: loading, eta_s}`. The router never blocks a request on a load.
- `johnny pin|unpin <seat> [--ttl]` — reaper exemption. Ephemeral pins live in
  johnny's telemetry SQLite (docker labels are immutable post-create).
- Telemetry: SAINT **provides** records in johnny's normalized ingest schema by
  appending JSONL (single-line `O_APPEND` records) to
  `$XDG_STATE_HOME/johnny/ingest/`; johnny rotates-before-read on its tick.
- SAINT fallback order for a non-ready johnny-bound seat:
  `while_loading` (per-backend → global) → the backend's own **static baseline**
  (unless `johnny_only = true`) → `default_on_failure`.

## Order of operations
1. Claim the PyPI names now: `johnny-fleet`, `saint-router` (plus `johnnyctl`,
   `johnny-llm`, `saintrouter` as defensive grabs — `johnny` and `johnny5` are taken).
2. johnny: scaffold the repo (its own OpenSpec mirroring SAINT's is recommended)
   and implement **P0**, then P1 → P3 in order. The reaper, `loading` state, the
   ingest spool, and `resolve` all land at P3.
3. SAINT: implement `add-johnny-integration` any time — its johnny calls are
   mockable (stub `resolve --json`); the live v0 contract arrives with johnny P3.
