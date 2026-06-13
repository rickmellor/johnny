# johnny v2 — a shareable local inference environment manager

A clean-slate, portable rewrite of the `johnny` CLI: a declarative-registry-driven
tool for managing local LLM inference across **pluggable backends** (vLLM first,
LM Studio next, Ollama later), multiple models per box, and multiple machines —
with an automated model-induction pipeline that tunes (and optionally benchmarks)
new models into optimal per-hardware configs.

Stack: **Python + Textual/Rich** (TUI is the final, deferred phase). vLLM runtime: docker.

**Names:** PyPI distribution **`johnny-fleet`**; CLI command and import path **`johnny`**.
Companion request router: **SAINT** (PyPI `saint-router`, CLI `saint`) — a peer data
plane integrated over a JSON contract, never a johnny component.

See [PLAN.md](PLAN.md) for the full design and phased implementation roadmap.

## Status

**Plan approved — build begins at Phase 0** (package skeleton + config + doctor/init).
The SAINT integration's v0 contract (`resolve` / `up --wait` / the ingest spool) comes
alive at Phase 3; SAINT's side is specified in its repo's OpenSpec change
`add-johnny-integration`.

## Highlights (from the plan)

- **Portable / shareable** — no hardwired paths or hardware; detects GPU vendor,
  count, VRAM, arch, and natively-accelerated dtypes at runtime. Versioned schemas
  with migrations from day 0.
- **Declarative registry** — the source of truth for models, validated placement
  configs (per GPU-count/TP, quant, context, MTP, KV-dtype), profiles, and fleets.
- **Multi-seat control** — runs co-existing models across the GPUs (e.g. orchestrator
  + coder + embeddings), replacing the single-LLM-on-8000 assumption.
- **Early idle reaper** — evicts idle seats so the cards reach deep idle (~16–18 W vs
  ~95 W); cron-able and stateless, landing long before the full JIT router.
- **Model induction** — `johnny induct <model>` runs a seeded search (not a brute grid)
  across viable placements and writes the optimal parameters into the registry.
- **Control plane, not data plane** — request routing (SAINT, the classifier
  router) integrates as a peer over a JSON contract; johnny manages seat lifecycle and
  liveness, SAINT classifies and picks per request — johnny supplies the on-demand
  loading SAINT's static policy grid structurally lacks.
- Reuses the proven mlops scripts (bench, wait-ready, audit, probes, eval harnesses)
  as orchestrated subprocesses rather than reimplementing them.
