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

Built and in daily use. Seats, profiles, the registry, induction/tuning, benchmarking,
the reaper, telemetry and the SAINT resolve contract are all live; the Textual dashboard
(`johnny tui`) and the multi-node controller (`johnny nodes`, `johnny daemon`) are the
newest and least exercised surfaces. [PLAN.md](PLAN.md) is the original design record —
useful for *why*, no longer accurate as a schedule.

## What it does

A box with several GPUs can host several models at once. johnny decides what runs where,
keeps the placements that actually work written down, and answers "where do I send this
request right now" for whatever sits in front of it.

- **Declarative registry** — the source of truth for models and their *validated
  placements*: per GPU-count/TP, quant, context length, KV dtype, image pin, environment.
  A placement is a configuration that has been proven to load and serve on this hardware,
  not a guess.
- **Seats** — `up` / `down` / `swap` a named model instance on chosen cards and a port,
  without disturbing its siblings. Seats coexist (e.g. chat + coder + embeddings).
- **Profiles** — a named fleet of seats brought up together (`johnny profile up <name>`),
  optionally at boot via a systemd user unit. Roles (`chat`, `coder`, `embed`,
  `classifier`) name what a seat is *for*; `role_aliases` lets one seat answer to several.
- **Induction & tuning** — `johnny induct <model>` runs a seeded search (not a brute grid)
  over viable placements and writes the winner into the registry. `johnny bench` scores a
  placement for both throughput and quality, and records the result.
- **Idle reaper** — evicts idle, unpinned seats so the cards drop to deep idle. Stateless
  and cron-able; `pin` exempts a seat.
- **Resolve contract** — `johnny resolve <role> --json` returns the live endpoint, model,
  readiness and ETA. This is the hot path a router calls per request; it is deliberately a
  CLI/HTTP contract, never a library import, so the router stays independent of johnny.
- **Control plane, not data plane** — johnny manages seat lifecycle and liveness; a router
  (SAINT) classifies and picks per request. johnny supplies the on-demand loading a static
  policy grid structurally lacks.

## Command surface

```
Seats      up · down · swap · reap · pin · unpin · resolve
Observe    status · logs · metrics · alive · tui
Models     induct · tune · bench · search · download · login · registry
Fleet      profile · nodes · daemon · provider
Setup      init · doctor · migrate · hinfo · cleanup · rm · version
```

`johnny init` detects the box and writes a starter config; `johnny doctor` preflights
docker, the GPU runtime, arch support, disk and backends before you spend an hour on a
load that was never going to work.

## Benchmarking

`johnny bench <placement> --suite <name>` records into the registry alongside the placement,
so a config carries its own evidence:

| suite | measures |
|---|---|
| `perf` | throughput: single-stream and peak concurrent tok/s |
| `arc` · `icl` · `humaneval` | reasoning, in-context learning, code generation |
| `needle` · `depth` · `ctxsafe` | long-context retrieval and where a seat stops being safe |
| `automationbench` · `planbench` | agentic tool loops, and planning in isolation |

`ctxsafe` matters more than it sounds: `max_model_len` is a *request*, not a guarantee, and a
seat that launches at a given context can still fail at depth.

## Per-seat agent guidance

A profile seat may carry a `guidance` hint describing how *clients* should brief work sent to
it. It changes nothing about how johnny launches or manages the seat — johnny only declares it
and surfaces it through the resolve contract, so a client can adapt without hardcoding model
names.

```yaml
seats:
  - model: <model-id>
    placement: <placement-id>
    port: 8002
    role: chat
    guidance: roadmap      # this model executes delegated work well but plans it poorly
role_aliases:
  coder: chat              # 'coder' resolves to the chat seat — and inherits its guidance
```

```console
$ johnny resolve chat --json
{ "seat": "...", "endpoint": "...", "model": "...", "state": "ready", "guidance": "roadmap" }
```

`roadmap` means *give this seat an explicit ordered brief rather than an open-ended question*.
Resolution follows `role_aliases`, so a role pointed at another seat inherits that seat's hint.
Seats that declare nothing return `"guidance": null`, and clients are expected to treat the
field as optional — it is advisory metadata, not a contract a client must honour.
