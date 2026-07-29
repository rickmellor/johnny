# AGENTS.md — working in this repo

johnny is a local-inference fleet manager (typer CLI, `src/johnny/`), launching
vLLM / llama.cpp seats as docker containers. `PLAN.md` is the authoritative v2
design doc; `HANDOFF.md` is migration history from the bash-era johnny.

## Live editable install

The `johnny` on PATH (`~/.local/bin/johnny`) is a **pipx editable install of
this repo** — edits to `src/johnny/` are live immediately, no reinstall.
`python` is not on PATH; use the venv interpreter:

    ~/.local/share/pipx/venvs/johnny-fleet/bin/python -m pytest tests/ -q

## State layout (not in the repo)

- `~/.config/johnny/` — `config.yaml`, `registry.yaml`, `profiles.yaml`
- `registry.yaml` is **machine-written**: mutate via the CLI or
  `johnny.registry.store` (load/save), never by hand-editing YAML. House
  convention: copy a `registry.yaml.bak-<timestamp>` before any surgery.
- Models live under `roots.models_dir` (`/home/rick/models`); registry
  `local_path`s are relative to it.

## Registry philosophy

Shape and numbers are different things. `registry normalize` fills structure
and derivable identity (params/quant from GGUF headers or naming) but **never
fabricates a tok/s** — missing perf stays `unmeasured` and is `johnny tune`'s
job. Keep that split when adding features.

## This box / runtime pins

- 4× AMD R9700 (gfx1201, 32 GB each), ROCm.
- vLLM-ROCm is **pinned to v0.20.2** — 0.21+ breaks multi-GPU here (hangs,
  NCCL/HIP failures). Smoke-test TP=2 before any image bump.
- llama.cpp seats use the local `johnny-llamacpp-*:gfx1201` images.
- DeepSeek-family archs need `-fa off` on RDNA4 (head-dim-512 kernel
  unstable); induction handles this by arch name.

## Gotchas

- llama-server splits `-c` across `--parallel` slots: a 32k/par4 seat gives
  each request only 8k. Agent-facing seats want `parallel: 1`.
- GGUF `general.file_type` is an int enum, not a string — quant names are
  derived from filenames/headers (`registry/normalize.py:identity_gaps`).
- Placement `extra.extra_flags` passes raw llama-server flags (e.g.
  `["-ctk", "q8_0"]` for quantized KV).
- SAINT (~/repos/saint-router) resolves seats by **role** (`coder`/`chat`/
  `embed`/`classifier`) by shelling out to this CLI — `johnny resolve` output
  is a compatibility contract. Boot default is profile `coder` (Ornith serves
  chat+coder via role_aliases).
