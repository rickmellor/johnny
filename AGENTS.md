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
- vLLM-ROCm default is **v0.27.1** (bumped 2026-08-23 after a per-seat validation; see
  the NOVA MegaPlan `20260823-vllm-rocm-0-27-1-fleet-validation…`). **GPU P2P is ON
  since 2026-09-03** — the "gfx1201 RCCL bug" (`hipIpcGetMemHandle: invalid argument` on
  every image ≥0.21) was `HSA_ENABLE_IPC_MODE_LEGACY=1` baked into those images' env,
  not RCCL; `multi_gpu_env` now sets `HSA_ENABLE_IPC_MODE_LEGACY=0` + NCCL_PROTO=Simple
  for TP≥2 (RCCL logs `via P2P/IPC`; 19.7 vs 11.6 GB/s all-reduce busbw). BIOS ACS was
  disabled the same day (full KFD p2p_links mesh, 25 GB/s D2D). The old
  NCCL_P2P_DISABLE=1 + RCCL_NET=Socket pair is gone from code and registry
  (`registry.yaml.bak-20260903-224027-pre-p2p`). NCCL_PROTO=Simple stays until an image
  carries RCCL PR #2187 (LL-protocol deadlock, a separate bug). **Gemma-4 is pinned per-placement to
  v0.20.2**: 0.27.0/0.27.1 can't load it (transformers 5.15 per-layer `head_dim`,
  vllm#52768) and the nightly that can is 13–15 % slower here — re-test at 0.28.
  Smoke-test TP=2 + `johnny bench perf` before any further image bump.
- llama.cpp seats use the local `johnny-llamacpp-*:gfx1201` images.
- **No desktop on the compute GPUs (2026-08-23).** The box boots to `multi-user.target`
  (`sudo systemctl start gdm` if a GUI is ever needed). Reason: GPU3 (`0000:03:00.0`) was
  the display GPU; a long Triton kernel there (int4-KV prefill) tripped the gfx-ring
  watchdog 232× (`ring gfx_0.0.0 timeout → reset → "device wedged, but recovered"`) and
  afterwards every multi-GPU seat decoded at ~½ speed until reboot. Single-GPU, PCIe, VRAM
  and CPU all measured normal — the cross-GPU RCCL path (host-bounce at the time) was what degraded.
  If you ever see amdgpu ring resets in `journalctl -k`, treat that GPU as suspect.
- **Qwen3.5-family (GDN) warm-up — deep-prefill trigger:** a freshly launched Qwen3.6/3.8
  seat decodes at ~½ speed **until it serves its first deep prefill**; one ~32K-token
  prompt flips the process to full speed permanently (validated: 18 min of short-prompt
  load stayed at 14 tok/s/stream; one 32K prefill → 40.8 single / 142 c4 within a minute).
  **Automated (2026-08-23): `johnny up --wait` and `johnny profile up` warm GDN seats by
  default** (`engine/warmup.py`; `--no-warmup` skips; profile up waits for GDN seats even
  without `--wait` so the fleet comes up at rated speed). gemma seats are unaffected. See
  `scratch/rdna4-kernel-tuning-and-4bit-kv-report-20260823.md` §E.
- **Qwen3.8 `reasoning_effort` (2026-08-24):** its chat template defaults to **`xhigh`**, which badly
  hurts agentic throughput/quality. Pin it per-seat with
  `--default-chat-template-kwargs '{"reasoning_effort":"medium"}'` (or `low`) — AutomationBench pass rate
  goes 14.3 % (xhigh) → 30–40 % (low/medium), vs qwen-27b-coder's 16.7 %. Per-request
  `chat_template_kwargs` overrides. Placements `effort-{low,medium}-{tp2,tp4}` carry it. Report §G.
- **Per-seat `guidance` (2026-08-24).** A profile seat may declare `guidance: roadmap` when its
  model executes delegated work well but plans it poorly. `johnny resolve <role> --json` surfaces
  it (`guidance` field, follows `role_aliases`), and input's `spawn_agent` reads it to hand that
  sub-agent an explicit ordered brief instead of an open question. Set on `gemma-tp4` from
  measurement: gemma-4-26B scores PlanBench 34% / AutomationBench 0–6.7% on long loops, yet 5/5
  on short delegate tasks — where a roadmapped brief cut it 6.4 → 5.1 tool steps and kept the
  ambiguous task off the 8-step ceiling. Efficiency + margin, not a correctness crutch.
  **Contract note:** `resolve` takes a johnny *role* (`coder`), not a router pin. SAINT pins
  are full `saint-<backend>` ids, and `johnny resolve saint-local-coder` correctly reports
  `state: absent` — a client holding a pin must fold it back to the role before asking
  (input's `seat_key`). Getting this wrong loses the guidance silently, since the field is
  optional and a miss is indistinguishable from a seat that declares none.
- **RDNA4 kernel tuning + 4-bit KV (2026-08-23)** — measured and parked; see
  `scratch/rdna4-kernel-tuning-and-4bit-kv-report-20260823.md`. Tuned block-FP8 / MoE
  Triton configs live in `johnny-vllm-rocm:<tag>-gfx1201` images (+2–4 % on the Qwen
  seats, ~0 on gemma); `int4_per_token_head` / `turboquant_4bit_nc` KV are not usable
  here (dequant-in-the-hot-loop → 3–7× slower, engine death ≥200K, TQ crashes). bf16 KV.
- DeepSeek-family archs need `-fa off` on RDNA4 (head-dim-512 kernel
  unstable); induction handles this by arch name.

## Context safety — `max_model_len` is a request, not a guarantee

**2026-08-06 incident.** Two `Ornith-1.0-397B-Featherweight-v0` llama.cpp placements
were found (manual, ad hoc agent work, real needle-in-haystack testing with live
`rocm-smi` VRAM monitoring) to **reproducibly crash the whole container** — HIP
`Memory access fault... page not present` on GPU0, i.e. real VRAM exhaustion — at
real prefill depths well short of their configured `max_model_len`. One placement's
nominal cap was 262144 tokens but it crashed at a real depth of n_tokens=59392
(~23% of the configured cap); a sibling crashed at 47104 tok/slot against a
409600-total/parallel=4 config. Both got fixed by empirically walking real depths
until a genuinely safe ceiling was found — see those two placements' `extra.note` in
`registry.yaml` for the full write-up (root cause: GPU0's VRAM use grows with real
token depth reached during prefill, at an *accelerating* rate deep into the window;
it is not simply proportional to the configured `max_model_len`, so a smaller
`max_model_len` barely buys headroom and a larger one doesn't cost proportionally
more — only a real deep prefill tells you where the wall actually is).

**Why this matters generally.** `max_model_len` is a request the launcher honors by
sizing a KV-cache reservation up front. It says nothing about whether the seat
survives an actual prompt that reaches that many tokens — that depends on real,
depth-dependent VRAM growth (compute buffers, context checkpoints, backend-specific
overhead) that a static config number cannot capture. A placement can look
completely healthy — validated status, good perf numbers, a configured context that
matches or even undershoots the model's trained `native_context` — and still be
silently unsafe the first time something actually fills its context window. That is
exactly the gap that caused this incident, and there is no reason to believe it's
unique to these two placements — any placement with a large configured context and
no real deep-prefill test carries the same unverified risk.

**The fix: `johnny bench <target> --suite ctxsafe`.** This formalizes the manual
methodology above into reusable infrastructure (`src/johnny/ctxsafe.py` +
`bench.py`'s `_run_ctxsafe`): real needle-in-haystack prompts at progressively
deeper depths, a live, disposable probe seat, `rocm-smi` polled every ~2s *during*
each request (not just before/after), real crash detection via `docker inspect`
(not just a dropped connection), and multiple confirmation trials near the
configured ceiling. It writes `quality.ctxsafe` — `verified_safe_tokens`,
`crashed_at`, `tested_depths`, `vram_peak_gb` — which is a **third, distinct number**
from the model's trained `capabilities.native_context` and the placement's
configured `knobs.max_model_len`; `registry show`'s CONTEXT column and `johnny
bench`'s console output both surface the gap between them when it's non-zero.

Run it (and check the result) before treating any placement with a non-trivial
context as production-ready — especially before relying on it for real long-context
work. `johnny registry validate` flags placements with a large configured
`max_model_len` (≥32768 tokens — see `normalize.CTXSAFE_THRESHOLD_TOKENS`) that have
never had a `ctxsafe` run, the same way it already flags stale/unmeasured perf
numbers, so this doesn't depend on anyone remembering to check by hand. For a
placement whose native/configured context is too large to sweep to in one run (e.g.
a seat with a 1M-token `max_model_len`), use `--limit <tokens>` to cap the deepest
depth tested rather than skipping the suite entirely — a partial, honestly-scoped
verification is far better than none, and `quality.ctxsafe.limited` records that the
full configured cap wasn't reached.

## Gotchas

- llama-server splits `-c` across `--parallel` slots: a 32k/par4 seat gives
  each request only 8k. Agent-facing seats want `parallel: 1`.
- GGUF `general.file_type` is an int enum, not a string — quant names are
  derived from filenames/headers (`registry/normalize.py:identity_gaps`).
- Placement `extra.mounts` (vLLM) is a list of docker `-v` specs (`host:container[:ro]`, host
  `~`-expanded) emitted before the image — how a seat runs an image with patched vLLM source
  files without a rebuild. Qwen3.8-Flash-Next rides on it (seven files from `~/scratch/fnrepo`,
  the checkout of `rickmellor/qwen3.8-flash-next-rdna4`); placements
  `flashnext-awq-tp4-{,ep-}p2p-piecewise-mtp-mml131072`.
- **Boot default profile is `flashnext` since 2026-09-04** (`johnny-profile@flashnext.service`):
  Flash-Next TP4 + expert-parallel as `chat` with `coder` aliased onto it, nomic embed + 1B
  classifier on CPU. GPUs 4,5 are free. `split`/`gemma-tp4` remain as profiles, not enabled.
- Placement `extra.extra_flags` passes raw llama-server flags (e.g.
  `["-ctk", "q8_0"]` for quantized KV).
- SAINT (~/repos/saint-router) resolves seats by **role** (`coder`/`chat`/
  `embed`/`classifier`) by shelling out to this CLI — `johnny resolve` output
  is a compatibility contract. Boot default is profile `coder` (Ornith serves
  chat+coder via role_aliases).

## WIP — 2026-08-06 night: new bench suites, paused for a GPU upgrade

Session paused mid-project (not finished, not abandoned) — Rick is adding a 5th/6th
R9700 to the box tomorrow before we continue. Read this before touching benchmarks
or the registry again.

**Fleet state:** everything is torn down — `docker ps` has no `johnny-*` containers,
all GPUs idle. This is deliberate (explicit "down everything, going to bed" — not
the usual "restore the standing seat" convention); don't auto-restore a baseline
seat on the next session without checking what Rick actually wants first, especially
post-upgrade when `hardware.detect()` will report a different card count.

**Code state:** a large pile of uncommitted work sits in the tree (`git status`) —
this session's `registry/normalize.py` NAS-aware fix + the new `automationbench`
suite (`bench.py`, `cli.py`), plus earlier tonight's other agents' fixes (NAS mounts,
llamacpp induction/resolve_image, ctxsafe). None of it has been committed — nobody
asked for that yet. Don't commit without asking; do expect `git status` to look busy.

**Registry model comparison work (done):** `deepseek-v4-flash-0731` was inducted for
real (`induct-llama-ngl999-par4-mml32768`: ngl=999, parallel=4, mml=32768,
`extra.extra_flags: ["--fit", "off", "--split-mode", "layer"]` — this image's
llama-server defaults `--fit` to `on`, which crashes on this box's multi-GPU +
high-expert-count combo, see the placement's own `extra.note`). HumanEval 90.24%
(148/164) — a statistical tie with the Preview checkpoint (90.85%, 149/164), still
behind qwen-27b-coder (96.34%) and qwen-122b-awq (93.29%). ctxsafe found
verified_safe=4,915 tok against a configured 32,768 — not a crash, just the
`-c`/`--parallel` split gotcha above (par=4 → 8,192 tok/slot in practice) biting a
long probe request. The original `deepseek-v4-flash` (Preview, NAS-only) entry is
untouched, both coexist for comparison.

**New agentic-benchmark suites — researched 9, queued 4, implemented 1 (partial):**

Researched (web search, not yet built) against DeepSeek's own 0731-vs-Preview
release table: Terminal-Bench 2.1, DeepSWE, CyberGym, Toolathlon-Verified, NL2Repo-
Bench, DSBench-FullStack/Hard, Agents' Last Exam, AutomationBench (Public). All nine
are **agentic/multi-turn** (tool-calling loop + environment feedback) — a real gap
vs. every existing suite here (`arc`/`icl`/`needle`/`humaneval` are one-shot
completions). Fit assessment landed on a queue of four, cheapest-to-hardest:

1. **AutomationBench (Public)** — zapier/AutomationBench, MIT, fully self-contained
   (47 simulated SaaS tools, no live creds). **Wired in** as `johnny bench <target>
   --suite automationbench [--domains sales|marketing|...|all] [--limit N]
   --concurrency N` (see `bench.py`'s `_run_automationbench` docstring). Vendored
   checkout self-bootstraps via `uv sync` at
   `~/.local/state/johnny/tools/automationbench` (its own Python-3.13/`uv` project —
   doesn't fit johnny-fleet's venv, same reasoning as lm-eval for `humaneval` but one
   step further since there's no lighter pip-installable form). Needs `uv` on PATH.
   - **Real bug found + fixed tonight:** the export JSON path is fixed per model, so
     a run killed/crashed before writing its own output left a **prior run's stale
     file in place** — `_run_automationbench` was reading that stale file and
     reporting it as a fresh result (caught via a `total: 5` vs `num_examples: 25`
     mismatch in the registry). Fixed: the export path is `unlink`ed before every
     invocation, and a non-zero exit code is now treated as failure regardless of
     whether a file happens to exist. Bad registry write already reverted (backup:
     `registry.yaml.bak-20260806-233622-ab-badwrite`).
   - **No valid score exists yet for anything.** The only completed run was a 5-task
     smoke test on `qwen-122b-awq` (0/5 pass, 14.17% avg partial credit) — too small
     to mean anything, correctly excluded from the registry.
   - **Real cost lesson:** a single task's tool-calling transcript can run 400K+
     cumulative tokens across ~50 steps; one task took 38 minutes solo. A 25-task
     single-domain run at `--concurrency 25` against `qwen-122b-awq` (vLLM, fast)
     was still going after 45+ minutes (4/25 done) when it got killed for the night —
     budget on the order of **an hour+ for 25 tasks**, not minutes. Consider a
     smaller `--limit` (10?) for the next attempt, or just let a full domain run
     unattended overnight instead of mid-session.
   - vLLM placements are the right target for this suite — `max_model_len` is the
     real per-request ceiling there. llama.cpp placements split `-c` across
     `--parallel` (see Gotchas above) and will abort long tool-use transcripts; this
     bit the very first exploratory smoke test (against the MiniMax llama.cpp seat).
2. **Toolathlon-GYM** (not the raw "Toolathlon-Verified" split DeepSeek's table
   uses — that needs live third-party API creds) — self-contained variant, 503
   tasks, 25 mocked MCP servers, no external calls. Not started.
3. **Terminal-Bench 2.1** — 89 CLI tasks, real per-task Docker sandboxes + an agent
   shell loop (`tbench` CLI). Most industry-recognized name of the four, biggest
   lift. Not started.
4. **NL2Repo-Bench** — 104 tasks, single NL spec → full installable Python repo,
   graded against the real upstream pytest suite. No external creds needed, but
   each task is a long multi-file agentic generation — expect it to be slow on this
   hardware too. Not started.

Deliberately not pursuing (see full reasoning earlier in this session's transcript):
**DeepSWE** (not really its own released benchmark — redundant with Terminal-
Bench/NL2Repo territory), **CyberGym** (real exploit execution, heavy build farm),
**DSBench-FullStack/Hard** (DeepSeek's exact subset cut is unconfirmed, dataset-
heavy), **Agents' Last Exam** (90% private task pool, needs GUI/desktop control —
not really self-hostable).

**GPU upgrade (tomorrow, before resuming any of this):** adding a 5th/6th R9700
changes `hardware.detect()`'s reported card count. Known implications to check once
it lands, before trusting any new induction/tune output:
- `validation_key.hardware_fingerprint` (`4xamd-gfx1201-32g` today) will read
  `5x...`/`6x...` for anything tuned post-upgrade — existing 4-GPU placements keep
  their old fingerprint (correct — they were validated on 4 cards) and won't
  auto-flag as stale (`normalize._is_stale` compares docker image tags, not the
  fingerprint, so this is silent — worth eyeballing `johnny registry validate`
  output post-upgrade anyway).
- Existing placements are pinned to `gpu_count: 4`/`tensor_parallel_size: 4` and
  won't automatically spread onto the new card(s) — that's fine, no action needed
  until something is deliberately re-tuned wider.
- TP=5 is very unlikely to be viable for most architectures (attention-head-count
  divisibility); TP=6 depends on the model. A 5th card more likely wants to serve as
  a 5th independent seat (e.g. a small model, or `johnny down`'s embed/classifier
  CPU seats moved onto GPU) rather than widening TP on the big MoE models. Worth
  deciding deliberately, not assuming wider TP "just works."
- New VRAM headroom (5×32=160GB / 6×32=192GB) opens the door to less-aggressive
  quants of models already in the registry (e.g. a Q4/Q5 DeepSeek-V4-Flash instead
  of IQ3_XXS) — worth a registry scan for "what would this model's quality look
  like with more headroom" once the hardware lands, separate from the benchmark
  queue above.
