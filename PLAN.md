# johnny v2 — a shareable local inference environment manager (design & plan)

> **TL;DR.** Rewrite a personal bash CLI (`johnny`) that today manages a *single*
> vLLM model into a portable, shareable Python tool for managing **local LLM inference
> across pluggable backends** (vLLM first, **LM Studio** next, Ollama later), multiple
> models per box, and **multiple machines** you own. The source of truth is a
> **declarative registry** of models + validated, backend-specific launch configs. Two
> headline capabilities: **model induction** (point it at a model and it auto-tunes the
> optimal run parameters for *your* hardware — benchmarking is opt-in) and **first-class
> observability** (normalized token-rate / TTFT / context / concurrency / queue-depth
> telemetry per provider+model). Get the data model, backend abstraction, and engine
> right first; a Textual TUI comes last. This doc is for review — see **Open questions**.
>
> **This rev:** the **idle reaper** moves up to P3 (the ~$100/mo feature — deep-idle the
> cards without waiting for the JIT router); **`loading` is a first-class seat state**
> (occupancy from docker labels, not readiness); telemetry persists from P3 with
> **source-tagged metrics**; induction is a **seeded search**, not a brute grid;
> `schema_version` + migrations from P0; and **SAINT** (the classifier request
> router, now reviewed from source) integrates as a **peer data plane over a JSON
> contract** (§3.13) — johnny supplies the **liveness + on-demand loading** SAINT
> structurally lacks, never owning the request path or SAINT's policy grid.

---

## 1. Background (read this first if you're new to the project)

**The box it grew up on.** A workstation with a Threadripper PRO 3945WX, 256 GB RAM,
and **4× AMD R9700 AI Pro GPUs** (RDNA4 / `gfx1201`, 32 GB each), serving models with
**vLLM** in docker (ROCm build). vLLM is where johnny started — but the *mission* is
broader than vLLM (see "what johnny is becoming").

**What `johnny` is today.** A 466-line bash script that assumes exactly one swappable
GPU LLM on port 8000 plus a CPU embeddings service on 8001. It grep-parses a directory
of bash "launcher" scripts and probes the vLLM `/v1/models` endpoint. Subcommands:
list / load / stop / embed / `alive` (hands off to a chat TUI).

**The walls we hit.** (1) We now run *two* models at once — an orchestrator on GPUs 0,1
and a coder on GPUs 2,3 — plus embeddings; the old `start-vllm.sh` force-stops every
sibling, fatal to co-existence. (2) There's no notion of profiles, per-GPU placement, or
knowing a model's right settings without hand-tuning. (3) It's vLLM-only, single-box, and
not shareable.

**What johnny is becoming.** A **local-inference manager, not a vLLM manager.** The
people it's for are developers running their own *cobbled-together* stacks across
*disparate hardware and heterogeneous OSes* — which is exactly the vLLM / LM Studio /
Ollama / Apple-MLX world, not a datacenter. So backends are **pluggable**: the semantics
are shared (load, unload, list, observe, place), only the *mechanics and the available
knobs* vary per backend. **vLLM is the first driver; LM Studio is the second; Ollama
later.** (Practical scoping: vLLM is Linux + datacenter-class GPU — **we do not support
vLLM under WSL**. Windows/Mac developers get the LM Studio or Ollama backends; if you
want vLLM, use Linux.)

**The name.** He stays **johnny** — after Johnny 5 from *Short Circuit* ("Number 5 is
alive!"; today's `alive` command already plays on it). But johnny isn't a vLLM manager —
he's a **local inference environment manager**.

**Why a rewrite, not a patch.** Pluggable backends, multi-model placement, named
profiles, automated tuning, observability, discovery, multi-machine, and a real TUI —
all *shareable* — is a different shape of program. We keep the proven *ideas*
(state-by-probe, the chat-TUI handoff) and the proven *scripts* (tuning/benchmark
harnesses), and rebuild the spine.

**Glossary.**
- **Backend / driver** — an inference engine johnny drives (vLLM, LM Studio, Ollama)
  behind a common interface. A driver declares its *capabilities*; johnny adapts.
- **Seat** — one running model instance: its backend, container/process, port, and the
  GPU(s) it occupies.
- **Provider** — the served endpoint a client (e.g. a chat TUI) points at.
- **TP (tensor-parallel size)** — how many GPUs one model is split across (a vLLM knob).
- **Placement / config** — a specific, validated way to run a model on a backend: the
  backend-specific knobs, with measured speed + (optional) quality, keyed to hardware
  *and* runtime version.
- **Profile** — a named fleet of seats, e.g. `dev` = orchestrator + coder + embeddings.
- **Quant (FP8 / AWQ-INT4 / BF16 / FP4)** — weight precision; affects speed/memory/
  quality. Native acceleration is hardware-specific (RDNA4: FP8 yes, FP4 no;
  Blackwell: both) — a fact to *detect*, not assume.
- **MTP** — multi-token-prediction speculative decoding; can ~2× single-stream speed.
- **KV cache / KV-dtype** — per-request attention memory; its dtype is a context lever.
- **TTFT** — time-to-first-token; a key latency metric for observability.
- **Induction** — johnny's automated process that finds a model's optimal placement by
  tuning (and optionally benchmarking) it on the actual hardware.
- **Gated model** — a HF model requiring license acceptance + an auth token (e.g. Gemma,
  Llama).
- **Hermes** — the chat TUI johnny launches `alive` against today; one provider client.
- **Reaper** — idle-TTL eviction: `down` a seat after N idle minutes so the cards reach
  deep idle (~16–18 W vs ~95 W with a live HIP context). Stateless + cron-able from P3.
- **SAINT** — the external request router (formerly *goorouter*; PyPI `saint-router`,
  CLI `saint`, virtual models `saint-auto`/`saint-explain`/`saint-<backend>`): classifies each
  request into `domain × complexity` and a per-urgency **policy grid** picks a
  TOML-declared backend (static baseline, optionally johnny-bound — §3.13); LiteLLM
  dispatches. A johnny *client* over the §3.13 contract —
  the data plane to johnny's control plane — never a johnny component. johnny supplies
  the liveness/on-demand-loading it lacks; SAINT keeps owning its policy.
- **Lease / pin** — a seat marked exempt from the reaper: declaratively via profile
  `pinned: true`, or ephemerally via `johnny pin <seat> [--ttl]`. **Storage note:**
  docker container labels are *immutable after create*, so ephemeral pins can't ride the
  label mechanism — they live in the same small SQLite the reaper already consults
  (alongside activity history; cache/advisory state, never placement authority).

---

## 2. Goals & principles

1. **Local-inference manager, not a vLLM manager — pluggable backends.** A backend-driver
   interface abstracts vLLM / LM Studio / Ollama: shared semantics, per-backend knobs and
   capabilities. **vLLM first, LM Studio next, Ollama later.** Baked in as a seam from the
   start, even while only vLLM is implemented.
2. **Portable / multi-vendor / multi-OS.** No hardwired paths, IPs, or GPU assumptions.
   Detect GPU vendor (AMD ROCm / NVIDIA CUDA), count, per-GPU VRAM (or unified memory),
   arch, and natively-accelerated dtypes at runtime. `pipx`-installable. Tested targets:
   the 4× AMD RDNA4 box, a colleague's **NVIDIA DGX Spark cluster** (Grace-Blackwell,
   unified memory, FP4-native, ARM64), and **RTX 40xx (Ada)** in our test loop. **No
   vLLM-under-WSL**; Windows is served by the LM Studio/Ollama backends.
3. **Declarative registry is the source of truth** — models + validated, *backend-specific*
   placement configs + profiles. Bash launchers are *imported* into it, then retired.
4. **Multi-seat by default.** Run, observe, and tear down co-existing models safely; never
   the "stop everything" behavior that breaks co-existence.
5. **The tool figures out the parameters.** Induction auto-tunes optimal configs; users
   may override. **Tuning is the default; quality benchmarking is opt-in** (`--bench`).
6. **Observability is first-class.** A normalized telemetry model — token rates, TTFT,
   context size/utilization, concurrency, queue depth, KV/GPU utilization — per
   **provider / model / seat / node**, surfaced cheaply early and richly in the TUI.
7. **Reuse, don't reimplement.** Orchestrate the existing mlops scripts (bench, readiness,
   context-maximizer, model audit, dtype/MTP probes, eval harnesses) as subprocesses.
   Docker stays the vLLM runtime.
8. **Foundation first, TUI last.** Data model + backend drivers + engine + induction +
   telemetry usable from the CLI before the (fiddly) Textual interface.
9. **Process model: fire-and-forget by default, persistent only where needed.** The
   control plane is stateless and exits; **docker/the backend — not johnny — keeps models
   alive**. Continuous behaviors (auto-evict, JIT, live metrics) live in an *optional soft
   daemon*. All front-ends are thin clients over one shared engine library (§3.11).
10. **Leverage your own boxes — minimal multi-machine, not a datacenter scheduler.** A few
    machines you own, coordinated with explicit placement + locality. Deliberately dumb;
    "fancy cobbled," not k8s (§3.12).
11. **Control plane ≠ data plane.** johnny owns seat lifecycle, placement, induction, and
    telemetry; *request routing* (classification, policy, fallbacks) is a peer concern —
    **SAINT** first — integrated over a stable JSON contract (§3.13). SAINT's gap
    is **liveness**: its policy grid names backends but assumes they're up. johnny fills
    exactly that (ready-state + on-demand load), supplying `resolve --json` / `status
    --json` while stateless and a johnnyd HTTP API once the daemon exists. johnny's core stays **LiteLLM-free**;
    SAINT keeps running standalone against static backends.

---

## 3. Architecture

**At a glance.** Thin front-ends over one shared engine; the engine reads/writes the
registry and drives pluggable backends, which run the actual model *seats* on GPUs/CPU.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │ FRONT-ENDS   johnny CLI · status --watch (P3) · Textual TUI (P9) · │
 │ (thin)       alive → chat TUI · on-demand router (P10)             │
 └──────────────────────────────┬───────────────────────────────────┘
                                │  every front-end calls the same engine library
 ┌──────────────────────────────▼───────────────────────────────────┐
 │ ENGINE CORE   hardware-detect · registry · placement · launch ·   │
 │               profiles · induction · telemetry · discovery        │
 └─────────┬───────────────────────────────────────────┬────────────┘
           │ reads / writes                             │ drives via
 ┌─────────▼───────────────┐               ┌────────────▼────────────┐
 │ DATA  (XDG dirs)        │               │ BACKEND DRIVERS         │
 │  config.yaml            │               │  one interface: caps ·  │
 │  registry.yaml          │               │  launch · stop · state ·│
 │  profiles.yaml          │               │  metrics · logs · probe │
 │  telemetry · runs/      │               │   vLLM (docker)   #1    │
 │                         │               │   LM Studio (lms) #2·P7 │
 │                         │               │   Ollama (api)    later │
 └─────────────────────────┘               └────────────┬────────────┘
                                                        │ runs
                                           ┌────────────▼────────────┐
                                           │ SEATS on GPUs / CPU      │
                                           │ orchestrator·coder·embed │
                                           └─────────────────────────┘

 The optional soft daemon (johnnyd) hosts ONLY the always-on pieces — the
 router, the telemetry poller, the cluster agent/controller. Everything else
 is fire-and-forget CLI over the same engine; docker/the backend keeps seats
 alive regardless of johnny.

 SAINT is a front-end too — an EXTERNAL one: a client of the §3.13
 contract sitting in the request path. It resolves each backend's liveness +
 endpoint through johnny and loads cold seats on demand; it is never linked
 into johnny, and still runs standalone against static backends.
```

### 3.1 Package / distribution / onboarding
Standalone pip-installable package, installed via **pipx** (`johnny` on PATH). Code +
bundled reference scripts live inside the package; the user's **data lives outside it**:
- `$XDG_CONFIG_HOME/johnny/` → `config.yaml`, `registry.yaml`, `profiles.yaml`
- `$XDG_STATE_HOME/johnny/` → `induct/<model-id>/state.json`, `runs/<run-id>/...`, telemetry cache

`config.yaml` makes every root config-driven with env override + autodiscovery (models
dir, backend stores, vllm cache, results dir, docker images by detected vendor, bind/
advertise address, port base/reserved/range, enabled backends). Each bundled script can
be pointed at a user's own copy.

**Schema versioning from day 0 (shareable-tool make-or-break #2).** Every YAML and
state file johnny owns carries a `schema_version`; forward migrations ship with the
package (`johnny migrate`, auto-run with a timestamped backup on version bump). Without
this, v0.2 strands the first adopters. (**PyPI naming, checked & decided:** `johnny` is
taken (a dependency tracker) and `johnny5` is squatted; the distribution is
**`johnny-fleet`** — pairing with `saint-router` — with `johnny-llm`/`johnnyctl` also
free as fallbacks. CLI command and import path stay `johnny`.)

**Onboarding (shareable-tool make-or-break).**
- `johnny doctor` — preflight checks with guided fixes: docker present? GPU runtime/driver?
  correct image arch (ARM64 for DGX Spark!)? disk space for large pulls? which backend CLIs
  are installed (`lms`, `ollama`)? Reports what works and what's missing.
- `johnny init` — detect hardware, choose available backend(s), write a starter config,
  pull the right image. First-run from zero to a working `status`.
- **Security default: localhost.** Seats bind `127.0.0.1` by default — *not* `0.0.0.0` —
  because a seat is an unauthenticated OpenAI endpoint. LAN exposure is an explicit opt-in
  (with a warning); cross-machine access goes through the agent transport (§3.12), not by
  flinging open ports.

Module map (`src/johnny/`): `cli.py`, `config.py`, `hardware/{detect,dtypes}.py`,
`backends/{base,vllm,lmstudio,ollama}.py`, `registry/{schema,store,importer}.py`,
`runtime/{state,probe,lock}.py`, `engine/{placement,launch,profiles}.py`,
`induct/{pipeline,stages,grid,report}.py`, `telemetry/{schema,collect,sources}.py`,
`discover/{search,fit,auth}.py`, `external/tui.py`, `cluster/{controller,agent,transport}.py`,
`tui/` (deferred), `scripts/` (bundled).

### 3.2 Hardware abstraction (portability keystone)
`hardware.detect() -> Hardware`: vendor, GPU count, **per-GPU** VRAM, arch/gfx, and
**natively-accelerated dtypes** (ISA probe in-container, no GPU needed; RDNA4 → fp8 yes /
fp4 no, Blackwell → both). On **unified-memory** parts (DGX Spark) there is no discrete
VRAM pool — fit math treats CPU+GPU as one coherent budget. The abstraction must not
assume discrete per-card VRAM or a fixed dtype set.
- **Heterogeneous GPUs in one box** (the literal "cobbled" rig — a 4090 + a 3090): reason
  **per-GPU**, never per-box-uniform. Viability is computed per candidate GPU *group*, so a
  model can be pinned to the two matching cards while another uses the odd one.
- **Validation key = hardware fingerprint × backend × runtime/image version.** A config
  tuned on vLLM 0.20.2 isn't trusted on 0.22, nor across backends, nor on different
  hardware. Stale on any axis → `unvalidated_here`, re-induction offered. (The fingerprint
  is per-GPU-group; a homogeneous box is the special case.)

### 3.3 Backend drivers (the pluggability seam)
A **backend driver** abstracts one inference engine behind a common interface. The
semantics are shared; the mechanics and the knobs vary — so a driver *declares its
capabilities* and the engine/UI degrade gracefully.

Shared interface (every driver implements):
- `capabilities()` — what this backend supports: tunable knobs? per-GPU placement?
  metrics? logs? structured output? JIT/TTL native? This drives how induction, placement,
  and the UI behave.
- `list_local()` / `acquire(model)` — models in the backend's store; fetch one.
- `launch(spec)` / `stop(seat)` — bring a model up/down.
- `runtime_state()` — what's serving, where, on which devices/ports.
- `metrics(seat)` / `logs(seat)` — normalized telemetry (§3.7) + log stream.
- `probe_model(model)` — model-level capabilities (multimodal, tool-use, context, MTP).

Per-backend variables:
- **vLLM (docker)** — the rich case. Explicit `docker run`; full flag surface (TP, mml,
  gmu, MTP, kv-dtype, parsers); per-GPU pinning; Prometheus `/metrics`. Induction has many
  knobs to sweep. *Driver #1.* Linux only.
- **LM Studio (`lms`/llmster)** — its own model store + OpenAI API; knobs = GPU-offload,
  context, a few load params; **JIT + idle-TTL native**; Mac/Windows/Linux. Fewer tuning
  knobs → induction collapses to fit + context. *Driver #2.*
- **Ollama** — its own store + API; minimal knobs (`num_gpu`, context, `keep_alive`);
  broadest reach. *Later.*
- **(llama.cpp direct / MLX)** — possible later.

Not every operation maps 1:1 — a driver advertises what it can do, the engine never
assumes. The registry stores **backend-specific config blocks** (§3.4). This seam is built
in from P2/P3; only the vLLM driver exists in early phases.

**Seam validation early (cheap insurance).** Abstractions designed against one
implementation are reliably wrong. A **read-only LM Studio driver spike** — just
`capabilities()` / `list_local()` / `runtime_state()`, no launch — lands at **P2–P3**
to pressure-test the interface before induction and telemetry calcify vLLM-shaped
assumptions. Costs about a day; the full driver remains P7.

**Knob normalization — normalize, but not too much.** johnny exposes a *small common
vocabulary* of the knobs a user actually reaches for — context length, GPU placement /
offload, quant choice, concurrency, MTP on/off — and each driver maps those to its real
flags (vLLM `--max-model-len` / `--tensor-parallel-size` / …; LM Studio context + GPU
offload; Ollama `num_ctx` / `num_gpu`). The esoterica are deliberately **not** hoisted into
the common layer: each placement also carries a **raw passthrough block** of backend-native
flags for power users, and going hard at tuning is induction's job (§3.6) or a DIY path
(edit the config/launcher yourself). Keeping the common set small is what lets the UI and
induction stay generic instead of drowning in backend-specific detail.

### 3.4 Registry schema (YAML)
Per model: **identity** (repo id, local path, vendor, arch, params, quant),
**capabilities** (multimodal, MTP head, tool/reasoning parser, thinking toggle, native
context, chat template), and a list of validated **placements**.

Each placement is tagged with its **`backend`** and carries a knob block — a small
normalized common-knob set plus a raw passthrough of backend-native flags (§3.3).
For vLLM: `(gpu_count/TP, quant, max_model_len, gpu_memory_util, max_num_seqs,
max_num_batched_tokens, mtp{enabled,n}, kv_cache_dtype, extra_flags, env)`. For LM Studio/
Ollama: their smaller knob sets. Plus, for every placement: `use_case` (throughput |
latency | context), measured `perf` (peak & single-stream tok/s, MTP acceptance, TTFT),
`resource` (VRAM/card, KV pool), optional `quality` (only if `--bench` was run),
**`validation_key`** = `{hardware_fingerprint, backend, runtime_version}`, `validated_at`,
`source` (imported | induction | manual). Top level holds **profiles** (named fleets with
seat→role/port assignment, plus per-seat **`idle_ttl`** and **`pinned`** for the reaper)
and seen fingerprints. CPU embeddings = a model with a `cpu`
placement so the engine treats everything uniformly.

An importer seeds the registry from the existing bash launchers (image, model path, served
name, flags, env, GPU pin → gpu_count, port → role, pooling → embeddings), cross-references
the model-audit script (quant/multimodal/MTP), and pulls measured perf/quality from the
existing tuning/benchmark reports — all stamped as vLLM placements. This subsumes the
launchers and retires the start/stop helpers.

### 3.5 Placement & launch engine
- **State is derived, no state file.** Backends stamp their seats (vLLM: `johnny.*` docker
  labels; others: their native listing); live state is rebuilt from the driver's
  `runtime_state()` + endpoint probes. GPU occupancy = union across seats.
- **Occupancy from labels, never readiness — `loading` is a first-class state.** A 27B
  FP8 load is *minutes* of weight-load + graph capture before `/v1/models` answers; the
  mutation lock guards the placement *decision*, not that window. So occupancy derives
  from **docker labels (present at container start)**, and a seat whose label exists but
  whose endpoint isn't ready is `loading` — it owns its GPUs/ports for placement, shows
  distinctly in `status`, and transitions to `ready`/`failed` (failure → launch
  diagnostics, §3.7). No second `up` can double-claim mid-launch.
- **Concurrency lock (correctness).** Placement *mutations* (up/down/swap) take a lock
  (file lock on the config dir, or the daemon serializes) so two `up`s can't both claim the
  same GPUs/ports. Reads stay lock-free on derived state.
- **Placement** (per-GPU bin-packing, heterogeneous-aware): pick free GPU group(s) by
  viability (fits per-GPU VRAM × util, dtype natively accelerated), honoring reserved ports
  and the **allocation strategy** (pack vs spread, pinned/excluded GPUs, VRAM ceiling —
  borrowed from LM Studio). Emit the right `*_VISIBLE_DEVICES` for vLLM.
- **Compose is delegated to the backend driver.** The engine computes *what* and *where*;
  the driver renders *how* (vLLM: docker argv reproducing the proven launcher shape; others:
  their launch call).
- **spawn vs swap + the guard:** `spawn` adds a seat on free GPUs without touching others;
  `swap` replaces one *named* seat in place. Teardown is always by a single seat — never
  "all". The engine refuses any op that would remove a seat it didn't target; needing busy
  GPUs errors unless `--swap`/`--force` (which prints exactly which seats would die).
- **Drain is a router-layer concern.** vLLM has no drain mode — `swap`/`down` is
  SIGTERM, in-flight requests die. Acceptable single-user; when a request plane exists
  (§3.13/P10), graceful drain = stop admitting → wait `running==0` (from `/metrics`) →
  stop. The engine exposes a `--drain` flag that no-ops without a router to coordinate.

### 3.6 Model induction (centerpiece — tuning by default)
A **resumable** state machine (kill-and-resume via idempotent skip-guards) that turns a
model into a validated registry placement. **Default = tuning only** (make it run well);
quality benchmarking is opt-in (`--bench`):
1. **Discover / acquire** (§3.8) — incl. **gated-model auth** (HF token); human gate on
   disk/license.
2. **Audit** — quant, size, multimodal, MTP head → capabilities; auto-derive parser/template
   from the arch family.
3. **Hardware-fit** — enumerate viable placements for the chosen backend (TP/quant/dtype
   that fit per-GPU VRAM and are natively accelerated); prune the rest with recorded reasons.
4. **Seeded search, not brute grid** — *for backends that expose knobs* (vLLM):
   **coordinate descent / successive halving** over batch-token / seqs / memory (± MTP),
   **seeded by priors** — imported launchers, shared `unvalidated_here` placements from
   other boxes (they center and narrow the search), and arch-family heuristics
   (gpu-memory-util: highest stable; batched-tokens: knee-finding, near-monotone). The
   context-max pass runs the **KV-preflight math *before* any launch** to prune
   impossible contexts for free. Each surviving point: driver-launch → wait-ready →
   throughput bench → parse; at ~2–3 min cold-start per point, seeding + halving is a
   3–5× wall-clock win over the naive grid. For low-knob backends (LM Studio/Ollama)
   this collapses to fit + context. Resumable per point.
5. **Quality benchmark — opt-in (`--bench`).** HumanEval / ARC / GSM8K / MMLU / long-context
   needle, **"thinking-off" plumbed** (else reasoning models score 0). Heavy datasets +
   harness are pulled *only* when asked. Inherits the campaign's robustness (timeouts,
   per-bench score files, resumable, summary).
6. **Synthesize + write** — winner per use-case (throughput = max peak; latency = max
   single-stream/TTFT; context = largest coherent context); if `--bench` ran, gate on a
   quality floor. Write the placement(s) (with the full validation key) + emit report
   artifacts.

Automatic: viability pruning, the sweep, winner selection, registry write. Human-gated
(default-on, skippable): acquisition/license, parser/template for an unrecognized arch,
overwriting an existing validated placement. **Reaper-safe:** induction pins its working
seat for the duration of the run (releasing on completion or resume-abort), so an
aggressive idle TTL can't reap a sweep seat between points.

### 3.7 Observability & telemetry (first-class)
"Observability is key and will become key." A **normalized telemetry model** every driver
populates, surfaced per **provider / model / seat / node**:
- **Throughput** — generation tok/s, prompt tok/s.
- **Latency** — TTFT, inter-token latency, end-to-end.
- **Context** — context length in use, KV-cache utilization, max context.
- **Concurrency** — running vs waiting requests, queue depth, admission.
- **Resource** — per-GPU util / VRAM / power / temp, host RAM.
- **Counts** — requests, tokens in/out, errors, evictions.

Sources by backend: vLLM **Prometheus `/metrics`** (rich — running/waiting, TTFT, KV usage,
throughput) is the gold mine and free; LM Studio + Ollama expose their own, normalized into
one schema so the TUI can break everything down by provider/model and compare side by side.
**Every metric carries a `source` tag — `engine | proxy | derived | unavailable`** — because
the comparison is honestly apples-to-fruit-shaped-objects: LM Studio/Ollama expose nothing
near Prometheus histograms, and the UI must degrade visibly rather than silently show
blanks as zeros.

**Two ingest paths, source-tagged.** johnny's telemetry arrives two ways and the `source`
tag keeps them honest: **engine-pull** — johnny polls vLLM's `/metrics` directly (rich,
free, but vLLM-only); and **proxy-push** — the request router *provides* normalized
per-request latency that johnny **accepts**. The proxy-push path exists because pulling
doesn't generalize: LM Studio and Ollama expose nothing near Prometheus, so the proxy
(which sees every request to every backend identically) is the *only* uniform cross-backend
tap for TTFT/tok-s. **johnny owns the ingest schema** — the provider maps its internal data
to johnny's normalized format, not the reverse — so any future router conforms to one
contract.

**Ingest is daemon-optional: a durable spool.** Because johnny's core is stateless until
P10, the provider appends one record per request to a johnny-owned **append-only spool**
(`$XDG_STATE_HOME/johnny/ingest/`); johnny's reaper/poller ingests it on its tick.
Concurrency discipline: writers append single-line records (`O_APPEND` keeps small
appends atomic on local filesystems); the ingester **rotates the file before reading**
(rename, then consume) so a mid-append record is never split and nothing is
double-ingested. This is
*better* than push-to-daemon for intermittent use — if johnny isn't running, telemetry
accumulates and is ingested later instead of dropped. HTTP POST to johnnyd is an optional
low-latency path once the daemon exists, never a requirement.

**Persistence starts at P3, small.** A single-table SQLite (ring-buffered/downsampled),
written by the poller from both ingest paths. Not optional once the reaper exists —
**idle detection requires last-activity history** (request counters + timestamps); it also
feeds trends, capacity decisions, and P10 admission. Long-horizon rollups land P8.
(Resolves Q4.)

Plus:
- **Logs** — `johnny logs <seat> [-f]` (the old `logs.sh`, generalized per seat).
- **Launch diagnostics** — on a failed bring-up, tail the log and recognize known failure
  signatures (KV-preflight OOM, NCCL hang, port clash, image/arch mismatch), reusing the
  induction KV-ceiling parser.
- **Cheap early win** — `johnny status --watch` (a plain Rich live table) lands at P3, long
  before the full Textual TUI, to scratch the monitoring itch.

Define the telemetry schema + the driver `metrics()` contract early (P3), with the SQLite
history alongside; rich dashboards come with the TUI (P9). This data feeds the **reaper
(P3)** and auto-evict policy + admission (P10).

### 3.8 Model discovery & acquisition
Help the user *choose*, not just download. **Scope:**
- **v1** — `johnny search <query>` against Hugging Face: candidates with quant variants,
  params, context, capability badges (tool-use, vision), and a **fit verdict for the
  detected hardware** (fits / tight / won't-fit + limiting factor, via the §3.6 fit
  estimator) so unrunnable models are flagged *before* download. `johnny download`/`acquire`
  honors **gated-model auth** (`johnny login` / `HF_TOKEN` — you hit this for Gemma/Llama).
  Acquisition lands weights where the target backend expects them (vLLM: models_dir; others:
  via the driver into their store).
- **Later** — backend-native catalogs (Ollama library, LM Studio catalog), `model.yaml`
  alignment, recommend-by-task.

### 3.9 External chat-TUI routing
Generalize `alive`: pick the seat by **role** (default chat/orchestrator) with `--model`/
`--seat` overrides; resolve port + served name from live state (so chat targets the
orchestrator even while the coder is also up). A small adapter keeps "launch a TUI against a
seat" provider-agnostic; the built-in adapter reproduces today's Hermes/tmux behavior.
`provider sync` patches *only* the relevant block of the chat tool's config (endpoint +
model catalog from the registry), never a blind rewrite.

### 3.10 CLI surface (foundation-first, pre-TUI)
`status [--watch] [--json]`, `up <model> [--backend|--placement|--port|--swap|--force|--wait]`,
`up --profile <p>`, `down <model|seat|--profile> [--drain]`, `swap <seat> <model>`,
`reap [--idle-ttl <dur>] [--dry-run]`, `pin|unpin <seat> [--ttl]`, `resolve <role|seat>`,
`induct <repo|path> [--backend|--use-case|--bench|--resume|--yes]`, `tune <model>`,
`bench <model|seat>`, `search <query>`, `download <repo>`, `login`, `logs <seat> [-f]`,
`metrics <seat>`, `registry show|edit|import|validate`, `cleanup [--dry-run]`, `migrate`,
`doctor`, `init`, `alive [--model|--seat|--role]`, `provider sync`, `gpu`, `nodes` (cluster).
`--json` on everything — **`status --json`, `resolve --json`, and `up --wait --json` *are*
the v0 request-plane contract (§3.13)**, with `resolve` the per-dispatch hot-path call.
Back-compat shims: `load` → swap the primary LLM seat; `stop` → down
it; `embed start|stop|restart|status`; `alive` (role-aware).

### 3.11 Process model & the (soft) daemon
johnny is **both** a fire-and-forget CLI and — when continuous behavior is needed — a
persistent service, split so we don't pay daemon complexity until a feature demands it:
- **Stateless CLI core (P0–P9 work without it).** Every command derives runtime truth from
  the backends and exits. No state file. The fleet is owned by docker/the backend, so johnny
  restarting/crashing never drops a seat. **The reaper is the proof case:** `johnny reap`
  is a stateless one-shot (reads activity history *and ephemeral pins* from the telemetry
  SQLite — container labels are immutable post-create, so pins can't be labels — downs
  idle-past-TTL unpinned seats) driven by cron/systemd-timer — no daemon required. When
  johnnyd exists, it hosts the same loop.
- **Optional soft daemon (`johnnyd`).** Introduced with the first feature that *requires* a
  clock or request stream (multi-machine agent, or the auto-evict/JIT router). Owns *policy,
  not containers* — reconstructs from the backends on start; if it dies you lose auto-evict/
  JIT/cluster-coordination until restart but **never a model**.
- **One engine, many front-ends.** CLI, daemon, TUI, router, agent are thin clients over the
  same `engine/`+`registry/`+`backends/`+`telemetry/` library. The TUI runs standalone or
  attaches to the daemon for a live stream; it never *requires* the daemon.

### 3.12 Multi-machine (minimal — leverage your own boxes)
You have a few machines with real GPUs and want to use them together; today that's painful.
Keep this **deliberately minimal — explicit, not a scheduler.** "A few daemons that take
orders," not k8s.

```
   johnny CLI / TUI
         │
         ▼
   ┌─────────────────┐   agents dial OUT to the controller
   │ CONTROLLER      │   (join token + TLS; mDNS optional discovery)
   │ (johnnyd)       │◄──────────────┬──────────────┬──────────────┐
   │ registry +      │               │              │              │
   │ placement       │               │              │              │
   └─────────────────┘        ┌──────┴─────┐  ┌──────┴─────┐  ┌─────┴──────┐
                              │ agent boxA │  │ agent boxB │  │ agent boxD │
                              │ vLLM       │  │ LM Studio  │  │ vLLM       │
                              │ 4×RDNA4    │  │ RTX 40xx   │  │ DGX Spark  │
                              │ seats…     │  │ seats…     │  │ seats…     │
                              └────────────┘  └────────────┘  └────────────┘
   docker/backend = each node's source of truth → seats survive controller loss
```

- **Roles.** A **controller** (the main daemon) holds the cluster registry; a lightweight
  **agent** (`johnnyd --agent`) runs on each box and owns local backends. CLI/TUI are
  controller clients. Single-box is the degenerate case (controller + agent collapse, or
  none — the stateless CLI).
- **Enrollment & transport (pluggable).** The agent **dials out** to the controller
  (NAT-friendly) with a join token over TLS, presenting a stable **node ID** (hostname +
  hardware fingerprint). **Steal the k3s trick:** controller init generates a private CA;
  the join token embeds the CA cert hash, the agent pins it — no cert-management ceremony,
  no TOFU ambiguity. Default wire: a dependency-light direct connection (**HTTP+websocket
  preferred over gRPC** — gRPC wheels on ARM64/odd Pythons are avoidable pain). **mDNS**
  is optional zero-config discovery on a single subnet (feeding, never replacing,
  enrollment), with a static address fallback. **MQTT** is an optional adapter for sites
  that already run a broker — never required infra.
- **Inventory.** The agent reports the `Hardware` struct, software context (docker/backend
  versions, available images, drivers, OS/arch), **which model weights are local**
  (locality), live seat state + free GPUs + telemetry, and a TTL heartbeat.
- **Placement = explicit + locality (v1 ambition is low on purpose).** You say "run this on
  `that-box`," or johnny picks by fingerprint match + model locality + free GPUs. **No
  reconciler, no affinity/anti-affinity engine** in v1. Induction is dispatched to a node
  whose fingerprint matches the target; results are keyed by fingerprint and shared across
  matching nodes.
- **Survival.** Docker/the backend stays each node's source of truth: if an agent or the
  controller dies, that node's seats keep serving; on reconnect the agent re-syncs and the
  controller reconciles without double-launching. Any node stays drivable by its local CLI.

### 3.13 Request plane: SAINT integration (control plane ≠ data plane)
**SAINT** (reviewed from source, `feat/initial-router`) is a working localhost
OpenAI-compatible proxy: a request's latest user message is classified into
`domain ∈ {code, general} × complexity ∈ {trivial, medium, hard}`, then a per-urgency
**policy grid** (`(urgency, domain, complexity) → backend`) picks a backend, with
`!`-prefix and `saint-<backend>` model-pin overrides, a `saint-explain` dry-run, SQLite
request logging (already versioned + migrated + relabel-for-training), and LiteLLM as the
dispatch SDK. Today **backends are static TOML** with **no liveness concept**: a policy
cell names `local-coder` and SAINT *assumes it's serving*.

**SAINT predates johnny and is in-scope to evolve** (its own repo/OpenSpec — see the
companion `add-johnny-integration` change). So this is **co-design, not adaptation** —
but the boundary still holds for independent reasons: the integration is **CLI shell-out
(v0) or johnnyd HTTP (v1), never a library import**, so johnny stays **LiteLLM-free** and
SAINT keeps running **standalone against static backends** when johnny is absent. No
shared dependency, two processes, one contract.

**SAINT's evolution (its side, summarized):** the **static backend config stays the
default and the baseline** — every backend keeps its `base_url`+`model` and works exactly
as today with johnny absent. A backend *optionally* gains a **johnny binding**
(`johnny_role`/`johnny_seat`); when `[johnny]` is enabled, reachable, and the seat
resolves `ready`, johnny's live endpoint/model/liveness **overrides** the static baseline.
When johnny is disabled, unreachable, or the seat isn't ready, the backend **falls back to
its own static config** — not to some other backend. johnny is a pure override overlay on
a config that already stands on its own. The policy grid's *vocabulary is unchanged* —
johnny does **not** generate it; the grid stays human-owned, which is SAINT's design
intent. What changes is that a johnny-bound cell can resolve to a live endpoint **and** a
readiness state at dispatch, with the static endpoint as the floor under it.

**The contract johnny exposes (versioned JSON; v0 = CLI, v1 = johnnyd HTTP):**
1. **`resolve <role|seat>` — the purpose-built hot-path primitive.** Read-only, fast,
   cacheable (~1 s TTL): `→ {seat, endpoint, model, state: ready|loading|absent|failed,
   eta_s, queue_depth}` (`eta_s` is best-effort, derived from recorded historical
   cold-start durations for that model/placement in the telemetry SQLite — absent on a
   first-ever load). This is the single call SAINT needs per dispatch — a focused
   projection of `status`, not the whole fleet dump. Exists as `johnny resolve --json`
   from the moment placement does (P3).
2. **Ensure-loaded — non-blocking.** `johnny up <role> --wait --json` → ready endpoint,
   or immediate `{state: loading, eta_s}`. Cold start is *minutes* here, so the rule is
   **the router never blocks a request on a load**: on a `loading`/`absent` resolve, fire
   ensure-loaded (idempotent) and serve *this* request via SAINT's **`while_loading`**
   target (below); subsequent requests land on the warmed seat.
3. **Leases / pins.** SAINT pins a seat it's leaning on (`johnny pin <seat> --ttl`);
   the reaper honors pins. Common case needs nothing — routed traffic *is* activity, so
   activity-TTL keeps hot seats warm; pins only cover the load-in-progress gap.
4. **Telemetry — SAINT *provides*, johnny *accepts*.** SAINT already measures
   per-request latency, tokens, and success; it maps each dispatched request to **johnny's
   normalized ingest schema** (johnny owns the contract) and provides it — appending to the
   durable spool (§3.7) by default, or POSTing to johnnyd when present. This is the uniform
   cross-backend latency johnny can't pull from LM Studio/Ollama. SAINT keeps its own
   request log for its own purposes (relabel-for-training, routing analytics, `log show`);
   the push is *derived* from the same data, best-effort and non-fatal (same discipline as
   its existing log write). One addition on SAINT's side: real **TTFT** needs a
   first-chunk timestamp in its streaming path (today it measures total stream time).

**SAINT's `while_loading` and the static floor.** On a non-ready seat, the served
fallback order honors "static is the default" at every step: **`while_loading`** target
(per-backend override, else global) → else the backend's **own static baseline** (try it;
it may be independently live in a mixed setup) → else `default_on_failure`. `while_loading`
is distinct from `default_on_failure` because on this hardware a warming seat is a *normal
expected* state, not a failure; folding them would make logs and `explain` lie and force
the last-resort target (cloud) to double as the warm-up target. And **johnny-unreachable
degrades to the static baseline directly** — if the static endpoint is still serving, the
request just works; johnny being down doesn't break routing. This is a small, clean change
built on SAINT's existing fallback machinery.

**Emergent property worth naming:** because a johnny-bound backend resolves its endpoint
*from johnny*, and johnny owns placement across boxes (§3.12), SAINT transparently
gains **multi-machine routing** — a backend can resolve to a seat on another node with no
SAINT awareness. Caveat: the resolved endpoint must be reachable from SAINT's host,
which interacts with johnny's **localhost-default** posture — cross-box seats need explicit
LAN exposure / the agent transport, not open ports.

**Classifier-seat caveat:** if SAINT's *classifier* backend is itself johnny-managed,
it runs on ~every auto request, so a reap → cold-start would tax the first post-idle
request on the classifier. Mitigation falls out of existing parts: **pin the classifier
seat** (or keep the classifier on cheap static/cloud — Haiku is fast), and treat
"classifier seat loading" as a classify-failure that takes SAINT's *already-existing*
classifier-fallback path. No new mechanism.

**The synergy, corrected.** johnny does **not** auto-generate SAINT's policy (fixed,
human-owned vocabulary). What flows is sharper: (a) **liveness + on-demand load** turns
every policy cell from "hope it's up" into "route to it, warming it on demand" — johnny
supplies exactly the layer SAINT lacks; (b) **cloud fallback covers the reaper** —
because SAINT already spills to cloud under pressure, johnny reaping a local seat is
*safe* (a cold seat just routes to cloud or `while_loading` until warm), which lets you run
a **more aggressive idle TTL** than a johnny-only setup would dare — more of that ~$100/mo
back; (c) `johnny suggest-policy` can print induction `--bench` winners as a *suggested*
grid edit, advisory only. In return, SAINT's request log tells johnny which seats a
profile should keep warm (demand-driven profiles, §7).

---

## 4. Phasing (foundation + backends + engine + induction + telemetry first; TUI/router/cluster later)

- **P0 — Skeleton, config, onboarding.** Package, entry point, config/roots discovery,
  `--json`, **`doctor` + `init`**, localhost-default, **`schema_version` in every owned
  file + `migrate` scaffold**. *Verify:* `pipx install`; `johnny doctor` reports the box
  honestly; `johnny status` reproduces today's view; a version-bumped file migrates with
  backup. (Check the PyPI name before first publish.)
- **P1 — Hardware abstraction.** Vendor/count/per-GPU-VRAM-or-unified/arch/native dtypes;
  heterogeneous-aware. *Verify:* reports the real GPUs; fp8 native, fp4 not (RDNA4).
- **P2 — Backend-driver interface + vLLM driver + registry (backend-aware) + importer.**
  Includes the **read-only LM Studio spike** (§3.3) to pressure-test the seam.
  *Verify:* registry round-trips the real vLLM coder/orchestrator/gemma placements with
  validation keys; `driver.runtime_state()` matches docker; `validate` passes; the spike
  driver lists/states an LM Studio install (or skips cleanly when absent).
- **P3 — Placement & launch engine + telemetry foundation + reaper.** Driver-delegated
  compose, locking, heterogeneous per-GPU placement, multi-seat, **label-derived occupancy
  with `loading` as a first-class state**; **telemetry schema (source-tagged) + vLLM
  `/metrics` engine-pull + the normalized proxy-push ingest spool + SQLite history +
  `logs` + launch diagnostics + `status --watch`**; **the
  reaper** (`johnny reap`, cron-able) with profile `idle_ttl`/`pinned`. *Verify end-to-end:*
  one command brings up orchestrator + coder + embeddings; a second `up` during a
  multi-minute load cannot double-claim the loading seat's GPUs; targeted `down` removes
  only the named seat; the guard refuses sibling-killing ops; `status --watch` shows live
  token rate / TTFT / queue depth per seat; `logs` tails a seat; **an idle seat past TTL
  is reaped, a pinned seat survives, and the freed cards reach deep idle (~16–18 W)**;
  **a record appended to the ingest spool is picked up and attributed `source=proxy`**;
  **`resolve` returns `ready` with the live endpoint for a serving seat and `loading` +
  best-effort `eta_s` mid-launch** (the v0 SAINT contract is live from here).
  Retire start/stop helpers *and the standalone idle-watchdog plan — the reaper absorbs it*.
- **P4 — Induction (tuning default; `--bench` opt-in).** **Seeded search** (§3.6), priors
  from imported/shared configs. *Verify:* inducting a known model reproduces its hand-tuned
  vLLM placement (tuning only, fast) **in materially less wall-clock than a naive grid**;
  `--bench` records quality; kill-and-resume continues; an over-VRAM option is pruned with
  a reason; KV-preflight math prunes impossible contexts without a launch.
- **P5 — Model discovery & acquisition.** HF `search` with fit verdicts + capability badges;
  gated-model `login`/download. *Verify:* search flags an unrunnable model as won't-fit; a
  gated model (Gemma) downloads with a token.
- **P6 — Chat-TUI routing + provider sync.** *Verify:* with the fleet up, `alive` lands on
  the orchestrator seat; `provider sync` edits only the intended config block.
- **P7 — LM Studio backend driver (second backend).** *Verify:* johnny lists/launches/
  observes an LM Studio model *alongside* a vLLM seat, with normalized telemetry from both;
  capability negotiation makes induction collapse appropriately.
- **P8 — Lifecycle + cleanup + monitoring history.** `cleanup --dry-run` (audit gaps +
  staleness + validation-key mismatch); **long-horizon telemetry rollups/trends** (raw
  series persist since P3).
- **P9 — Textual/Rich TUI.** Full dashboards over the telemetry API: live seats, token
  rates, TTFT, context, concurrency, queue depths — **broken out by provider / model / node.**
  Presentation only; no new domain logic.
- **P10 — Request-plane API + built-in JIT gateway.** Formalize §3.13 on johnnyd:
  `GET /v1/fleet` + watch stream, **`resolve`**, ensure-loaded endpoint, leases; plus a
  **minimal built-in OpenAI-compatible gateway** (JIT-load on first request,
  admission/concurrency caps, router-coordinated `--drain`) for users *without* an
  external router — **SAINT is the reference external consumer**, now co-designed
  (its johnny-bound backends resolve liveness via `resolve` and load on demand, §3.13).
  (Prior art worth twenty minutes: **llama-swap** — OpenAI gateway that swaps models on
  demand with TTL; its config model and failure handling, not its code.) *Verify:*
  SAINT resolves a cold johnny-bound backend, johnny JIT-loads it while SAINT
  serves the first request via its `while_loading` target, and the warmed seat takes
  subsequent traffic; pinned seats survive the reaper; `down --drain` waits for
  `running==0`. **Note:** the v0 contract (`resolve --json` + `up --wait`) lands far
  earlier as a spike — it needs nothing past P3.
- **P11 — Multi-machine (minimal).** Controller + per-node agents (§3.12): enrollment +
  heartbeat, explicit/locality placement, fingerprint-keyed induction dispatch. *Verify:* a
  second box joins, reports its hardware, the controller places a seat on it; killing the
  controller leaves both nodes' seats serving and a reconnect re-syncs without double-launch.

*P7+ are partly independent tracks — order by need. (Multi-machine is "make it exist," not
make it clever.)*

---

## 5. Ideas borrowed from (and now partly *built on*) LM Studio

LM Studio started as inspiration and is now also **backend driver #2 (P7)**. Mechanics worth
taking, and where they land:

| LM Studio feature | What it does | How we adapt it | Lands in |
|---|---|---|---|
| **Idle TTL + auto-evict** | Unloads idle models; "evict before load" caps residents | Per-seat idle TTL → auto-`down`; eviction policy when a new seat needs busy cards | **P3 (reaper) / P10 (policy)** |
| **JIT / on-demand loading** | First request loads the model | A gateway that spawns the right seat on first request; with TTL = self-managing pool | **P10** |
| **Hardware-aware fit (traffic-light)** | Fits-or-not *before* load (size × quant × ctx vs VRAM) | A fast fit verdict in `search`/`up`/pre-induction — fits / tight / won't-fit + limiting factor | **P1/P3/P5** |
| **`lms` CLI: `ls` vs `ps`, `--json`, `--host`** | On-disk vs in-memory; scriptable; remote | `ls` (registry/on-disk) vs `ps` (seats), `--json` everywhere, `--host`/agents for remote | **P0/P3/P11** |
| **Multi-GPU allocation policy** | Enable/disable GPUs; pack vs spread; VRAM ceiling; tensor-split | Named placement **strategies** rather than raw flags | **P3** |
| **Per-model defaults honored everywhere** | Load params persist across GUI + CLI | Registry placements are the single source every entry point honors | **P2** |
| **`model.yaml` portable model+variants spec** | Open-standard model + variants + default params | Align the registry's identity/variants layer where sensible (cross-backend!) | **P2** (Q6) |
| **Structured output + tool-use first-class** | `response_format: json_schema`; tool-trained badges | Surface capability metadata per model; vLLM already does guided decoding | **P2/P5** |
| **Per-model concurrency / admission control** | Max concurrent predictions; queue the rest | Optional per-seat concurrency cap / admission queue | **P10** |
| **Catalog: quant variants + recommended quant + badges** | One model, many quants, hardware pick | The discovery view (§3.8) + TUI catalog | **P5/P9** |
| **LM Studio itself as a backend** | Native JIT/TTL, Mac/Windows reach, OpenAI API | **Driver #2** — johnny manages LM Studio seats too | **P7** |

Deliberately **not** borrowing (better in vLLM / out of scope): the chat UI,
RAG/chat-with-documents, split-view, the spec-decode *picker* (keep only the vocab-match
validation idea), preset Hub sharing.

---

## 6. Open questions (for review)

*Resolved — review round 1 (scope):* johnny is a pluggable **local inference environment manager**
(vLLM first, LM Studio next, Ollama later); induction defaults to **tuning-only**
(`--bench` opt-in); knobs are **lightly normalized** — a small common user-facing set + raw
passthrough, deep tuning left to induction/DIY (§3.3); multi-machine stays **minimal /
explicit** (§3.12); NVIDIA is a first-class tested target, no vLLM-under-WSL; the **name
stays johnny**.

*Resolved — review round 2 (architecture):*
1. **Profiles location → separate `profiles.yaml`.** The registry is machine-written
   (induction); profiles are human intent. Different write authority → different files.
2. **Cross-box config sharing → both.** Shared configs land `unvalidated_here` *and*
   serve as **priors that seed/narrow the induction search** (§3.6). Re-induction stays
   the only trusted path, but it gets cheap.
3. **Quality floor → relative + catastrophic guard.** Within ~2–3% of *that model's*
   best observed per benchmark (absolute thresholds don't transfer across sizes), plus a
   hard absolute floor to catch kv-dtype/quant corruption (near-zero GSM8K etc.).
4. **Telemetry storage → persist from P3.** Small SQLite ring at the poller; the reaper's
   idle detection requires activity history anyway (§3.7). Rollups at P8.
5. **`model.yaml` → native schema + import/export.** It covers identity/variants but not
   placements or validation keys — johnny's whole point. Compatibility shim, not adoption.
   (Worth a quick check on the spec's traction outside LM Studio before investing.)
6. **Discovery depth → HF-only for v1.** Ollama's library has no real API; LM Studio's
   catalog is mostly HF-backed anyway.
7. **Multimodal & embeddings induction →** embeddings get a tiny grid (batch × seq —
   cheap, do it); vision models get fit verdict + smoke test only in v1.
8. **Transport & locking → dial-out HTTP+websocket; `flock` now, daemon serializes
   later.** k3s-style CA-hash-in-join-token (§3.12). MQTT stays an optional adapter.

*Resolved — review round 3 (telemetry direction):* **SAINT provides, johnny accepts** — johnny
owns the normalized ingest schema; SAINT pushes proxy-measured latency to a durable
append-only **spool** (stateless-compatible, survives johnny being down), with
HTTP-to-johnnyd as an optional low-latency path. Engine-pull (vLLM `/metrics`) stays the
second source; `source` tags keep them distinct (§3.7). SAINT keeps its own request log.

Still open (the §3.13 contract, co-designed with SAINT):
1. **Spool format & ingest cadence.** JSONL append (simplest, durable) vs a small WAL'd
   SQLite SAINT inserts into (johnny shares the reader). And does johnny ingest on the
   reaper tick, a dedicated `johnny ingest` cron, or only when the daemon runs? (Leaning:
   JSONL spool + ingest on any johnny telemetry tick; SQLite-shared-writer is more coupling
   for little gain.)
2. **`ensure_load` authority.** Should a SAINT `[johnny] ensure_load = false` mode
   exist — observe liveness + fall back, but never *trigger* loads (human/johnny owns
   load decisions)? (Leaning: yes; cheap safety valve, some users want it.)
3. **`while_loading` granularity & the static floor.** Global default + per-backend
   override (chosen), with the backend's **own static baseline** as the implicit floor
   beneath it (johnny-unreachable or not-ready → try the static endpoint before
   `default_on_failure`). `johnny_only` backends opt out of the static floor. Confirm no
   real case needs per-policy-cell warm-up targets.
4. **`resolve` vs `up --wait` overlap.** `resolve` is read-only/cacheable (hot path);
   `up --wait` is the imperative load. Keep them distinct (chosen) — but should `resolve`
   optionally take `--ensure` to fuse the two for callers that want one round-trip?
5. **Built-in gateway scope.** With SAINT as the real external router, does johnny's
   built-in P10 gateway shrink to a demo, or stay a genuine minimal product for
   non-SAINT users? (Leaning: stay minimal-but-real — not everyone runs SAINT.)
6. **Lease model.** Explicit pins only, or router-held leases with heartbeat (auto-expire
   if SAINT dies mid-load)? Pins are simpler; leases survive a router crash.
7. **Contract version location & deprecation** once SAINT depends on it (shared
   `johnny --contract-version`? a pinned `CONTRACT.md`?).

---

## 7. Future horizons

Beyond the foundation, designed *toward* so we don't paint ourselves into a corner:

- **More backends — Ollama, llama.cpp/MLX.** Ollama (driver, broad reach incl. Windows/Mac)
  and possibly llama.cpp/MLX direct. The driver interface (§3.3) is built for exactly this;
  these are "next backends," not a rearchitecture.
- **Richer cluster scheduling — only if it earns its keep.** v1 is explicit placement +
  locality (§3.12). Auto bin-packing with affinity/anti-affinity is a *deliberate non-goal*
  unless real pain demands it (honoring "not datacenter management").
- **Demand-driven profiles.** SAINT's request log (it already records backend +
  latency + tokens per request in SQLite) telling johnny which seats a profile should
  keep warm — placement shaped by measured demand instead of guesswork. The §3.13
  telemetry read is the data; this is the policy.
- **`johnny suggest-policy`.** Induction `--bench` scores printed as a *suggested* edit
  to SAINT's policy grid — which local seat wins code/chat/long-context on this
  hardware — leaving the grid human-owned (SAINT's design intent) but evidence-fed.
- **LM Link-style secure remote access.** Reach a johnny controller / a seat from off-LAN
  with proper auth (mTLS / identity), beyond the trusted-LAN agent transport. The enrollment
  design (§3.12) is the first step.
- **Full cross-OS / cloud reach.** The hardest portability (Windows-native paths, cloud GPU
  rentals as just another node). The abstractions (hardware, backends, registry, telemetry,
  agents) are deliberately not vLLM- or Linux-locked, so this stays open.

---

*This document is the working design for review. It will keep evolving here; the
implementation targets the local box (docker + GPUs + the existing mlops scripts) first,
then the LM Studio backend and the colleague's NVIDIA cluster.*
