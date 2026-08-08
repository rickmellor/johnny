"""johnny CLI (P0 surface).

Implemented now: status, doctor, init, migrate, version. The broader surface from
PLAN §3.10 (up/down/induct/reap/resolve/...) is stubbed with honest "lands at Pn"
messages so a mistyped command is friendly rather than cryptic.

Design notes:
- Every command supports `--json` for scripting (the foundation of the v0
  request-plane contract once `resolve`/`up --wait` land at P3).
- Bare `johnny` runs `status` (reproducing the old bash tool's default view).
- The control plane is fire-and-forget: these commands derive truth from docker +
  endpoints and exit. No daemon, no state file (§3.11).
"""

from __future__ import annotations

import json as _json
import sys as _sys
import time
import zlib

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from . import config as C
from . import doctor as _doctor
from . import migrate as _migrate
from .runtime import probe

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="johnny — a shareable local inference environment manager.",
)
console = Console()
err = Console(stderr=True)

_STATE_STYLE = {"ready": "green", "running": "yellow", "loading": "yellow", "down": "red"}
_CHECK_STYLE = {"ok": "green", "warn": "yellow", "fail": "red"}
_STATUS_STYLE = {"validated": "green", "unmeasured": "yellow", "incomplete": "red",
                 "stale": "magenta", "unverified": "yellow"}

# --help is grouped into task-family panels (rich_help_panel). Panels render in the
# order their first command is defined below, so the section order here mirrors the
# order commands appear in this file: Seats → Observe → Models → Fleet → Setup.
_P_SEATS = "Seats — serve & lifecycle"
_P_OBSERVE = "Observe — status, logs, metrics"
_P_MODELS = "Models & tuning"
_P_FLEET = "Fleet & integrations"
_P_SETUP = "Setup & maintenance"


# --------------------------------------------------------------------------- status
def _seat_image(s) -> str:
    return (s.extra or {}).get("image") or "—"


def _fmt_context(ctx: int) -> str:
    """Format context window: 2048 → '2K', 32768 → '32K', 1048576 → '1M'."""
    if ctx >= 1_048_576:
        return f"{ctx // 1_048_576}M"
    elif ctx >= 1024:
        return f"{ctx // 1024}K"
    return str(ctx)


def _seat_spec() -> dict:
    """model_id -> compact spec string (params · quant · context) from the registry."""
    from .registry import store

    spec_cache = {}
    for mid, m in store.models(store.load()).items():
        ident = m.get("identity") or {}
        ctx = (m.get("capabilities") or {}).get("native_context")
        parts = [str(v) for v in (ident.get("params"), ident.get("quant")) if v]
        if ctx:
            parts.append(_fmt_context(ctx))
        spec_cache[mid] = " · ".join(parts) if parts else "—"
    return spec_cache


def _seats_as_dicts(seats) -> list[dict]:
    from .registry import store

    models = store.models(store.load())
    out = []
    for s in seats:
        m = models.get(s.model) or {}
        ident = m.get("identity") or {}
        out.append(
            {"seat": s.name, "backend": s.backend, "port": s.port, "model": s.model,
             "state": s.state, "gpus": s.gpus, "image": _seat_image(s),
             # raw registry fields, null when unknown — formatting is the table's job
             "params": ident.get("params"), "quant": ident.get("quant"),
             "native_context": (m.get("capabilities") or {}).get("native_context")})
    return out


def _render_status(json_output: bool = False) -> None:
    from . import engine

    seats = engine.all_seats()
    if json_output:
        console.print(_json.dumps({"seats": _seats_as_dicts(seats)}, indent=2))
        return
    if not seats:
        if not probe.docker_available():
            err.print("[red]docker is not reachable[/] — is the daemon running? (`johnny doctor`)")
        else:
            console.print("[dim]no inference seats running.[/] Start one with [bold]johnny up <model>[/].")
        return
    spec_map = _seat_spec()
    table = Table(title="johnny — seats", title_style="bold")
    for col, style in (("SEAT", "bold"), ("BACKEND", "dim"), ("PORT", None),
                       ("MODEL", "cyan"), ("SPEC", "dim"), ("STATE", None), ("GPUS", None), ("IMAGE", "dim")):
        table.add_column(col, style=style, no_wrap=(col == "SPEC"))
    for s in seats:
        gpus = ",".join(map(str, s.gpus)) if s.gpus else "—"
        table.add_row(s.name, s.backend, str(s.port or "—"), s.model or "—",
                      spec_map.get(s.model, "—"),
                      f"[{_STATE_STYLE.get(s.state, 'white')}]{s.state}[/]", gpus, _seat_image(s))
    console.print(table)


@app.command(rich_help_panel=_P_OBSERVE)
def status(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
    watch: bool = typer.Option(False, "--watch", help="Refresh live (basic; full TUI is P9)."),
) -> None:
    """Show running inference seats (docker + endpoint probe)."""
    if watch and not json_output:
        try:
            from rich.live import Live

            # Load the spec map once (registry rarely changes while watching).
            spec_map = _seat_spec()
            with Live(console=console, refresh_per_second=4, screen=True) as live:
                while True:
                    table = _build_status_renderable(spec_map)
                    live.update(table)
                    time.sleep(2)
        except KeyboardInterrupt:
            return
    else:
        _render_status(json_output=json_output)


def _build_status_renderable(spec_map=None):
    """A Rich renderable of current seats (used by --watch).

    `spec_map` is loaded once before the loop to avoid re-parsing the registry
    YAML on every refresh tick (the registry rarely changes while watching)."""
    from . import engine

    seats = engine.all_seats()
    if spec_map is None:
        spec_map = _seat_spec()
    table = Table(title="johnny — seats (live)", title_style="bold")
    for col, style in (("SEAT", "bold"), ("BACKEND", "dim"), ("PORT", None),
                       ("MODEL", "cyan"), ("SPEC", "dim"), ("STATE", None), ("GPUS", None), ("IMAGE", "dim")):
        table.add_column(col, style=style, no_wrap=(col == "SPEC"))
    for s in seats:
        gpus = ",".join(map(str, s.gpus)) if s.gpus else "—"
        table.add_row(s.name, s.backend, str(s.port or "—"), s.model or "—",
                      spec_map.get(s.model, "—"),
                      f"[{_STATE_STYLE.get(s.state, 'white')}]{s.state}[/]", gpus, _seat_image(s))
    return table


# --------------------------------------------------------------------------- doctor
@app.command(rich_help_panel=_P_SETUP)
def doctor(json_output: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
    """Preflight checks: docker, GPU runtime, arch, disk, backends, config."""
    checks = _doctor.run_checks()
    if json_output:
        console.print(_json.dumps(checks, indent=2))
        return
    table = Table(title="johnny doctor", title_style="bold")
    table.add_column("CHECK", style="bold")
    table.add_column("STATUS")
    table.add_column("DETAIL")
    for c in checks:
        s = c["status"]
        mark = {"ok": "✓", "warn": "!", "fail": "✗"}.get(s, "?")
        table.add_row(c["name"], f"[{_CHECK_STYLE.get(s, 'white')}]{mark} {s}[/]", c["detail"])
    console.print(table)
    if any(c["status"] == "fail" for c in checks):
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- init
@app.command(rich_help_panel=_P_SETUP)
def init(
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config."),
    pull: bool = typer.Option(False, "--pull", help="Also `docker pull` the vLLM image (large)."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Detect the box, write a starter config (+ registry/profiles stubs)."""
    paths = C.get_paths()
    if paths.config_file.exists() and not force:
        err.print(f"[yellow]config already exists:[/] {paths.config_file}  (use --force to overwrite)")
        raise typer.Exit(code=1)

    disc = C.autodiscover()
    cfg = C.build_default_config(disc)

    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.ingest_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)

    header = (
        f"# johnny config — schema v{C.CONFIG_SCHEMA_VERSION}\n"
        "# Written by `johnny init`. Edit freely; run `johnny migrate` after a tool upgrade.\n"
        "# Roots/scripts were autodiscovered on this box (only existing paths are recorded).\n"
        "# Security: seats bind to network.bind_address (default 127.0.0.1 = localhost only)."
    )
    C.write_yaml(paths.config_file, cfg, header=header)
    if not paths.registry_file.exists() or force:
        C.write_yaml(paths.registry_file, C.registry_stub(),
                     header=f"# johnny registry — schema v{C.REGISTRY_SCHEMA_VERSION} (machine-written; seeded by `registry import` at P2)")
    if not paths.profiles_file.exists() or force:
        C.write_yaml(paths.profiles_file, C.profiles_stub(),
                     header=f"# johnny profiles — schema v{C.PROFILES_SCHEMA_VERSION} (human-authored fleets)")

    pulled = None
    if pull:
        img = cfg["docker"]["vllm_image"]
        from .util import run as _run
        err.print(f"[dim]pulling {img} … (this can take a while)[/]")
        rc, _, perr = _run(["docker", "pull", img], timeout=1800)
        pulled = {"image": img, "ok": rc == 0, "error": perr.strip() if rc != 0 else None}

    summary = {
        "config": str(paths.config_file),
        "registry": str(paths.registry_file),
        "profiles": str(paths.profiles_file),
        "vendor": disc["vendor"],
        "backends_enabled": cfg["backends"]["enabled"],
        "scripts_found": sorted(disc["scripts"].keys()),
        "pulled": pulled,
    }
    if json_output:
        console.print(_json.dumps(summary, indent=2))
        return
    console.print(f"[green]✓ wrote[/] {paths.config_file}")
    console.print(f"  registry: {paths.registry_file}")
    console.print(f"  profiles: {paths.profiles_file}")
    console.print(f"  detected GPU vendor: [bold]{disc['vendor'] or 'none'}[/]")
    console.print(f"  backends enabled: [bold]{', '.join(cfg['backends']['enabled']) or 'none'}[/]")
    if disc["scripts"]:
        console.print(f"  reusable scripts found: [dim]{', '.join(sorted(disc['scripts']))}[/]")
    if pulled:
        console.print(f"  image pull: {'[green]ok[/]' if pulled['ok'] else '[red]failed[/]'} ({pulled['image']})")
    console.print("\nNext: [bold]johnny doctor[/] then [bold]johnny status[/].")


# --------------------------------------------------------------------------- migrate
@app.command(rich_help_panel=_P_SETUP)
def migrate(
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would change; touch nothing."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Migrate owned files to the current schema (timestamped backups)."""
    paths = C.get_paths()
    results = _migrate.migrate_all(paths, dry_run=dry_run)
    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    if json_output:
        console.print(_json.dumps(public, indent=2))
        return
    table = Table(title="johnny migrate" + (" (dry-run)" if dry_run else ""), title_style="bold")
    table.add_column("FILE", style="bold")
    table.add_column("FROM")
    table.add_column("TO")
    table.add_column("ACTION")
    style = {"migrated": "green", "up-to-date": "dim", "absent": "dim",
             "would-migrate": "yellow", "newer-than-tool": "red"}
    for r in public:
        if not r.get("exists"):
            table.add_row(r["kind"], "—", "—", "[dim]absent[/]")
            continue
        act = r["action"]
        table.add_row(r["kind"], str(r.get("version")), str(r.get("target")),
                      f"[{style.get(act, 'white')}]{act}[/]")
    console.print(table)
    if any(r.get("action") == "newer-than-tool" for r in public):
        err.print("[red]a file is newer than this johnny[/] — upgrade johnny rather than downgrading the file.")
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- version
@app.command(rich_help_panel=_P_SETUP)
def version(json_output: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
    """Print johnny + schema versions."""
    info = {
        "johnny": __version__,
        "schema": {
            "config": C.CONFIG_SCHEMA_VERSION,
            "registry": C.REGISTRY_SCHEMA_VERSION,
            "profiles": C.PROFILES_SCHEMA_VERSION,
        },
    }
    if json_output:
        console.print(_json.dumps(info, indent=2))
    else:
        console.print(f"johnny [bold]{__version__}[/]  "
                      f"[dim]schema: config v{info['schema']['config']} · "
                      f"registry v{info['schema']['registry']} · profiles v{info['schema']['profiles']}[/]")


# --------------------------------------------------------------------------- gpu
def _dimm_summary(dimms) -> str:
    """Group identical DIMMs: '8× 32 GB DDR4-2667' (or 'a + b' when mixed)."""
    from collections import Counter

    groups = Counter((d.size_gb, d.type, d.configured_mts or d.speed_mts) for d in dimms)
    out = []
    for (size, typ, spd), n in sorted(groups.items(), key=lambda x: -x[1]):
        label = f"{size:g} GB"
        if typ and spd:
            label += f" {typ}-{spd}"
        elif typ:
            label += f" {typ}"
        out.append(f"{n}× {label}")
    return " + ".join(out)


def _fmt_link(mbps) -> str:
    """NIC link speed as an Ethernet class (10 GbE / 2.5 GbE / 1 GbE), or raw Mb/s."""
    if not mbps or mbps <= 0:
        return "[dim]—[/]"
    if mbps >= 1000 and mbps % 1000 == 0:
        return f"{mbps // 1000} GbE"
    if mbps >= 1000:
        return f"{mbps / 1000:g} GbE"
    return f"{mbps} Mb/s"


def _hinfo_json(hw, host, specdb) -> dict:
    from dataclasses import asdict

    from .hardware import detect as hwdetect
    from .hardware import specs as specmod

    gpus = []
    for g in hw.gpus:
        gd = asdict(g)
        gd["theoretical_tflops"] = hwdetect.theoretical_tflops(g)
        gd["ai_spec"] = specmod.spec_for(specdb, g.arch, g.cu_count)
        gpus.append(gd)
    gpu_block = {k: v for k, v in asdict(hw).items() if k != "gpus"}
    gpu_block["gpus"] = gpus
    return {
        "gpu": gpu_block,
        "cpu": asdict(host.cpu),
        "memory": asdict(host.mem),
        "storage": [asdict(d) for d in host.disks],
        "network": [asdict(n) for n in host.nics],
    }


def _ai_spec_line(spec: dict, count: int) -> str:
    """One-line AI-matrix spec from the curated DB, with sparsity, box total, provenance."""
    i8, f16 = spec.get("int8_matrix_tops"), spec.get("fp16_matrix_tflops")
    i8s, f16s = spec.get("int8_matrix_tops_sparse"), spec.get("fp16_matrix_tflops_sparse")
    approx = "~" if spec.get("approx") else ""
    parts = []
    if i8:
        parts.append(f"{approx}{i8:g} TOPS INT8")
    if f16:
        parts.append(f"{approx}{f16:g} TFLOPS FP16")
    body = " · ".join(parts) or "—"
    spar = f" [dim](sparse {i8s:g}/{f16s:g})[/]" if (i8s or f16s) else ""
    box = f"  [dim]· box ×{count} ≈ {i8 * count:g} TOPS INT8[/]" if (i8 and count > 1) else ""
    src = spec.get("source", "")
    host = src.split("/")[2] if "//" in src else src
    prov = f"  [dim]\\[{host} · {spec.get('as_of', '?')}][/]" if host else ""
    return f"  [bold]AI matrix (per card):[/] {body}{spar}{box}{prov}"


def _render_hinfo(hw, host, specdb) -> None:
    from .hardware import detect as hwdetect
    from .hardware import specs as specmod

    # ---- CPU ----
    c = host.cpu
    console.print("[bold underline]CPU[/]")
    console.print(f"  {c.model}")
    if c.max_mhz and c.base_mhz:
        freq = f"{c.base_mhz / 1000:.2f}–{c.max_mhz / 1000:.2f} GHz"
    elif c.max_mhz:
        freq = f"up to {c.max_mhz / 1000:.2f} GHz"
    elif c.base_mhz:
        freq = f"{c.base_mhz / 1000:.2f} GHz"
    else:
        freq = "—"
    sock = f"{c.sockets}× socket · " if c.sockets > 1 else ""
    l3 = f" · L3 {c.l3_mb:.0f} MB" if c.l3_mb else ""
    console.print(f"  {sock}{c.cores} cores / {c.threads} threads · {freq}{l3}")
    bogo = f"{c.bogomips_total:,.0f}" if c.bogomips_total else "—"
    console.print(f"  BogoMIPS {bogo} [dim](Linux MIPS proxy, not a benchmark)[/] · "
                  f"AI ISA: [green]{', '.join(c.ai_flags) or '—'}[/]")

    # ---- Memory ----
    m = host.mem
    console.print("\n[bold underline]Memory[/]")
    if m.dimms:
        slots = f"{m.populated}/{m.slots} slots" if m.slots else f"{m.populated} DIMM(s)"
        maxc = f" · max {m.max_capacity_gb:.0f} GB" if m.max_capacity_gb else ""
        ecc = " · ECC" if (m.ecc and m.ecc.lower() not in ("none", "")) else " · no ECC"
        tag = f"  [dim](cached {m.captured_at})[/]" if m.cached else ""
        console.print(f"  {m.total_gb:.0f} GB · {_dimm_summary(m.dimms)} · {slots}{maxc}{ecc}{tag}")
        parts = sorted({d.part_number for d in m.dimms if d.part_number})
        if parts:
            console.print(f"  [dim]{', '.join(parts)}[/]")
    else:
        console.print(f"  {m.total_gb:.0f} GB [dim](DIMM detail needs root — "
                      f"run `johnny hinfo --seed-memory` once to cache it)[/]")

    # ---- GPU ----
    console.print("\n[bold underline]GPU[/]")
    if not hw.gpus:
        console.print(f"  [yellow]none detected[/] (vendor={hw.vendor or 'none'}) — CPU / LM Studio / Ollama only.")
    else:
        het = "" if hw.homogeneous else "  [yellow](heterogeneous)[/]"
        console.print(f"  [bold]{hw.vendor}[/] · {len(hw.gpus)} GPU(s) · {hw.total_vram_gb:.0f} GB VRAM · "
                      f"fingerprint [cyan]{hw.fingerprint}[/]{het}")
        cu_by_arch = {g.arch: g.cu_count for g in hw.gpus}
        for grp in hw.groups:
            dl = ", ".join(grp.native_dtypes) or "—"
            console.print(f"  [bold]{grp.arch}[/] ×{grp.count} @ {grp.vram_gb:.0f}GB — native dtypes: "
                          f"[green]{dl}[/] [dim](source: {hw.dtype_source})[/]")
            spec = specmod.spec_for(specdb, grp.arch, cu_by_arch.get(grp.arch))
            if spec:
                console.print(_ai_spec_line(spec, grp.count))
            else:
                console.print(f"  [dim]AI matrix: — (no cached spec for {grp.arch}; add it to gpu_specs.json)[/]")
        t = Table(pad_edge=False)
        for col in ("IDX", "NAME", "ARCH", "VRAM", "CU", "CLK", "FP32*", "FP16*"):
            t.add_column(col, style="cyan" if col == "ARCH" else None)
        tot32 = tot16 = 0.0
        for g in hw.gpus:
            th = hwdetect.theoretical_tflops(g) or {}
            f32, f16 = th.get("fp32_tflops"), th.get("fp16_tflops")
            tot32 += f32 or 0
            tot16 += f16 or 0
            t.add_row(str(g.index), g.name, g.arch, f"{g.vram_gb:.0f} GB",
                      str(g.cu_count or "—"), f"{g.clk_mhz:.0f} MHz" if g.clk_mhz else "—",
                      f"{f32:.1f}" if f32 else "—", f"{f16:.1f}" if f16 else "—")
        console.print(t)
        if tot32:
            console.print(f"  [dim]* theoretical vector TFLOPS (CU×clock×2, no dual-issue) · box ≈ "
                          f"{tot32:.0f}/{tot16:.0f} FP32/FP16. Matrix/tensor AI-TOPS is a spec value — not shown.[/]")

    # ---- Storage ----
    console.print("\n[bold underline]Storage[/]")
    if not host.disks:
        console.print("  [dim]—[/]")
    else:
        t = Table(pad_edge=False)
        for col in ("DEVICE", "SIZE", "KIND", "MODEL", "THROUGHPUT*"):
            t.add_column(col)
        for d in host.disks:
            size = f"{d.size_gb / 1024:.1f} TB" if d.size_gb >= 1024 else f"{d.size_gb:.0f} GB"
            t.add_row(d.name, size, d.kind, d.model or "—", d.throughput_est or "—")
        console.print(t)
        console.print("  [dim]* sequential throughput, bus-class estimate — not measured[/]")

    # ---- Network ----
    console.print("\n[bold underline]Network[/]")
    if not host.nics:
        console.print("  [dim]—[/]")
    else:
        for n in host.nics:
            st = "green" if n.state == "up" else "red"
            console.print(f"  [bold]{n.name}[/]  {_fmt_link(n.speed_mbps)}  "
                          f"[{st}]{n.state}[/]  [dim]{n.mac or ''}[/]")


@app.command(rich_help_panel=_P_SETUP)
def hinfo(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
    refresh: bool = typer.Option(False, "--refresh", help="Re-run the GPU dtype ISA probe (ignore cache)."),
    refresh_specs: bool = typer.Option(False, "--refresh-specs", help="Re-pull the curated GPU AI-TOPS spec DB into the cache."),
    seed_memory: bool = typer.Option(False, "--seed-memory", help="Cache DIMM detail via `sudo dmidecode` (prompts for a password) so later unprivileged runs show it."),
) -> None:
    """Host hardware inventory: GPUs, CPU, memory, storage, network — with derived metrics
    (theoretical GPU TFLOPS, matrix AI-TOPS, BogoMIPS, link throughput).

    AI-TOPS are manufacturer spec-sheet figures from a curated DB — pulled into a per-user
    cache on first run (`--refresh-specs` re-pulls), shown with source + date, and '—' for
    archs with no cached spec. Theoretical vector TFLOPS are computed (CU×clock) and labelled
    as such. DIMM detail needs root: run `--seed-memory` once to cache it for later runs.
    Nothing here is fabricated — unknown specs stay '—'.
    """
    from .hardware import detect as hwdetect
    from .hardware import hostinfo as hostmod
    from .hardware import specs as specmod

    state_dir = C.get_paths().state_dir
    if seed_memory:
        res = hostmod.seed_memory(state_dir)
        if res and res.dimms:
            console.print(f"[green]✓ cached memory info[/] — {res.populated} DIMM(s) "
                          f"({_dimm_summary(res.dimms)}). Future `johnny hinfo` runs show it.\n")
        else:
            err.print("[yellow]couldn't read DIMM detail[/] — dmidecode/sudo failed or was declined.\n")

    hw = hwdetect.detect(refresh=refresh)
    host = hostmod.host_info(state_dir)
    specdb = specmod.load_specs(state_dir, refresh=refresh_specs)
    if json_output:
        # Plain print, not console.print: Rich soft-wraps long lines (e.g. the spec source
        # URL), inserting newlines that corrupt the JSON when piped.
        print(_json.dumps(_hinfo_json(hw, host, specdb), indent=2))
        return
    _render_hinfo(hw, host, specdb)


@app.command(name="gpu", hidden=True)
def _gpu_alias(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
    refresh: bool = typer.Option(False, "--refresh", help="Re-run the GPU dtype ISA probe (ignore cache)."),
) -> None:
    """Deprecated alias for `hinfo` (kept for muscle memory)."""
    hinfo(json_output=json_output, refresh=refresh, refresh_specs=False)


# --------------------------------------------------------------------------- registry
registry_app = typer.Typer(add_completion=False, help="Inspect / seed / validate / normalize the model registry.")
app.add_typer(registry_app, name="registry", rich_help_panel=_P_MODELS)


def _print_quant_mix(ident: dict, backends: str) -> None:
    """Below the arch/quant summary line, a per-tensor quant breakdown — only when
    a GGUF model's weights are on disk AND more than one type is significant (a
    plain single-quant GGUF already says everything in `quant=`). This is the same
    ground truth registry normalize / induct derive identity.quant from, just shown
    in full instead of collapsed to the compact label."""
    lp = ident.get("local_path")
    if "llamacpp" not in backends or not lp or not str(lp).endswith(".gguf"):
        return
    md = (C.load_yaml(C.get_paths().config_file) or {}).get("roots", {}).get("models_dir")
    if not md:
        return
    from pathlib import Path

    p = Path(md).expanduser() / lp
    if not p.exists():
        return
    from .backends.llamacpp import gguf_shard_paths, quant_mix

    mix = quant_mix(gguf_shard_paths(p))
    total_elems = sum(d["elems"] for d in mix.values())
    total_bytes = sum(d["bytes"] for d in mix.values())
    if len(mix) <= 1 or not total_elems:
        return
    rows = sorted(mix.items(), key=lambda kv: -kv[1]["elems"])
    parts = [f"{t} {100 * d['elems'] / total_elems:.1f}%·{d['bytes'] * 8 / d['elems']:.2f}bpw"
             for t, d in rows]
    console.print(f"  [dim]quant mix: {' · '.join(parts)}  "
                  f"(overall {total_bytes * 8 / total_elems:.2f} bpw)[/]")


def _ident_params(ident: dict) -> str:
    """identity.params for tables — prefers the derived exact total (params_total,
    e.g. '1.03T') over the stored label, which for MoE headers is expert shorthand
    ('384x14B' is a ~1T model); '—' when neither is set. The detail view shows both."""
    total = (ident or {}).get("params_total")
    if total:
        active = (ident or {}).get("params_active")
        return f"{total}-A{active}" if active else str(total)
    p = (ident or {}).get("params")
    if not p:
        return "[dim]—[/]"
    return _fmt_b(p) if isinstance(p, (int, float)) else str(p)


def _registry_compact_table(models: dict) -> Table:
    """Terse one-row-per-model index (shared by `registry show -c` and `up`'s picker)."""
    t = Table(title=f"registry — {len(models)} model(s)", title_style="bold")
    for col in ("MODEL", "ARCH", "PARAMS", "QUANT", "CTX", "#PL", "BACKENDS", "PATH"):
        t.add_column(col, style="dim" if col == "PATH" else None)
    for mid, m in sorted(models.items(), key=lambda kv: kv[0].lower()):
        ident = m.get("identity", {})
        pls = m.get("placements", [])
        backends = sorted({p.get("backend", "?") for p in pls})
        path = ident.get("local_path") or ident.get("repo_id") or "—"
        t.add_row(mid, str(ident.get("arch") or "—"), _ident_params(ident),
                  str(ident.get("quant") or "—"),
                  str(m.get("capabilities", {}).get("native_context") or "—"), str(len(pls)),
                  ", ".join(backends), path)
    return t


def _backup_registry_file():
    """Timestamped copy of registry.yaml before a hand-rolled mutation outside the normal
    induction/import write paths (house convention — same pattern as `registry normalize
    --apply`). Returns the backup Path, or None if there's no registry file yet to copy."""
    import shutil
    from datetime import datetime

    p = C.get_paths().registry_file
    if not p.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = p.with_name(f"{p.name}.bak-{ts}")
    shutil.copy2(p, backup)
    return backup


def _resolve_model(models: dict, query: str) -> str:
    """Exact model id, else the unique case-insensitive substring match — same resolution
    style as `registry delete`. Exits (candidates listed) on zero or multiple hits."""
    if query in models:
        return query
    hits = [mid for mid in models if query.lower() in mid.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        err.print(f"[red]no model[/] matching '{query}' in the registry.")
        raise typer.Exit(code=1)
    err.print(f"[red]'{query}' is ambiguous[/] — matches {len(hits)}:")
    for mid in hits:
        console.print(f"  • {mid}")
    raise typer.Exit(code=1)


def _current_runtimes() -> dict:
    """Backend -> current launch image, for staleness checks (loaded once per command)."""
    from .registry import normalize as N

    cfg = C.load_yaml(C.get_paths().config_file) or {}
    return N.current_runtimes(cfg)


def _running_pins() -> dict:
    """(model_id, placement_id) -> actual GPU indices, for placements running *right now*.

    A placement config has no fixed pins (they're assigned at launch from what's free), so
    we only ever show real cards — read back from the running seat's `johnny.*` labels.
    Best-effort: no docker / no seats -> {} and the view degrades to card counts."""
    try:
        from . import engine

        pins: dict = {}
        for s in engine.all_seats():
            labels = (s.extra or {}).get("labels") or {}
            model, pid = labels.get("johnny.model"), labels.get("johnny.placement")
            if model and pid and s.gpus:
                pins[(model, pid)] = list(s.gpus)
        return pins
    except Exception:
        return {}


def _fmt_toks(peak, single) -> str:
    """peak/single tok/s as 'peak/single', '—' where unmeasured."""
    if peak is None and single is None:
        return "[dim]—[/]"
    p = f"{peak:g}" if peak is not None else "—"
    s = f"{single:g}" if single is not None else "—"
    return f"{p}/{s}"


def _status_cell(status: str) -> str:
    return f"[{_STATUS_STYLE.get(status, 'white')}]{status}[/]"


# TOOL column palette — visually distinct, dark-terminal-safe; avoids green/yellow/red
# (already claimed by STATUS) and dim/white/black. zlib.crc32 (not the builtin hash(),
# which is PYTHONHASHSEED-randomized per process) gives a stable value -> color mapping
# so the same runtime tag always renders the same color across separate `johnny` runs.
_TOOL_PALETTE = ["cyan", "blue", "magenta", "bright_cyan", "bright_blue",
                "bright_magenta", "purple", "turquoise2", "orchid", "deep_sky_blue1"]


def _tool_cell(tool: str) -> str:
    if not tool or tool == "—":
        return "[dim]—[/]"
    color = _TOOL_PALETTE[zlib.crc32(tool.encode()) % len(_TOOL_PALETTE)]
    return f"[{color}]{tool}[/]"


def _emit_table(renderable, wide: bool = False) -> None:
    """Print a table fit-to-terminal (default — columns may collapse to '…'), or at full
    natural width when `wide`: nothing is truncated, and the terminal wraps the long lines
    (or pipe to `less -S` to scroll them). Uses a very wide virtual console so Rich sizes
    every column to its content instead of squeezing to fit."""
    if wide:
        Console(width=10_000).print(renderable)
    else:
        console.print(renderable)


def _print_worklist(worklist: list[dict]) -> None:
    """Placements whose fix is a real benchmark (`johnny tune`), not a normalize pass."""
    if not worklist:
        return
    console.print(f"\n[bold]needs `johnny tune`[/] — {len(worklist)} placement(s) lack trustworthy numbers:")
    for w in worklist:
        console.print(f"  {_status_cell(w['status'])}  [cyan]{w['model']}[/] / [bold]{w['placement']}[/]")


def _print_ctxsafe_worklist(worklist: list[dict]) -> None:
    """Large-context placements that have never had `ctxsafe` run — see the 2026-08-06
    incident in AGENTS.md § Context safety: a configured max_model_len alone is not
    proof a real deep prefill survives it."""
    if not worklist:
        return
    console.print(f"\n[bold]needs `johnny bench <target> --suite ctxsafe`[/] — {len(worklist)} "
                 "large-context placement(s) never had their configured max_model_len "
                 "empirically verified safe:")
    for w in worklist:
        console.print(f"  [yellow]untested[/]  [cyan]{w['model']}[/] / [bold]{w['placement']}[/] "
                      f"(max_model_len={w['max_model_len']:,})")


def _mem_cell(p: dict) -> str:
    """Weight placement: '111G+111G' = VRAM + CPU-RAM offload (yellow), '111G' =
    all-GPU. Recorded at induction for llamacpp placements; '—' when unknown."""
    mem = p.get("mem") or {}
    v, r = mem.get("vram_gb"), mem.get("ram_gb")
    if v is None:
        return "[dim]—[/]"
    if r:
        return f"{v:.0f}G+[yellow]{r:.0f}G[/]"
    return f"{v:.0f}G"


def _gpus_cell(gcount, pins) -> str:
    """'×N' card count, augmented with the live '[i,j]' pins when the placement is
    running (pins are real; idle placements have no fixed cards, so count only)."""
    base = f"×{gcount}" if gcount else "[dim]—[/]"
    if pins:
        return f"{base} [green]\\[{','.join(str(g) for g in pins)}][/]"
    return base


# --------------------------------------------------------------------------- columns
# `registry show`'s column selector. Each key maps to a (HEADER, description) pair — the
# description only surfaces in `--columns list`. _DEFAULT_COLUMNS is the terse, "what do I
# need to know at a glance" set; _ALL_COLUMNS is everything (== --wide's column set, and
# roughly the table's original fixed shape plus SOURCE + the new USE column tacked on).
_COLUMN_SPECS: dict[str, tuple[str, str]] = {
    "model":  ("MODEL", "registry model id"),
    "params": ("PARAMS", "parameter count"),
    "id":     ("ID", "placement id — for `up --placement`, `registry delete`"),
    "backend": ("BACKEND", "vllm | llamacpp"),
    "quant":  ("DTYPE", "weights quant/dtype"),
    "gpus":   ("GPUS", "GPU card count (+ live pins when the seat is running)"),
    "mem":    ("MEM", "VRAM(+CPU-RAM offload) footprint"),
    "tp":     ("TP", "tensor-parallel size"),
    "priority": ("PRIORITY", "sweep winner-pick basis — throughput/latency/context "
                             "(`induct --use-case`); NOT the same as the USE column"),
    "context": ("CONTEXT", "configured/native context window"),
    "kv":     ("KV", "KV-cache dtype"),
    "speed":  ("TOK/S", "measured peak/single-stream tok/s"),
    "status": ("STATUS", "validated | unmeasured | stale | incomplete | unverified"),
    "tool":   ("TOOL", "runtime image tag the placement was tuned on"),
    "source": ("SOURCE", "imported | induction | manual"),
    "recommended_use": ("USE", "free-text 'what's this good for' (`registry set-use`)"),
}
_DEFAULT_COLUMNS = ["model", "params", "quant", "context", "speed", "recommended_use"]
_ALL_COLUMNS = ["model", "params", "id", "backend", "quant", "gpus", "mem", "tp",
                "priority", "context", "kv", "speed", "status", "tool", "source",
                "recommended_use"]
_NO_WRAP_COLUMNS = {"model", "params", "id", "backend", "quant", "gpus", "mem", "tp",
                    "priority", "context", "kv", "speed", "status", "tool", "source",
                    "recommended_use"}
# Every column truncates with '…' instead of wrapping (uneven multi-line rows are hard to
# scan) — min_width is a floor so short columns can't be squeezed to nothing when a wide
# variable-length column is also present at a narrow terminal; max_width is a ceiling only
# on the columns whose content length actually varies a lot (model names, recommended_use),
# generous enough to rarely truncate when they're one of just a few columns shown, but
# bounded so they don't monopolize width away from everything else in the default view.
_MIN_WIDTH_COLUMNS = {"model": 12, "params": 6, "quant": 6, "context": 8, "speed": 8}
_MAX_WIDTH_COLUMNS = {"model": 36, "recommended_use": 72}
_COLUMN_ALIASES = {
    "ctx": "context", "dtype": "quant", "tok_s": "speed", "toks": "speed",
    "tokspeed": "speed", "use": "recommended_use", "rec": "recommended_use",
    "recommend": "recommended_use", "recommended": "recommended_use",
}


def _parse_columns(spec: str | None, wide: bool) -> list[str]:
    """--columns parsing. None -> the default set (or every column under --wide); 'all' ->
    every column explicitly; else a comma-separated key list (aliases resolved, unknown keys
    rejected with a pointer to `--columns list`)."""
    if not spec:
        return list(_ALL_COLUMNS if wide else _DEFAULT_COLUMNS)
    if spec.strip().lower() == "all":
        return list(_ALL_COLUMNS)
    cols, unknown = [], []
    for raw in spec.split(","):
        k = raw.strip().lower()
        if not k:
            continue
        k = _COLUMN_ALIASES.get(k, k)
        (cols if k in _COLUMN_SPECS else unknown).append(k)
    if unknown:
        raise ValueError(f"unknown column(s): {', '.join(unknown)} — "
                         f"`johnny registry show --columns list` for the full set")
    return cols


def _print_column_help() -> None:
    t = Table(title="registry show --columns", title_style="bold")
    t.add_column("KEY", style="cyan")
    t.add_column("HEADER")
    t.add_column("MEANING")
    for key in _ALL_COLUMNS:
        header, desc = _COLUMN_SPECS[key]
        t.add_row(key, header, desc)
    console.print(t)
    console.print(f"[dim]default: {','.join(_DEFAULT_COLUMNS)}[/]")
    console.print(f"[dim]all (= --wide): {','.join(_ALL_COLUMNS)}[/]")
    console.print("[dim]aliases: ctx→context, dtype→quant, tok_s/toks→speed, use/rec→recommended_use[/]")


def _context_cell(mml, native, verified_safe=None) -> str:
    """'configured/native' context window (e.g. '32K/128K'); collapses to one value when
    they agree or only one is known, '—' when neither is. When a `ctxsafe` probe has run
    (see bench.py) and found the empirically verified-safe depth BELOW the configured
    max_model_len, that's appended in red — the exact silent gap (config says one thing,
    a real deep prefill crashes the seat) that caused the 2026-08-06 Ornith incident."""
    if not mml and not native:
        return "[dim]—[/]"
    if mml and native and mml == native:
        base = _fmt_context(native)
    else:
        base = f"{_fmt_context(mml) if mml else '—'}/{_fmt_context(native) if native else '—'}"
    if verified_safe and mml and verified_safe < mml:
        base += f" [red](safe:{_fmt_context(verified_safe)})[/]"
    return base


def _column_cell(key: str, mid: str, p: dict, v: dict, identities: dict, caps: dict, pins: dict) -> str:
    ident = identities.get(mid) or {}
    if key == "model":
        return mid  # sparse-blanked by the caller for repeat rows
    if key == "params":
        return _ident_params(ident)
    if key == "id":
        return v["id"]
    if key == "backend":
        return v["backend"]
    if key == "quant":
        return v["dtype"] or ident.get("quant") or "—"
    if key == "gpus":
        return _gpus_cell(v["gpus"], pins.get((mid, v["id"])))
    if key == "mem":
        return _mem_cell(p)
    if key == "tp":
        return v["tp"] or "—"
    if key == "priority":
        return v["priority"]
    if key == "context":
        vsafe = ((p.get("quality") or {}).get("ctxsafe") or {}).get("verified_safe_tokens")
        return _context_cell(v["mml"], (caps.get(mid) or {}).get("native_context"), vsafe)
    if key == "kv":
        return v["kv"] or "—"
    if key == "speed":
        return _fmt_toks(v["peak"], v["single"])
    if key == "status":
        return _status_cell(v["status"])
    if key == "tool":
        return _tool_cell(v["tool"])
    if key == "source":
        return v["source"]
    if key == "recommended_use":
        return ident.get("recommended_use") or "[dim]—[/]"
    return "—"


def _placements_table(rows, current, *, columns: list[str] | None = None, pins: dict | None = None,
                      identities: dict | None = None, caps: dict | None = None, title: str = "") -> Table:
    """The standardized per-placement view — column set driven by `columns` (see
    _DEFAULT_COLUMNS / _ALL_COLUMNS / `registry show --columns`).

    `rows` is an iterable of (model_id, placement) so the all-models view can render one
    scannable table (sparse MODEL column) rather than a header per model. `pins` maps
    (model, placement) -> live GPU indices for running seats; `identities` maps model -> its
    identity block (dtype fallback + recommended_use); `caps` maps model -> its capabilities
    block (native_context, for the CONTEXT column). Most columns are no_wrap so they never
    collapse to '…' on a narrow terminal — MODEL/BACKEND/USE are left free to wrap instead."""
    from .registry import normalize as N

    columns = columns or _DEFAULT_COLUMNS
    pins, identities, caps = pins or {}, identities or {}, caps or {}
    t = Table(title=title or None, title_style="dim", title_justify="left", pad_edge=False)
    for key in columns:
        header, _desc = _COLUMN_SPECS.get(key, (key.upper(), ""))
        style = "bold" if key == "model" else ("cyan" if key == "id" else None)
        t.add_column(header, style=style, no_wrap=key in _NO_WRAP_COLUMNS,
                     min_width=_MIN_WIDTH_COLUMNS.get(key), max_width=_MAX_WIDTH_COLUMNS.get(key),
                     overflow="ellipsis" if key in _NO_WRAP_COLUMNS else None)

    prev = None
    for mid, p in rows:
        v = N.placement_view(p, current)
        first = mid != prev
        prev = mid
        cells = [(mid if first else "") if key == "model"
                else _column_cell(key, mid, p, v, identities, caps, pins) for key in columns]
        t.add_row(*cells)
    return t


@registry_app.command("show")
def registry_show(
    model: str = typer.Argument(None, help="Model id to detail; omit to list all."),
    compact: bool = typer.Option(False, "--compact", "-c", help="Terse one-row-per-model index (omit placements)."),
    wide: bool = typer.Option(False, "--wide", "-w", help="Render at full width (no column collapsing; pipe to `less -S` to scroll), and — unless --columns overrides it — show every column. Shorthand for `--columns all`."),
    columns: str = typer.Option(None, "--columns", help="Comma-separated columns to show, or 'all' for every column, or 'list' to print the full set with descriptions and exit. Available: "
                                + ", ".join(_ALL_COLUMNS) + f". Default: {', '.join(_DEFAULT_COLUMNS)}."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """List registry models with their placements (or detail one model).

    Placements are what you load (`up --placement <id>`) and prune
    (`registry delete <model> <id>`), so they're shown by default; -c for the index.
    By default a terse column set scales to your terminal; --columns picks exactly which
    columns to show (`--columns list` prints the full set); --wide is shorthand for every
    column at full, uncollapsed width.
    """
    from .registry import store

    if columns and columns.strip().lower() in ("list", "help", "?"):
        _print_column_help()
        return
    if compact and columns:
        err.print("[red]--columns[/] doesn't apply with --compact (a fixed one-row-per-model table) — drop one or the other.")
        raise typer.Exit(code=1)
    try:
        cols = _parse_columns(columns, wide)
    except ValueError as e:
        err.print(f"[red]{e}[/]")
        raise typer.Exit(code=1)

    # Single-model detail: the reader is inspecting THIS model's placements, so default
    # to the columns that distinguish them (id for `up --placement`, tp/gpus/status) and
    # drop the redundant MODEL column. An explicit --columns/--wide still wins.
    if model and not columns and not wide:
        cols = ["id", "backend", "tp", "gpus", "quant", "context", "speed", "status"]

    reg = store.load()
    models = store.models(reg)
    if model:
        m = models.get(model)
        if not m:
            err.print(f"[red]no model[/] '{model}' in the registry")
            stray_key = _COLUMN_ALIASES.get(model.strip().lower(), model.strip().lower())
            if stray_key in _COLUMN_SPECS:
                err.print(f"[dim]'{model}' looks like a --columns value that got split by your shell — "
                          f"a space after a comma in an unquoted --columns list breaks into separate "
                          f"arguments. Try: --columns {model.strip().lower()}... (no space) or quote the "
                          f"whole list: --columns \"...\"[/]")
            raise typer.Exit(code=1)
        if json_output:
            console.print(_json.dumps(m, indent=2))
            return
        ident = m.get("identity", {})
        pls = m.get("placements", [])
        backends = ", ".join(sorted({p.get("backend", "?") for p in pls})) or "—"
        console.print(f"[bold]{model}[/]  [dim]path: {ident.get('local_path') or ident.get('repo_id') or '—'}[/]")
        _ptotal, _pactive = ident.get("params_total"), ident.get("params_active")
        if _ptotal:
            _params = f"{_ptotal} total" + (f" · {_pactive} active/token" if _pactive else "")
            if ident.get("params"):
                _params += f" (header: {ident['params']})"
        else:
            _params = ident.get("params") or "—"
        console.print(f"  arch={ident.get('arch')} params={_params} quant={ident.get('quant')} "
                      f"ctx={m.get('capabilities',{}).get('native_context')} backend={backends}")
        if ident.get("recommended_use"):
            console.print(f"  [dim]use:[/] {ident['recommended_use']}")
        _print_quant_mix(ident, backends)
        pins = _running_pins()
        _emit_table(_placements_table([(model, p) for p in pls], _current_runtimes(), columns=cols,
                                      pins=pins, identities={model: ident},
                                      caps={model: m.get("capabilities") or {}}), wide)
        if any((model, p.get("id")) in pins for p in pls):
            console.print("[dim]\\[i,j] = live GPU pins (running now)[/]")
        return

    if json_output:
        console.print(_json.dumps(reg, indent=2))
        return
    if not models:
        console.print("[dim]registry is empty.[/] Seed it with [bold]johnny registry import[/].")
        return
    if compact:
        _emit_table(_registry_compact_table(models), wide)
        md = (C.load_yaml(C.get_paths().config_file) or {}).get("roots", {}).get("models_dir")
        if md:
            from pathlib import Path

            console.print(f"[dim]models dir: {Path(md).expanduser()}  (PATH column is relative to this)[/]")
        return

    # Default: one scannable table across every model's placements (sparse MODEL column).
    console.print(f"[bold]registry — {len(models)} model(s)[/]  "
                  f"[dim]load: `johnny up <model> --placement <id>`  ·  -c for the terse index  ·  "
                  f"--columns list for column choices[/]")
    rows = [(mid, p) for mid, m in sorted(models.items(), key=lambda kv: kv[0].lower()) for p in (m.get("placements") or [])]
    pins = _running_pins()
    identities = {mid: (m.get("identity") or {}) for mid, m in models.items()}
    caps = {mid: (m.get("capabilities") or {}) for mid, m in models.items()}
    _emit_table(_placements_table(rows, _current_runtimes(), columns=cols, pins=pins,
                                  identities=identities, caps=caps), wide)
    if pins:
        console.print("[dim]\\[i,j] = live GPU pins (running now)[/]")
    empty = [mid for mid, m in sorted(models.items(), key=lambda kv: kv[0].lower()) if not (m.get("placements") or [])]
    if empty:
        console.print(f"[dim]no placements: {', '.join(empty)} — `johnny induct <model>`[/]")


@registry_app.command("set-use")
def registry_set_use(
    model: str = typer.Argument(..., help="Registry model id (exact or unique substring)."),
    text: str = typer.Argument(None, help="Free-text 'what is this model good for' (e.g. \"coding — top pick\", \"general flagship / reasoning\"). Omit with --clear to remove it."),
    clear: bool = typer.Option(False, "--clear", help="Clear identity.recommended_use instead of setting it."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Set (or clear) a model's `identity.recommended_use` — the free-text "what's this good
    for" summary shown as the USE column in `registry show` (and by `registry show <model>`).

    This is deliberately separate from `induct`/`tune --use-case`, which is a narrow
    throughput/latency/context knob that only decides which sweep candidate wins — it says
    nothing about what the model is actually good at. `recommended_use` is a broader, human
    call (coding, long-context, general reasoning, speed-critical, multimodal, ...), and it's
    normally best set or refined *after* real benchmark evidence rather than guessed up
    front — e.g. run `johnny bench <model> --suite humaneval` first, then:

      johnny registry set-use qwen-27b-coder "coding — top pick (92% HumanEval pass@1)"
      johnny registry set-use qwen-27b-coder --clear

    It can also be set at induction time via `induct --recommended-use`, for the cases
    where you already know the answer going in.
    """
    from .registry import store

    reg = store.load()
    mid = _resolve_model(store.models(reg), model)
    m = store.get(reg, mid)
    ident = m.setdefault("identity", {})

    if clear:
        had = ident.pop("recommended_use", None)
        if had is None:
            console.print(f"[dim]{mid} had no recommended_use set — nothing to clear.[/]")
            return
    else:
        if not text or not text.strip():
            err.print("[red]missing text[/] — pass a description, or --clear to remove it.")
            raise typer.Exit(code=1)
        ident["recommended_use"] = text.strip()

    backup = _backup_registry_file()
    store.save(reg)
    if json_output:
        console.print(_json.dumps({"model": mid, "recommended_use": ident.get("recommended_use")}, indent=2))
        return
    tail = f"  [dim](backup {backup.name})[/]" if backup else ""
    if clear:
        console.print(f"[green]✓ cleared[/] recommended_use on [bold]{mid}[/]{tail}")
    else:
        console.print(f"[green]✓ set[/] [bold]{mid}[/] recommended_use → {ident['recommended_use']!r}{tail}")


@registry_app.command("import")
def registry_import(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be imported; write nothing."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Seed the registry from the bash launchers (stamped source=imported)."""
    from .hardware import detect as hwdetect
    from .registry import importer, schema, store

    cfg = C.load_yaml(C.get_paths().config_file) or {}
    roots = cfg.get("roots") or {}
    launchers_dir = roots.get("launchers_dir")
    models_dir = roots.get("models_dir")
    if not launchers_dir:
        err.print("[red]no launchers_dir in config[/] — run `johnny init` on a box with ~/vllm/launchers, or set roots.launchers_dir.")
        raise typer.Exit(code=1)
    fingerprint = hwdetect.detect().fingerprint
    imported = importer.import_launchers(launchers_dir, models_dir, fingerprint)
    errors = schema.validate(imported)

    n_models = len(imported.get("models", {}))
    n_pl = sum(len(m.get("placements", [])) for m in imported.get("models", {}).values())
    summary = {"models": n_models, "placements": n_pl, "fingerprint": fingerprint,
               "valid": not errors, "errors": errors, "dry_run": dry_run}
    if not dry_run and not errors:
        merged = store.merge_imported(store.load(), imported)
        store.save(merged)
    if json_output:
        console.print(_json.dumps(summary, indent=2))
        return
    console.print(f"[bold]{n_models}[/] models, [bold]{n_pl}[/] placements  [dim]fingerprint {fingerprint}[/]")
    if errors:
        for e in errors:
            err.print(f"  [red]✗[/] {e}")
        raise typer.Exit(code=1)
    if dry_run:
        console.print("[yellow]dry-run[/] — nothing written. Run without --dry-run to save.")
    else:
        console.print(f"[green]✓ wrote[/] {C.get_paths().registry_file}")


@registry_app.command("validate")
def registry_validate(json_output: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
    """Validate the registry against the schema, and report the re-tune + ctxsafe worklists.

    Three distinct things: schema *errors* (structural — fix with `registry normalize` or by
    editing), the *retune worklist* of placements that are structurally fine but lack
    trustworthy numbers (unmeasured / incomplete / stale — those need `johnny tune`), and the
    *ctxsafe worklist* of large-context placements whose configured max_model_len was never
    empirically verified safe against a real deep prefill (`johnny bench --suite ctxsafe`) —
    see AGENTS.md § Context safety for why a configured number alone isn't proof.
    """
    from .registry import normalize as N, schema, store

    reg = store.load()
    errors = schema.validate(reg)
    worklist = N.retune_worklist(reg, _current_runtimes())
    ctxsafe_worklist = N.ctxsafe_worklist(reg)
    if json_output:
        console.print(_json.dumps({"valid": not errors, "errors": errors, "worklist": worklist,
                                   "ctxsafe_worklist": ctxsafe_worklist}, indent=2))
        raise typer.Exit(code=1 if errors else 0)
    if not errors:
        console.print("[green]✓ registry is valid[/]")
    else:
        for e in errors:
            err.print(f"[red]✗[/] {e}")
    _print_worklist(worklist)
    _print_ctxsafe_worklist(ctxsafe_worklist)
    if errors:
        raise typer.Exit(code=1)


@registry_app.command("normalize")
def registry_normalize(
    apply: bool = typer.Option(False, "--apply", help="Write the normalized registry (timestamped backup)."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Give every placement a consistent shape + honest status (preview by default).

    Fills only structural gaps — gpu_count (derived from TP), the perf {peak, single}
    shape, a default source, validation_key.backend, a validated_at placeholder — plus
    empty identity fields (params, quant) that are derivable from the GGUF header or the
    model's own naming. It never invents a tok/s number: a missing benchmark stays visibly
    `unmeasured`, and an aborted run with no provenance stays `incomplete`. Those need
    `johnny tune`, which this reports as a worklist. `--apply` rewrites registry.yaml
    (a timestamped backup is kept).
    """
    import shutil
    from datetime import datetime

    from .registry import normalize as N, store

    reg = store.load()
    current = _current_runtimes()
    models = store.models(reg)
    run_cfg = C.load_yaml(C.get_paths().config_file) or {}

    plan = []
    for mid, m in sorted(models.items(), key=lambda kv: kv[0].lower()):
        for p in m.get("placements") or []:
            changes = N.normalization_changes(p)
            if changes:
                plan.append({"model": mid, "placement": p.get("id"),
                             "status": N.placement_status(p, current), "changes": changes})
    ident_plan = []
    for mid, m in sorted(models.items(), key=lambda kv: kv[0].lower()):
        fills = N.identity_gaps(mid, m, run_cfg)
        if fills:
            ident_plan.append({"model": mid, "fills": fills})
    total = sum(len(x["changes"]) for x in plan) + sum(len(x["fills"]) for x in ident_plan)
    worklist = N.retune_worklist(reg, current)

    if json_output:
        console.print(_json.dumps(
            {"placements_touched": len(plan), "models_backfilled": len(ident_plan),
             "field_updates": total, "plan": plan, "identity": ident_plan,
             "worklist": worklist, "applied": apply}, indent=2))
    elif not plan and not ident_plan:
        console.print("[green]✓ registry already normalized[/] — every placement has a consistent shape.")
    else:
        verb = "normalized" if apply else "would normalize"
        console.print(f"[bold]{verb}[/] {total} field(s) across "
                      f"{len(plan)} placement(s) + {len(ident_plan)} identity block(s):")
        for x in plan:
            console.print(f"  [cyan]{x['model']}[/] / [bold]{x['placement']}[/]  {_status_cell(x['status'])}")
            for c in x["changes"]:
                console.print(f"      [dim]{c}[/]")
        for x in ident_plan:
            fills = "  ".join(f"identity.{k} → {v!r}" for k, v in x["fills"].items())
            console.print(f"  [cyan]{x['model']}[/]  [dim]{fills}[/]")

    if apply and (plan or ident_plan):
        p = C.get_paths().registry_file
        backup = None
        if p.exists():
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = p.with_name(f"{p.name}.bak-{ts}")
            shutil.copy2(p, backup)
        for m in models.values():
            m["placements"] = [N.normalize_placement(pl) for pl in (m.get("placements") or [])]
        for x in ident_plan:
            models[x["model"]].setdefault("identity", {}).update(x["fills"])
        store.save(reg)
        if not json_output:
            console.print(f"[green]✓ wrote[/] {p}" + (f"  [dim](backup {backup.name})[/]" if backup else ""))
    elif not apply and (plan or ident_plan) and not json_output:
        console.print("\n[dim]preview only — re-run with [bold]--apply[/] to write (a backup is kept).[/]")

    if not json_output:
        _print_worklist(worklist)


def _find_placement_global(reg: dict, pid: str) -> list[tuple[str, dict]]:
    """All (model_id, placement) whose id equals `pid` (exact), else contains it (substring).
    Placement ids are near-unique, so a single-arg delete can find one without the model."""
    models = reg.get("models") or {}
    exact = [(mid, p) for mid, m in models.items() for p in (m.get("placements") or []) if p.get("id") == pid]
    if exact:
        return exact
    return [(mid, p) for mid, m in models.items() for p in (m.get("placements") or []) if pid in (p.get("id") or "")]


@registry_app.command("delete")
def registry_delete(
    target: str = typer.Argument(..., help="Placement id to delete (searched across ALL models) — or a model id when a PLACEMENT arg or --all follows."),
    placement: str = typer.Argument(None, help="Placement id within TARGET, when TARGET is a model (exact or unique substring)."),
    all_placements: bool = typer.Option(False, "--all", help="Delete ALL placements of the model TARGET (keeps the model entry)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Delete placement(s) from the registry — placement-level pruning (keeps the model).

    Three forms:
      johnny registry delete <placement-id>          # by id, found across all models
      johnny registry delete <model> <placement-id>  # scoped to one model
      johnny registry delete <model> --all           # every placement of the model

    To remove a whole model (and optionally its weights), use `johnny rm <model>`.
    """
    from .registry import store

    reg = store.load()

    if placement is not None or all_placements:
        # Model-scoped: TARGET is a model.
        model = target
        m = store.get(reg, model)
        if not m:
            err.print(f"[red]no model[/] '{model}' in the registry")
            raise typer.Exit(code=1)
        pls = m.get("placements") or []
        if not pls:
            console.print(f"[yellow]'{model}' has no placements.[/]")
            return
        if all_placements:
            targets = [(model, p) for p in pls]
        else:
            hits = [p for p in pls if p.get("id") == placement] or [p for p in pls if placement in (p.get("id") or "")]
            if not hits:
                err.print(f"[red]no placement[/] matching '{placement}' in '{model}'.")
                console.print("  known: " + ", ".join(p.get("id", "") for p in pls))
                raise typer.Exit(code=1)
            if len(hits) > 1:
                err.print(f"[red]'{placement}' is ambiguous[/] in '{model}' — matches {len(hits)}:")
                for p in hits:
                    console.print(f"  • {p.get('id')}")
                raise typer.Exit(code=1)
            targets = [(model, hits[0])]
    else:
        # Single arg: resolve as a placement id across all models.
        is_model = store.get(reg, target) is not None
        exact = [(mid, p) for mid, m in (reg.get("models") or {}).items()
                 for p in (m.get("placements") or []) if p.get("id") == target]
        if is_model and exact:  # names BOTH a model and a placement — don't guess
            err.print(f"[yellow]'{target}' is both a model id and a placement id — be explicit:[/]")
            console.print(f"  that placement:                  johnny registry delete {target} {target}")
            console.print(f"  all of the model's placements:   johnny registry delete {target} --all")
            raise typer.Exit(code=1)
        hits = _find_placement_global(reg, target)
        if not hits:
            if is_model:
                pls = (store.get(reg, target) or {}).get("placements") or []
                console.print(f"[yellow]'{target}' is a model, not a placement.[/] Pass a placement id, add --all, "
                              f"or use `johnny rm {target}` to remove the model. It has:")
                for p in pls:
                    console.print(f"  • [cyan]{p.get('id')}[/]")
            else:
                err.print(f"[red]no placement[/] matching '{target}' in any model.")
            raise typer.Exit(code=1)
        if len(hits) > 1:
            err.print(f"[red]'{target}' is ambiguous[/] — matches {len(hits)} placements:")
            for mid, p in hits:
                console.print(f"  • [cyan]{mid}[/] / {p.get('id')}")
            console.print("  [dim]disambiguate with: johnny registry delete <model> <placement>[/]")
            raise typer.Exit(code=1)
        targets = [hits[0]]

    by_model: dict[str, list[str]] = {}
    for mid, p in targets:
        by_model.setdefault(mid, []).append(p.get("id", ""))
    if not yes and not json_output:
        console.print("will delete → " + "; ".join(f"[bold]{mid}[/]: {', '.join(ids)}" for mid, ids in by_model.items()))
        if not typer.confirm("Proceed?", default=False):
            raise typer.Exit(code=1)

    deleted = []
    for mid, ids in by_model.items():
        for pid in ids:
            if store.delete_placement(reg, mid, pid):
                deleted.append({"model": mid, "placement": pid})
    store.save(reg)
    if json_output:
        console.print(_json.dumps({"deleted": deleted}, indent=2))
    else:
        for mid, ids in by_model.items():
            remaining = len((store.get(reg, mid) or {}).get("placements") or [])
            console.print(f"[green]✓ deleted[/] {len(ids)} from {mid}  [dim]({remaining} remaining)[/]")


# --------------------------------------------------------------------------- seat lifecycle (P3)
def _emit_err(e: Exception, json_output: bool):
    if json_output:
        console.print(_json.dumps({"error": str(e)}, indent=2))
    else:
        err.print(f"[red]{e}[/]")
    raise typer.Exit(code=1)


def _render_pick(it: dict) -> str:
    """One placement line for the picker: model + id + the standardized knobs + status."""
    from .registry import normalize as N

    v = N.placement_view(it["p"], it.get("current"))
    tp = f"TP{v['tp']}" if v["tp"].isdigit() else v["tp"]
    gpus = _gpus_cell(v["gpus"], (it.get("pins") or {}).get((it["model"], v["id"])))
    return (f"[bold]{v['id']}[/] [dim]({it['model']})[/]  "
            f"[dim]{gpus} · {tp} · {v['priority']} · mml{v['mml'] or '—'} · "
            f"{_fmt_toks(v['peak'], v['single'])} tok/s[/]  {_status_cell(v['status'])}")


def _pick_placement_interactive(json_output: bool) -> tuple[str, str]:
    """Open the placement picker over the whole registry; return (model, placement_id)
    or exit (cancel / nothing to pick / --json with no model)."""
    from .external import picker
    from .registry import store

    if json_output:
        _emit_err(ValueError("`up` needs a model id with --json (the picker needs a TTY)"), True)
    models = store.models(store.load())
    current = _current_runtimes()
    pins = _running_pins()
    items = [{"model": mid, "p": p, "current": current, "pins": pins}
             for mid, m in sorted(models.items(), key=lambda kv: kv[0].lower())
             for p in (m.get("placements") or [])]
    if not items:
        err.print("[yellow]no placements in the registry[/] — run `johnny induct <model>` first.")
        raise typer.Exit(code=1)
    # Show what's loadable before the interactive pick — the "list of available models"
    # johnny lacked at the `up` prompt (full detail: `johnny registry show`).
    console.print(_registry_compact_table(models))
    console.print("[dim]↓ pick a placement to load · full detail: `johnny registry show`[/]\n")
    i = picker.select(items, render=_render_pick, title="load a placement",
                      hint="↑/↓ move · enter load · q cancel")
    if i is None:
        console.print("[dim]cancelled.[/]")
        raise typer.Exit(code=0)
    chosen = items[i]
    pid = chosen["p"].get("id")
    console.print(f"[dim]→ johnny up {chosen['model']} --placement {pid}[/]")
    return chosen["model"], pid


def _render_seat(s) -> str:
    """One running-seat line for the down picker."""
    gpus = ",".join(str(g) for g in (s.gpus or [])) or "—"
    state_style = _STATE_STYLE.get(s.state, "white")
    return (f"[bold]{s.name}[/]  [dim]{s.model or '—'} · port {s.port or '—'} · "
            f"gpus {gpus} ·[/] [{state_style}]{s.state}[/]")


def _pick_seat_interactive(json_output: bool) -> str:
    """Open a picker over the running seats; return the chosen seat name or exit."""
    from .engine import all_seats, load_config
    from .external import picker

    if json_output:
        _emit_err(ValueError("`down` needs a seat id with --json (the picker needs a TTY)"), True)
    seats = all_seats(load_config())
    if not seats:
        err.print("[yellow]no running seats[/] — nothing to down.")
        raise typer.Exit(code=1)
    i = picker.select(seats, render=_render_seat, title="down a seat",
                      hint="↑/↓ move · enter down · q cancel")
    if i is None:
        console.print("[dim]cancelled.[/]")
        raise typer.Exit(code=0)
    console.print(f"[dim]→ johnny down {seats[i].name}[/]")
    return seats[i].name


@app.command(rich_help_panel=_P_SEATS)
def up(
    model: str = typer.Argument(None, help="Registry model id. Omit to list models + pick a placement (see `registry show`)."),
    placement: str = typer.Option(None, "--placement", help="Placement id or unique substring (e.g. 'tp4'); else best fit for this hardware."),
    port: int = typer.Option(None, "--port", help="Serve on this port (else auto-assigned from the configured range)."),
    swap: str = typer.Option(None, "--swap", help="Seat to evict to free its GPUs/port."),
    force: bool = typer.Option(False, "--force", help="Place even if GPUs are busy."),
    wait: bool = typer.Option(False, "--wait", help="Block until the seat is serving."),
    profile: str = typer.Option(None, "--profile", help="Bring up a whole named profile instead "
                                "(alias for `johnny profile up`)."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Bring up a model seat (spawn on free GPUs, or swap a named seat).

    With no model id, lists the available models then opens an interactive picker
    over every registry placement — ↑/↓ to choose, enter to load. For a plain
    (non-interactive) list of what you can load, run `johnny registry show`.
    """
    from .engine import launch

    if profile is not None:
        if model is not None:
            _emit_err(ValueError("--profile brings up a fleet; don't pass a model too"), json_output)
        profile_up(profile, wait=wait, json_output=json_output)
        return

    if model is None:
        model, placement = _pick_placement_interactive(json_output)

    try:
        res = launch.up(model, placement_id=placement, port=port, swap=swap, force=force, wait=wait)
    except Exception as e:
        _emit_err(e, json_output)
    if json_output:
        console.print(_json.dumps(res, indent=2))
        return
    st = res.get("state")
    console.print(
        f"[green]●[/] {res['action']} [bold]{res['seat']}[/] · model={res['model']} · "
        f"port={res.get('port')} · gpus={res.get('gpus') or '—'} · state=[{_STATE_STYLE.get(st, 'white')}]{st}[/]"
    )
    if res.get("endpoint"):
        console.print(f"  endpoint: {res['endpoint']}")
    if st == "loading":
        console.print(f"  [dim]loading — poll `johnny resolve {res['model']}` or tail `johnny logs {res['seat']}`[/]")


@app.command(rich_help_panel=_P_SEATS)
def down(
    seat: str = typer.Argument(None, help="Seat/container name (or model id). Omit to pick interactively."),
    drain: bool = typer.Option(False, "--drain", help="Graceful drain (no-op without a router)."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Tear down a single named seat (never siblings).

    With no seat id, opens an interactive picker over the running seats —
    ↑/↓ to choose, enter to down.
    """
    from .engine import launch

    if seat is None:
        seat = _pick_seat_interactive(json_output)

    try:
        res = launch.down(seat, drain=drain)
    except Exception as e:
        _emit_err(e, json_output)
    console.print(_json.dumps(res, indent=2) if json_output else f"[green]✓[/] down {res['seat']}")


@app.command(rich_help_panel=_P_SEATS)
def swap(
    seat: str = typer.Argument(..., help="Running seat to replace."),
    model: str = typer.Argument(..., help="Model to launch in its place."),
    wait: bool = typer.Option(False, "--wait", help="Block until the replacement seat is serving."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Replace one seat in place (same cards/port)."""
    from .engine import launch

    try:
        res = launch.swap(seat, model, wait=wait)
    except Exception as e:
        _emit_err(e, json_output)
    console.print(_json.dumps(res, indent=2) if json_output else
                  f"[green]●[/] swapped {seat} → [bold]{res['seat']}[/] (state {res.get('state')})")


@app.command(rich_help_panel=_P_SEATS)
def reap(
    idle_ttl: int = typer.Option(None, "--idle-ttl", help="Idle seconds before reaping (default 1800)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List what would be reaped; evict nothing."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Evict idle, unpinned seats so the GPUs reach deep idle. Stateless / cron-able."""
    from .engine import service

    actions = service.reap(idle_ttl=idle_ttl, dry_run=dry_run)
    if json_output:
        console.print(_json.dumps(actions, indent=2))
        return
    if not actions:
        console.print("[dim]no seats to consider.[/]")
        return
    t = Table(title="johnny reap" + (" (dry-run)" if dry_run else ""), title_style="bold")
    for col in ("SEAT", "ACTION", "IDLE (s)", "REASON"):
        t.add_column(col)
    style = {"reap": "red", "would-reap": "yellow", "keep": "green", "skip": "dim"}
    for a in actions:
        t.add_row(a["seat"], f"[{style.get(a['action'], 'white')}]{a['action']}[/]",
                  str(a.get("idle_s", "—")), a.get("reason", ""))
    console.print(t)


@app.command(rich_help_panel=_P_SEATS)
def pin(
    seat: str = typer.Argument(..., help="Seat to exempt from the reaper (see `johnny status`)."),
    ttl: int = typer.Option(None, "--ttl", help="Seconds; omit for indefinite."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Exempt a seat from the reaper (ephemeral pin in the telemetry SQLite)."""
    from .telemetry import collect

    collect.add_pin(seat, ttl_s=ttl)
    msg = {"pinned": seat, "ttl_s": ttl}
    console.print(_json.dumps(msg, indent=2) if json_output else
                  f"[green]✓[/] pinned {seat}" + (f" for {ttl}s" if ttl else " (indefinite)"))


@app.command(rich_help_panel=_P_SEATS)
def unpin(seat: str = typer.Argument(..., help="Seat to remove the reaper exemption from."), json_output: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
    """Remove a seat's reaper exemption."""
    from .telemetry import collect

    collect.remove_pin(seat)
    console.print(_json.dumps({"unpinned": seat}, indent=2) if json_output else f"[green]✓[/] unpinned {seat}")


@app.command(rich_help_panel=_P_SEATS)
def resolve(
    target: str = typer.Argument(..., help="Role, seat, or model id."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Resolve a role/seat/model to its live endpoint + readiness (the SAINT hot path)."""
    from .engine import service

    res = service.resolve(target)
    if json_output:
        console.print(_json.dumps(res, indent=2))
        return
    st = res["state"]
    console.print(
        f"[{_STATE_STYLE.get(st, 'white')}]{st}[/] · seat={res.get('seat') or '—'} · "
        f"model={res.get('model')} · endpoint={res.get('endpoint') or '—'} · "
        f"eta_s={res.get('eta_s')} · queue={res.get('queue_depth')}"
    )


@app.command(rich_help_panel=_P_OBSERVE)
def logs(
    seat: str = typer.Argument(..., help="Seat whose container logs to tail."),
    follow: bool = typer.Option(False, "-f", "--follow", help="Stream new log lines (docker logs -f)."),
    tail: int = typer.Option(200, "--tail", help="Lines of history to show (default 200)."),
) -> None:
    """Tail a seat's logs (docker logs), with launch-failure context."""
    from .engine import all_seats, driver_for

    target = None
    for s in all_seats():
        labels = (s.extra or {}).get("labels", {})
        if seat in (s.name, s.model, labels.get("johnny.model")):
            target = s
            break
    if not target:
        err.print(f"[red]no running seat[/] '{seat}'")
        raise typer.Exit(code=1)
    drv = driver_for(target)
    out = drv.logs(target.name, follow=follow, tail=tail)
    if not follow and out is not None:
        console.print(out)


@app.command(rich_help_panel=_P_OBSERVE)
def metrics(
    seat: str = typer.Argument(..., help="Seat to report metrics for."),
    history: bool = typer.Option(False, "--history", help="Aggregate trends from the telemetry SQLite."),
    since: int = typer.Option(None, "--since", help="History window in seconds (default: all)."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show telemetry for a seat: live vLLM /metrics, or --history rollup trends."""
    if history:
        from .telemetry import collect

        rows = collect.rollup(seat=seat, since_s=since)
        if json_output:
            console.print(_json.dumps(rows, indent=2))
            return
        if not rows:
            console.print(f"[dim]no telemetry history for '{seat}' yet.[/]")
            return
        t = Table(title=f"metrics history — {seat}", title_style="bold")
        for col in ("SEAT", "SAMPLES", "AVG gen tok/s", "MAX gen tok/s", "AVG TTFT ms", "PEAK running"):
            t.add_column(col)
        for r in rows:
            t.add_row(r["seat"], str(r["samples"]),
                      f"{r['avg_gen_tok_s']:.1f}" if r["avg_gen_tok_s"] else "—",
                      f"{r['max_gen_tok_s']:.1f}" if r["max_gen_tok_s"] else "—",
                      f"{r['avg_ttft_ms']:.1f}" if r["avg_ttft_ms"] else "—",
                      str(r["peak_running"] if r["peak_running"] is not None else "—"))
        console.print(t)
        return

    from .engine import all_seats, driver_for

    target = None
    for s in all_seats():
        if seat in (s.name, s.model):
            target = s
            break
    if not target:
        err.print(f"[red]no running seat[/] '{seat}'")
        raise typer.Exit(code=1)
    m = driver_for(target).metrics(target.name)
    if json_output:
        console.print(_json.dumps(m, indent=2))
        return
    for k, v in m.items():
        console.print(f"  {k}: {v}")


# --------------------------------------------------------------------------- induction (P4)
def _render_plan(pl: dict) -> None:
    a = pl["audit"]
    console.print(f"[bold]{pl['model_id']}[/]  [dim]{pl['path']}[/]")
    console.print(f"  arch={a['arch']} quant={a['quant']} size={a['size_gb']}GB native_ctx={a['native_ctx']} · "
                  f"[bold]device={pl.get('device')}[/] · embeddings={pl.get('embeddings')} · "
                  f"free GPUs={pl['free_gpus']} · priors={pl['priors']}")
    if pl.get("device") == "cpu":
        for v in pl["viable"]:
            console.print(f"  [cyan]CPU placement[/] — fits host RAM ({v.get('per_host_gb')}GB weights)")
        for p in pl.get("pruned", []):
            console.print(f"  [yellow]✗ {p.get('tp')}[/] — {p.get('reason')}")
    else:
        vt = Table(title="viable placements", title_style="bold")
        for col in ("TP/GPUs", "QUANT", "GB/GPU", "KV-CEILING CTX"):
            vt.add_column(col)
        for v in pl["viable"]:
            # vLLM placements carry "tp"; llamacpp layer-split ones carry "gpu_count".
            span = v.get("tp") if v.get("tp") is not None else f"{v.get('gpu_count', '?')}×layer-split"
            vt.add_row(str(span), str(v.get("quant")), str(v.get("per_gpu_gb")), str(v.get("kv_ceiling_ctx")))
        console.print(vt)
        if pl["pruned"]:
            console.print("[dim]pruned:[/]")
            for p in pl["pruned"]:
                console.print(f"  [yellow]✗ tp={p.get('tp')}[/] — {p.get('reason')}")
    for w in pl.get("warnings", []):
        console.print(f"[yellow]⚠ {w}[/]")
    if pl.get("arch_supported") is False:
        console.print(f"[red]✗ unsupported architecture[/] — {pl.get('arch_warning')}")
        console.print("[dim]nothing to sweep; induct would abort before launching any seat.[/]")
        return
    console.print(f"[bold]{len(pl['points'])}[/] candidate config point(s) to sweep "
                  f"[dim](seeded search, not a brute grid)[/]")


def _norm_kv(s: str) -> str:
    """Normalize a --kv value to vLLM's KV-cache-dtype vocabulary (16-bit 'auto' or 'fp8')."""
    v = s.strip().lower()
    if v in ("8", "fp8"):
        return "fp8"
    if v in ("16", "fp16", "bf16", "auto"):
        return "auto"
    err.print(f"[red]--kv must be 8/fp8 or 16/fp16/auto[/] (got {s!r})")
    raise typer.Exit(code=2)


def _kv_dtypes(kv: str | None, sweep_kv: bool) -> tuple:
    """Resolve --kv / --sweep-kv to the KV-cache dtypes to sweep. --sweep-kv wins if both
    are given. vLLM offers only 16-bit ('auto') and 8-bit ('fp8')."""
    if sweep_kv:
        return ("auto", "fp8")
    if kv:
        return (_norm_kv(kv),)
    return ("auto",)


_KNOB_LABEL = {"tp": "TP", "max_model_len": "MML", "gpu_memory_util": "GMU",
               "max_num_seqs": "SEQS", "max_num_batched_tokens": "BT", "kv_cache_dtype": "KV",
               "threads": "THREADS", "parallel": "PAR", "n_gpu_layers": "NGL", "n_cpu_moe": "NCMOE"}


def _varying_knobs(results) -> list[str]:
    """The knob keys that VARIED across these benched points (so tables/pickers show
    only what differed). Backend-agnostic (vLLM tp/gmu/seqs, llama.cpp threads/parallel)."""
    keys = ["tp", "threads", "parallel", "n_gpu_layers", "n_cpu_moe", "gpu_memory_util",
            "max_num_seqs", "max_num_batched_tokens", "kv_cache_dtype", "max_model_len"]
    varying = [k for k in keys if len({(r.get("point") or {}).get(k) for r in results}) > 1]
    return varying or ["max_num_seqs"]


def _render_sweep_results(results, winner, use_case) -> list:
    """Compact table of the benched points showing only the knobs that varied, plus
    peak/single/KV, with the winner marked and the pick basis spelled out. Returns the
    ok results in row order (the seat picker's index space)."""
    ok_rows = [r for r in results if r.get("ok")]
    pts = [r for r in results if r.get("point")]
    if len(pts) < 2:
        return ok_rows
    varying = _varying_knobs(pts)

    t = Table(title="sweep results", title_style="dim", title_justify="left", pad_edge=False)
    t.add_column("")
    for k in varying:
        t.add_column(_KNOB_LABEL.get(k, k), no_wrap=True)
    for col in ("PEAK", "SINGLE", "KV-TOK"):
        t.add_column(col, no_wrap=True)
    for r in results:
        p = r.get("point") or {}
        win = r is winner  # winner is one of the result dicts — identity match, no fragile sig
        row = ["[green]✓[/]" if win else ""] + [str(p.get(k)) for k in varying]
        if r.get("ok"):
            kv = r.get("kv_cache_tokens")
            row += [str(r.get("peak_tok_s")), str(r.get("single_tok_s")), f"{kv/1e6:.2f}M" if kv else "—"]
        else:
            row += ["[red]fail[/]", "—", "—"]
        t.add_row(*row)
    console.print(t)
    basis = {"latency": "fastest single-stream · ties (≤5%) tipped by >15% peak",
             "context": "largest context · ties (≤5%) tipped by >15% peak"}.get(
                 use_case, "highest peak throughput · ties (≤5%) tipped by >15% single")
    console.print(f"  [dim]winner basis ({use_case or 'throughput'}): {basis}[/]")
    return ok_rows


def _pick_seats(results, winner, use_case, state) -> list | None:
    """End-of-sweep seat picker: after the sweep table, open a checkbox picker (↑/↓ move,
    space toggle, enter accept — same TUI as `up`'s placement picker) over the benched
    seats, so e.g. the best tp=1/tp=2 runs can be kept alongside the tp=4 winner. The
    winner starts selected; cancel or an empty pick → pipeline default (winner only)."""
    from .external import picker as _pk

    ok_rows = _render_sweep_results(results, winner, use_case)
    state["rendered"] = True
    if len(ok_rows) < 2:
        return None  # nothing to choose between — pipeline writes the winner
    varying = _varying_knobs(ok_rows)
    pts = [r.get("point") or {} for r in ok_rows]
    widths = {k: max(len(str(p.get(k))) for p in pts) for k in varying}

    def _line(r):
        p = r.get("point") or {}
        knobs = " ".join(f"{_KNOB_LABEL.get(k, k)}={str(p.get(k)).ljust(widths[k])}" for k in varying)
        kv = r.get("kv_cache_tokens")
        perf = f"peak {r.get('peak_tok_s')} · single {r.get('single_tok_s')} tok/s"
        if kv:
            perf += f" · KV {kv/1e6:.2f}M"
        return f"{knobs}  {perf}" + ("  [green]✓ winner[/]" if r is winner else "")

    pre = {ok_rows.index(winner)} if winner in ok_rows else set()
    idxs = _pk.multi_select(ok_rows, render=_line, title="write launchers (placements) for which seats?",
                            preselected=pre)
    if not idxs:
        return None  # cancelled or emptied — pipeline writes the winner only
    return [ok_rows[i] for i in idxs]


def _run_induct(model, use_case, device, tp, embeddings, bench, plan, resume, max_points, yes,
                json_output, kv=None, sweep_kv=False, mml=None, recommended_use=None) -> None:
    """Shared induction implementation for `induct` and `tune`. A plain function so neither
    command invokes the other: calling a Typer command directly passes its unfilled options
    as raw OptionInfo sentinels (the `--tp <OptionInfo>` bug), never the real defaults.

    `use_case` here is the narrow sweep winner-pick tiebreaker (throughput/latency/context —
    see `_render_sweep_results`/`_pick_seats`); `recommended_use`, when passed, is the
    broader free-text "what's this good for" note — unrelated field, written to
    identity.recommended_use once a placement is actually won (never on --plan or on error)."""
    from .induct import pipeline

    kv_dtypes = _kv_dtypes(kv, sweep_kv)

    if plan:
        try:
            pl = pipeline.plan(model, max_points=max_points, device=device, embeddings=embeddings, tp=tp,
                               kv_dtypes=kv_dtypes, mml_override=mml)
        except Exception as e:
            _emit_err(e, json_output)
        if json_output:
            console.print(_json.dumps(pl, indent=2))
        else:
            _render_plan(pl)
        return

    if not yes and not json_output:
        try:
            pl = pipeline.plan(model, max_points=max_points, device=device, embeddings=embeddings, tp=tp,
                               kv_dtypes=kv_dtypes, mml_override=mml)
        except Exception as e:
            _emit_err(e, json_output)
        _render_plan(pl)
        if not pl["points"]:
            raise typer.Exit(code=1)
        # --resume: points already benched into state.json replay from cache — only ask
        # the launch question about the ones that will really load a model.
        cached = pipeline.cached_count(pl["model_id"], pl["points"]) if resume else 0
        n_new = len(pl["points"]) - cached
        where = "CPU" if pl.get("device") == "cpu" else "GPU"
        if n_new == 0:
            console.print(f"[dim]all {len(pl['points'])} point(s) already benched — replaying from cache, no launches.[/]")
        elif not typer.confirm(f"Launch {n_new} {where} tuning seat(s)?"
                               + (f" ({cached} more replay from cache)" if cached else "")
                               + " (each is a real load + bench)"):
            raise typer.Exit(code=1)

    prog = None if json_output else (lambda m: console.print(f"[dim]· {m}[/]"))
    # End-of-sweep seat picker: interactive terminals get to choose which benched seats
    # become launchers (registry placements); --yes/--json/non-tty keep winner-only.
    sel = {"rendered": False}
    picker = (None if (json_output or yes or not _sys.stdin.isatty())
              else (lambda results, winner: _pick_seats(results, winner, use_case, sel)))
    try:
        res = pipeline.run(model, use_case=use_case, bench=bench, resume=resume, max_points=max_points,
                           progress=prog, device=device, embeddings=embeddings, tp=tp,
                           kv_dtypes=kv_dtypes, mml_override=mml, select=picker)
    except Exception as e:
        _emit_err(e, json_output)
    if recommended_use and not res.get("error"):
        from .registry import store as _store

        reg2 = _store.load()
        m2 = _store.get(reg2, res.get("model_id") or model)
        if m2 is not None:
            m2.setdefault("identity", {})["recommended_use"] = recommended_use
            _backup_registry_file()
            _store.save(reg2)
            if not json_output:
                console.print(f"  [dim]recommended_use set on {res.get('model_id') or model}: "
                              f"{recommended_use!r}[/]")
        elif not json_output:
            console.print(f"[yellow]couldn't resolve a registry model id to set recommended_use on "
                          f"(tried '{res.get('model_id') or model}') — use `registry set-use` once it's written.[/]")
    if json_output:
        console.print(_json.dumps(res, indent=2, default=str))
        return
    if res.get("error"):
        err.print(f"[red]{res['error']}[/]")
        raise typer.Exit(code=1)
    if not sel["rendered"]:
        _render_sweep_results(res.get("results") or [], res.get("winner"), use_case)
    w = res.get("winner")
    if w:
        wp = w["point"]
        # Show only the knobs that apply to this backend (vLLM: tp/seqs/kv/gmu; llama.cpp:
        # threads/parallel/ngl) — skip the ones that are None so the line isn't full of noise.
        knobs = [("TP", "tp"), ("threads", "threads"), ("par", "parallel"), ("ngl", "n_gpu_layers"),
                 ("seqs", "max_num_seqs"), ("kv", "kv_cache_dtype"), ("gmu", "gpu_memory_util"),
                 ("mml", "max_model_len")]
        shown = " · ".join(f"{label}={wp.get(key)}" for label, key in knobs if wp.get(key) is not None)
        console.print(f"[green]✓ winner[/] {shown} → peak {w.get('peak_tok_s')} · single {w.get('single_tok_s')} tok/s")
        pids = res.get("placement_ids") or ([res["placement_id"]] if res.get("placement_id") else [])
        if len(pids) == 1:
            console.print(f"  wrote placement [bold]{pids[0]}[/] to the registry")
        else:
            console.print(f"  wrote {len(pids)} placements to the registry:")
            for pid in pids:
                console.print(f"    [bold]{pid}[/]")
    else:
        console.print("[yellow]no winning config[/] (all points failed — see the report)")
    console.print(f"  report: {res['report']}  ·  bench: {res['bench']}")


@app.command(rich_help_panel=_P_MODELS)
def induct(
    model: str = typer.Argument(..., help="HF id, registry id, or local path."),
    use_case: str = typer.Option(None, "--use-case", help="Winner pick: throughput (max peak tok/s under concurrency) | latency (fastest single-stream tok/s) | context (largest usable context)"),
    device: str = typer.Option("auto", "--device", help="gpu | cpu | auto (auto falls back to CPU if no GPU fits)."),
    tp: int = typer.Option(None, "--tp", help="Force tensor-parallel size: sweep only this TP (must be a viable placement). Overrides the auto winner's TP."),
    kv: str = typer.Option(None, "--kv", help="Force KV-cache dtype: 8/fp8 or 16/fp16/auto (vLLM supports these two)."),
    sweep_kv: bool = typer.Option(False, "--sweep-kv", help="Sweep both 16-bit and 8-bit KV cache (fp8 ≈ 2× the context per GB). Overrides --kv."),
    mml: int = typer.Option(None, "--mml", help="Force max_model_len (capped at the VRAM ceiling for the KV dtype)."),
    embeddings: bool = typer.Option(None, "--embeddings/--no-embeddings", help="Force embeddings vs generative bench (default: auto-detect)."),
    bench: bool = typer.Option(False, "--bench", help="Also run the quality harness (heavy/opt-in)."),
    plan: bool = typer.Option(False, "--plan", help="Dry preview: viable placements + candidate grid, no launches."),
    resume: bool = typer.Option(False, "--resume", help="Continue a previous run, skipping done points."),
    max_points: int = typer.Option(None, "--max-points", help="Cap candidate points (bounded runs)."),
    recommended_use: str = typer.Option(None, "--recommended-use", help="Free-text 'what is this model good for' (coding, long-context, general reasoning, speed-critical, multimodal, ...) written to identity.recommended_use once a placement is won — unrelated to --use-case (that's only a sweep winner-pick tiebreaker). Often clearer *after* real bench evidence than up front; see `johnny registry set-use` to set/update it later."),
    yes: bool = typer.Option(False, "--yes", help="Skip prompts (pre-sweep confirmation + end-of-sweep seat picker; writes the winner only)."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Auto-tune a model into an optimal placement (tuning by default; GPU or CPU)."""
    _run_induct(model, use_case, device, tp, embeddings, bench, plan, resume, max_points, yes,
                json_output, kv, sweep_kv, mml, recommended_use)


@app.command(rich_help_panel=_P_MODELS)
def tune(
    model: str = typer.Argument(..., help="Registry id or local path."),
    use_case: str = typer.Option(None, "--use-case", help="Winner pick: throughput (max peak tok/s under concurrency) | latency (fastest single-stream tok/s) | context (largest usable context)"),
    device: str = typer.Option("auto", "--device", help="gpu | cpu | auto (auto falls back to CPU only if no GPU placement fits). Use `cpu` to force a CPU bench."),
    tp: int = typer.Option(None, "--tp", help="Force tensor-parallel size: sweep only this TP (must be viable on this hardware)."),
    kv: str = typer.Option(None, "--kv", help="Force KV-cache dtype: 8/fp8 or 16/fp16/auto (vLLM supports these two)."),
    sweep_kv: bool = typer.Option(False, "--sweep-kv", help="Sweep both 16-bit and 8-bit KV cache (fp8 ≈ 2× the context per GB). Overrides --kv."),
    mml: int = typer.Option(None, "--mml", help="Force max_model_len (capped at the VRAM ceiling for the KV dtype)."),
    resume: bool = typer.Option(False, "--resume", help="Continue a previous run, skipping done points."),
    max_points: int = typer.Option(None, "--max-points", help="Cap candidate points (bounded runs)."),
    recommended_use: str = typer.Option(None, "--recommended-use", help="Free-text 'what is this model good for' — see `johnny induct --help` for the full explanation. Best set/refreshed after real bench evidence; `johnny registry set-use` does the same thing without a re-tune."),
    yes: bool = typer.Option(False, "--yes", help="Skip prompts (pre-sweep confirmation + end-of-sweep seat picker; writes the winner only)."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Re-tune an existing model (induction, tuning-only). A focused alias for `induct`
    with tuning-only behavior; see `johnny induct --help` for the full option set."""
    _run_induct(model, use_case, device, tp, None, False, False, resume, max_points, yes,
                json_output, kv, sweep_kv, mml, recommended_use)


@app.command(rich_help_panel=_P_MODELS)
def bench(
    target: str = typer.Argument(None, help="Model id or placement id (exact or unique substring). Omit to pick from the registry."),
    suite: str = typer.Option("perf,arc", "--suite", help="Comma-separated suites: perf (throughput/single-stream bench — refreshes the placement's perf numbers) · arc (ARC-Challenge CoT accuracy; needs the optional eval deps: `pipx inject johnny-fleet openai datasets`) · icl (in-context-learning pattern-completion probe; needs `openai`) · needle (positional-recall/long-context probe against a bundled code corpus; needs `openai`) · depth (prefill/decode throughput + latency vs. context depth via llama-benchy; needs `llama-benchy`) · humaneval (real HumanEval pass@1 via lm-eval + a chat-aware re-scorer; needs `pipx inject johnny-fleet 'lm-eval[api]'`) · automationbench (real agentic tool-use eval via Zapier's public AutomationBench — 600 tasks over simulated SaaS tools, self-bootstraps a vendored `uv`-managed checkout; needs `uv` on PATH; see `--domains`) · ctxsafe (empirical context-safety probe — walks real needle-in-haystack requests at progressively deeper depths up to max_model_len against a dedicated disposable seat, live rocm-smi VRAM polling, real crash detection; writes quality.ctxsafe with the verified-safe depth vs. configured max_model_len vs. trained native_context — see AGENTS.md's Context safety section; needs `openai`, ideally `tiktoken`)."),
    limit: int = typer.Option(None, "--limit", help="arc/humaneval: only the first N questions/problems — a quick smoke. Full sets are 1172 CoT questions (arc, an hour-ish on a mid-size seat) / 164 problems (humaneval). automationbench: only the first N tasks (across --domains, dataset order) — full public set is 600 (100/domain). ctxsafe: cap the deepest depth tested (tokens) — for placements whose max_model_len is too large to sweep to in one run."),
    concurrency: int = typer.Option(8, "--concurrency", help="arc/humaneval: parallel requests against the seat. automationbench: max concurrent tasks (--max-concurrent)."),
    domains: str = typer.Option("all", "--domains", help="automationbench: comma-separated domains (sales/marketing/operations/support/finance/hr) or 'all'."),
    thinking: bool = typer.Option(False, "--thinking/--no-thinking", help="arc/icl/needle/humaneval: leave model thinking on. Default off — reasoning models score ~0 (or truncate mid-answer) when the answer drowns in an unclosed think block."),
    yes: bool = typer.Option(False, "--yes", help="Skip the temp-seat launch confirmation."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Benchmark an inducted placement (quality + perf) and record the scores.

    Reuses the placement's *running* seat when one matches (no relaunch); otherwise
    launches a temporary tuning seat from the registry knobs and stops it after.

    TARGET accepts a registry model id (its placement; picker if it has several) or a
    placement id — exact or unique substring:

      johnny bench Ornith-1.0-9B-AWQ-FP8

      johnny bench induct-tp2-gmu0.92-seqs32-bt16384-mml262144

      johnny bench tp2-gmu0.92 --suite arc --limit 100

    Perf refreshes the placement's `perf`; quality lands under its `quality` block;
    both plus a BENCH_REPORT.md under the runs dir. To re-tune knobs instead, see
    `johnny tune`.
    """
    from . import bench as B
    from .engine import load_config as _load_cfg
    from .registry import store

    suites = [s.strip().lower() for s in suite.split(",") if s.strip()]
    if not suites:
        err.print("[red]--suite is empty[/] — available: " + ", ".join(B.SUITES))
        raise typer.Exit(code=1)
    for s in suites:
        if s in B.PLANNED:
            err.print(f"[yellow]suite '{s}' isn't wired yet[/] — {B.PLANNED[s]}.")
            raise typer.Exit(code=1)
        if s not in B.SUITES:
            err.print(f"[red]unknown suite '{s}'[/] — available: {', '.join(B.SUITES)}; planned: {', '.join(B.PLANNED)}.")
            raise typer.Exit(code=1)

    reg = store.load()
    if target:
        cands = B.resolve_target(reg, target)
    else:
        if json_output:
            _emit_err(ValueError("`bench` needs a model or placement id with --json (the picker needs a TTY)"), True)
        cands = [(mid, p) for mid, m in sorted(store.models(reg).items())
                 for p in (m.get("placements") or [])]
    if not cands:
        err.print(f"[red]'{target}' matches no model or placement[/] — ids: `johnny registry show`; "
                  "create one: `johnny induct <model>`.")
        raise typer.Exit(code=1)
    if len(cands) == 1:
        model_id, placement = cands[0]
    else:
        if json_output:
            _emit_err(ValueError(f"'{target}' is ambiguous ({len(cands)} placements) — pass a placement id"), True)
        from .external import picker as _pk

        current, pins = _current_runtimes(), _running_pins()
        items = [{"model": mid, "p": p, "current": current, "pins": pins} for mid, p in cands]
        i = _pk.select(items, render=_render_pick, title="bench which placement?",
                       hint="↑/↓ move · enter bench · q cancel")
        if i is None:
            console.print("[dim]cancelled.[/]")
            raise typer.Exit(code=0)
        model_id, placement = cands[i]

    pid = placement.get("id")
    cfg = _load_cfg()
    if not yes and not json_output and B.find_running_seat(model_id, pid, cfg) is None:
        console.print(f"[dim]no running seat matches {pid} — a temporary tuning seat will be "
                      "launched (and stopped after).[/]")
        if "arc" in suites and not limit:
            console.print("[dim]full ARC-Challenge is 1172 CoT questions — `--limit 100` is a quick smoke.[/]")
        if "humaneval" in suites and not limit:
            console.print("[dim]full HumanEval is 164 problems — `--limit 20` is a quick smoke.[/]")
        if "automationbench" in suites and domains == "all" and not limit:
            console.print("[dim]full AutomationBench (all domains) is 600 tasks — `--domains sales` "
                          "or `--limit 20` is a quick smoke.[/]")
        if not typer.confirm(f"Bench {model_id} · {pid} ({', '.join(suites)})?"):
            raise typer.Exit(code=1)

    prog = None if json_output else (lambda m: console.print(f"[dim]· {m}[/]"))
    try:
        res = B.run(model_id, placement, suites, cfg=cfg, limit=limit, concurrency=concurrency,
                    thinking=thinking, automationbench_domains=domains, progress=prog)
    except Exception as e:
        _emit_err(e, json_output)
    if json_output:
        console.print(_json.dumps(res, indent=2, default=str))
        return
    if res.get("error"):
        err.print(f"[red]{res['error']}[/]")
        raise typer.Exit(code=1)
    failed = False
    for s in suites:
        r = (res.get("results") or {}).get(s) or {}
        if s == "ctxsafe" and r.get("tested_depths") is not None:
            # Always render the three-way gap (trained native_context vs configured
            # max_model_len vs empirically verified-safe) — that gap is exactly what
            # caused the 2026-08-06 incident this suite exists to catch. A crash is a
            # real finding, not just a suite error, so it's shown even though ok=False.
            native, mml = r.get("native_context"), r.get("configured_max_model_len")
            safe, crash = r.get("verified_safe_tokens"), r.get("crashed_at")
            gap = f"native_context={native:,}" if native else "native_context=?"
            gap += f" · configured max_model_len={mml:,}" if mml else ""
            if crash is not None:
                failed = True
                safe_part = f"verified_safe={safe:,} tok" if safe else "no depth verified safe"
                console.print(f"[red]✗ ctxsafe — CRASHED at {crash:,} tok[/] ({safe_part}) · {gap} "
                              f"· VRAM peak {r.get('vram_peak_gb')}GB")
                console.print("  [red]placement is UNSAFE at its configured max_model_len[/] — "
                              "lower max_model_len (see AGENTS.md § Context safety) or re-tune.")
            elif safe:
                note = " [yellow](capped by --limit — not the full configured cap)[/]" if r.get("limited") else ""
                shortfall = f" [yellow](< configured {mml:,})[/]" if mml and safe < mml else ""
                console.print(f"[green]✓ ctxsafe[/] verified_safe={safe:,} tok{shortfall} · {gap}"
                              f" · VRAM peak {r.get('vram_peak_gb')}GB{note}")
            else:
                failed = True
                console.print(f"[red]✗ ctxsafe[/] — {r.get('error')} · {gap}")
        elif not r.get("ok"):
            failed = True
            console.print(f"[red]✗ {s}[/] — {r.get('error')}")
        elif s == "perf":
            kv = r.get("kv_cache_tokens")
            console.print(f"[green]✓ perf[/] peak {r.get('peak_tok_s')} · single {r.get('single_tok_s')} tok/s"
                          + (f" · KV {kv/1e6:.2f}M tok" if kv else ""))
        elif s == "arc":
            console.print(f"[green]✓ arc[/] ARC-Challenge {r.get('accuracy_pct')}% "
                          f"({r.get('correct')}/{r.get('total')}"
                          + (f", first {r['limit']}" if r.get("limit") else "") + ")")
        elif s == "icl":
            total = (r.get("pass") or 0) + (r.get("fail") or 0)
            console.print(f"[green]✓ icl[/] {r.get('pass')}/{total}"
                          + (f", first {r['limit']}" if r.get("limit") else ""))
        elif s == "needle":
            total = (r.get("pass") or 0) + (r.get("fail") or 0)
            console.print(f"[green]✓ needle[/] {r.get('pass')}/{total}")
        elif s == "depth":
            pts = ", ".join(f"d={p['depth']} pp={p['pp_tok_s']}/tg={p['tg_tok_s']} tok/s"
                            for p in (r.get("points") or []))
            console.print(f"[green]✓ depth[/] {pts}")
        elif s == "humaneval":
            console.print(f"[green]✓ humaneval[/] pass@1 {r.get('pass_at_1_pct')}% "
                          f"({r.get('passed')}/{r.get('total')}"
                          + (f", first {r['limit']}" if r.get("limit") else "") + ")")
        elif s == "automationbench":
            console.print(f"[green]✓ automationbench[/] pass rate {r.get('pass_rate_pct')}% "
                          f"({r.get('passed')}/{r.get('total')}, avg partial credit {r.get('avg_score_pct')}%)"
                          f" · domains={r.get('domains_run')}"
                          + (f" · aborted={r['aborted']}" if r.get("aborted") else ""))
            for dm in (r.get("domains") or []):
                console.print(f"    {dm['domain']}: {dm['passed']}/{dm['total']} ({dm.get('pass_rate_pct')}%)")
    if res.get("registry_updated"):
        console.print(f"  [dim]scores recorded on [bold]{pid}[/bold] in the registry[/]")
    console.print(f"  report: {res['report']}")
    if failed:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- discovery (P5)
_VERDICT_STYLE = {"fits": "green", "tight": "yellow", "offload": "yellow", "wont-fit": "red",
                  "unknown": "dim", "inducted": "green"}


def _dtype_cell(d: dict) -> str:
    """Render a dtype-fit verdict: native ✓ / not-native ✗ / unknown —."""
    ok = (d or {}).get("ok")
    need = (d or {}).get("need")
    if ok is True:
        return f"[green]✓ {need}[/]"
    if ok is False:
        return f"[red]✗ {need or (d or {}).get('detail', '')}[/]"
    return "[dim]—[/]"


def _fmt_b(n: float) -> str:
    if n >= 999.5e9:
        return f"{n / 1e12:.1f}T"
    v = n / 1e9
    return f"{v:.0f}B" if v >= 9.95 else f"{v:.1f}B"


def _fmt_count(n) -> str:
    if not n:
        return "—"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return str(n)


def _model_cell(r: dict) -> str:
    """Model id with inline ✓ (inducted) / 🔒 (gated) markers — one-glyph booleans
    don't earn their own table columns when names already wrap. The ✓ leads (padded
    so every name aligns and a mark pops at the left edge); a suffix after a long
    name is invisible."""
    mark = "[green]✓[/] " if r.get("inducted") else "  "
    return mark + r["id"] + (" 🔒" if r.get("gated") else "")


def _params_cell(r: dict) -> str:
    """'754B·A44B' for MoE, plain '27B' for dense, '—' when nothing states it."""
    p, a = r.get("params"), r.get("active_params")
    if not p:
        return "[dim]—[/]"
    if a and a < p:
        return f"{_fmt_b(p)}[dim]·A{_fmt_b(a)}[/]"
    return _fmt_b(p)


@app.command(rich_help_panel=_P_MODELS)
def search(
    query: str = typer.Argument(..., help="HF search query, or a base model id with --quants."),
    quants: bool = typer.Option(False, "--quants", "-q",
                                help="List quantizations of QUERY (a base model id) with a dtype-fit verdict."),
    limit: int = typer.Option(50, "--limit", help="Max results to scan (default 50)."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Search Hugging Face with a fit verdict for your hardware + capability badges.

    With --quants, QUERY is treated as a base model id and johnny enumerates its
    quantized variants, flagging which run natively on your GPUs (e.g. FP8 ✓ but
    NVFP4 ✗ on RDNA4) so you don't download a quant your silicon can't accelerate.
    """
    from .discover import search as dsearch
    from .hardware import detect as hwdetect

    hw = hwdetect.detect()

    if quants:
        res = dsearch.list_quantizations(query, hw, limit=limit)
        if res.get("error"):
            err.print(f"[red]{res['error']}[/]")
            raise typer.Exit(code=1)
        if json_output:
            console.print(_json.dumps(res, indent=2))
            return
        t = Table(title=f"quantizations of {query}  ·  native dtypes: {', '.join(hw.native_dtypes) or '—'}",
                  title_style="bold")
        for col in ("MODEL", "QUANT", "PARAMS", "DTYPE", "SIZE", "FIT"):
            t.add_column(col)
        for r in res["results"]:
            v = r["fit"]
            verdict = f"[{_VERDICT_STYLE.get(v['verdict'], 'white')}]{v['verdict']}[/]"
            label = _model_cell(r) + ("  [dim](base)[/]" if r.get("base") else "")
            t.add_row(label,
                      str(r.get("quant") or "—"), _params_cell(r), _dtype_cell(r["dtype"]),
                      f"{r['size_gb']}GB" if r["size_gb"] else "—",
                      f"{verdict} [dim]{v.get('detail', '')}[/]")
        console.print(t)
        console.print("[dim]✓ = compute dtype natively accelerated here · ✗ = runs un-accelerated or won't load[/]")
        return

    res = dsearch.search(query, hw, limit=limit)
    if res.get("error"):
        err.print(f"[red]{res['error']}[/]")
        raise typer.Exit(code=1)
    if json_output:
        console.print(_json.dumps(res, indent=2))
        return
    t = Table(title=f"HF search: {query}", title_style="bold")
    for col in ("MODEL", "DOWNLOADS", "PARAMS", "SIZE", "DTYPE", "FIT", "BADGES"):
        t.add_column(col)
    for r in res["results"]:
        v = r["fit"]
        verdict = f"[{_VERDICT_STYLE.get(v['verdict'], 'white')}]{v['verdict']}[/]"
        t.add_row(_model_cell(r), _fmt_count(r.get("downloads")),
                  _params_cell(r),
                  f"{r['size_gb']}GB" if r["size_gb"] else "—", _dtype_cell(r.get("dtype")),
                  f"{verdict} {v.get('detail', '')}", ", ".join(r["badges"]) or "—")
    console.print(t)
    legend = []
    if any(r.get("inducted") for r in res["results"]):
        legend.append("✓ = inducted in the registry (`johnny registry show`)")
    if any(r.get("gated") for r in res["results"]):
        legend.append("🔒 = gated (accept the license on huggingface.co + `johnny login`)")
    if legend:
        console.print(f"[dim]{' · '.join(legend)}[/]")


@app.command(rich_help_panel=_P_MODELS)
def download(
    repo: str = typer.Argument(..., help="Hugging Face repo id to download."),
    quant: str = typer.Option(None, "--quant", "-q", help="GGUF quant/variant to download (e.g. UD-Q4_K_XL); default: best fit for this hardware."),
    include: list[str] = typer.Option(None, "--include", help="Only download files matching this glob (repeatable; overrides --quant)."),
    all_files: bool = typer.Option(False, "--all", help="Download the entire repo, every quant included."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be downloaded and stop."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Download a model into the models dir (gated models need `johnny login`).

    Multi-quant GGUF repos download a single variant — the best fit for this
    hardware, or the one named with --quant — never the whole multi-TB repo
    unless --all is given. Refuses downloads that won't fit on disk."""
    from .discover import search as dsearch
    from .hardware import detect as hwdetect

    cfg = C.load_yaml(C.get_paths().config_file) or {}
    models_dir = (cfg.get("roots") or {}).get("models_dir")
    if not models_dir:
        err.print("[red]no models_dir in config[/] — run `johnny init`.")
        raise typer.Exit(code=1)
    kw = dict(variant=quant, include=list(include) if include else None,
              all_files=all_files, hardware=hwdetect.detect())
    plan = dsearch.acquire(repo, models_dir, dry_run=True, **kw)
    if plan.get("error"):
        err.print(f"[red]{plan['error']}[/]")
        raise typer.Exit(code=1)
    what = f"{repo} [bold]{plan['variant']}[/]" if plan.get("variant") else repo
    console.print(f"[dim]downloading {what} → {models_dir}/{repo} — "
                  f"~{plan['download_gb']} GB to fetch ({plan['files']} files), {plan['free_gb']} GB free (^C to abort)[/]")
    if dry_run:
        console.print(_json.dumps(plan, indent=2) if json_output else "[yellow]dry run — nothing downloaded.[/]")
        return
    res = dsearch.acquire(repo, models_dir, **kw)
    if res.get("error"):
        err.print(f"[red]{res['error']}[/]")
        raise typer.Exit(code=1)
    console.print(_json.dumps(res, indent=2) if json_output else f"[green]✓ downloaded[/] {repo} → {res['path']}")


@app.command(rich_help_panel=_P_MODELS)
def login(
    token: str = typer.Option(None, "--token", help="HF token; omit to show status."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Store / check a Hugging Face token (for gated models)."""
    from .discover import auth

    if token:
        p = auth.save_token(token)
        console.print(_json.dumps({"saved": str(p)}, indent=2) if json_output else f"[green]✓[/] token saved to {p}")
        return
    present = auth.has_token()
    if json_output:
        console.print(_json.dumps({"token_present": present, "path": str(auth.token_path())}, indent=2))
    elif present:
        console.print(f"[green]✓[/] HF token present  [dim]({auth.token_path()})[/]")
    else:
        console.print("[yellow]no HF token[/] — set one with `johnny login --token <hf_...>` (needed for gated models).")


# --------------------------------------------------------------------------- chat TUI + provider (P6)
@app.command(rich_help_panel=_P_OBSERVE)
def alive(
    model: str = typer.Option(None, "--model", help="Target a specific model."),
    seat: str = typer.Option(None, "--seat", help="Target a specific seat."),
    role: str = typer.Option("orchestrator", "--role", help="Target seat by role (default)."),
    no_wait: bool = typer.Option(False, "--no-wait", help="Don't wait for a loading seat."),
    timeout: int = typer.Option(900, "--timeout", help="Seconds to wait for a loading seat (default 900)."),
    no_attach: bool = typer.Option(False, "--no-attach", help="Start detached (don't attach the tmux session)."),
    session: str = typer.Option(None, "--session", help="tmux session name (default from config)."),
    provider: str = typer.Option(None, "--provider", help="Chat provider name (default: config [external].provider)."),
) -> None:
    """Launch (or re-attach) the chat TUI against a seat (role/model/seat)."""
    import os

    from .external import tui

    target = seat or model
    res = tui.alive(target=target, role=role, wait=not no_wait, timeout=timeout,
                    attach=not no_attach, session=session, provider=provider)
    if res.get("error"):
        err.print(f"[red]{res['error']}[/]")
        raise typer.Exit(code=1)
    console.print(f"[green]●[/] {res['action']} session [bold]{res['session']}[/] · seat={res['seat']} · model={res['model']}")
    if res["action"] == "attach":
        os.execvp("tmux", ["tmux", "attach", "-t", res["session"]])
    else:
        console.print(f"  [dim]attach with: tmux attach -t {res['session']}[/]")


# --------------------------------------------------------------------------- profiles (named fleets)
profile_app = typer.Typer(add_completion=False,
                          help="Named fleets: capture, bring up/down, auto-start at boot.")
app.add_typer(profile_app, name="profile", rich_help_panel=_P_FLEET)


def _parse_role_flags(role_flags: list[str]) -> dict:
    roles = {}
    for r in role_flags or []:
        if "=" not in r:
            raise typer.BadParameter(f"--role expects <model>=<role>, got '{r}'")
        k, v = r.split("=", 1)
        roles[k.strip()] = v.strip()
    return roles


def _render_profile_seats(seats: list[dict]) -> None:
    t = Table(pad_edge=False)
    for col in ("ROLE", "MODEL", "PLACEMENT", "PORT", "PINNED"):
        t.add_column(col, no_wrap=(col != "PLACEMENT"))
    for s in seats:
        t.add_row(str(s.get("role") or "—"), str(s.get("model")), str(s.get("placement") or "—"),
                  str(s.get("port") or "—"), "✓" if s.get("pinned") else "")
    console.print(t)


def _render_profile_results(res: dict) -> bool:
    """Per-seat outcome table for profile up/down. Returns True if any seat errored."""
    t = Table(title=f"profile {res['profile']}", title_style="bold", pad_edge=False)
    for col in ("ROLE", "MODEL", "ACTION", "SEAT", "PORT", "GPUS", "STATE"):
        t.add_column(col, no_wrap=True)
    failed = False
    for s in res.get("seats") or []:
        action = s.get("action")
        if action == "error":
            failed = True
        style = {"error": "red", "exists": "dim", "launched": "green", "down": "yellow"}.get(action, "white")
        t.add_row(str(s.get("role") or "—"), str(s.get("model")),
                  f"[{style}]{action}[/]", str(s.get("seat") or "—"), str(s.get("port") or "—"),
                  " ".join(str(g) for g in s.get("gpus") or []) or "—",
                  str(s.get("state") or ("" if action != "error" else "see below") or "—"))
    console.print(t)
    for s in res.get("seats") or []:
        if s.get("action") == "error":
            console.print(f"  [red]✗ {s.get('model')}[/]: {s.get('error')}")
    return failed


def _report_validation(errors: list[str], warnings: list[str]) -> None:
    for e in errors:
        console.print(f"  [red]✗ {e}[/]")
    for w in warnings:
        console.print(f"  [yellow]⚠ {w}[/]")


@profile_app.command("save")
def profile_save(
    name: str = typer.Argument(..., help="Profile name (e.g. 'standard')."),
    role: list[str] = typer.Option(None, "--role", help="Role for a captured seat, as <model>=<role> "
                                   "(repeatable). Roles are what SAINT resolves (johnny_role); "
                                   "pooling seats are auto-inferred as 'embed'."),
    description: str = typer.Option(None, "--description", help="Human note stored on the profile."),
    no_pins: bool = typer.Option(False, "--no-pins", help="Don't pin the seats (default: pin all — kept warm, reaper-exempt)."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing profile of the same name."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Capture the currently-running johnny fleet as a named profile.

    Records each live seat's model + placement + port (+ role), so `profile up`
    can re-create the whole fleet — e.g. after a reboot. Every seat needs a role;
    supply non-inferable ones with --role.
    """
    from .engine import profiles
    from .hardware import detect as hwd
    from .registry import store

    if profiles.get_profile(name) and not force:
        _emit_err(ValueError(f"profile '{name}' exists (use --force to overwrite)"), json_output)
    cap = profiles.capture(roles=_parse_role_flags(role))
    seats = cap["seats"]
    if not seats:
        _emit_err(ValueError("no johnny-managed seats running — bring the fleet up first"), json_output)
    missing = [s["model"] for s in seats if not s.get("role")]
    if missing:
        _emit_err(ValueError("every seat needs a role; add "
                             + " ".join(f"--role {m}=<role>" for m in missing)), json_output)
    if not no_pins:
        for s in seats:
            s["pinned"] = True
    prof: dict = {"seats": seats}
    if description:
        prof["description"] = description

    errors, warnings = profiles.validate(prof, store.load(), hwd.detect(), name=name)
    if errors:
        if json_output:
            console.print(_json.dumps({"error": "validation failed", "errors": errors}, indent=2))
            raise typer.Exit(code=1)
        console.print("[red]validation failed[/] — profile not saved:")
        _report_validation(errors, warnings)
        raise typer.Exit(code=1)

    profiles.save(name, prof)
    if json_output:
        console.print(_json.dumps({"profile": name, "seats": seats, "skipped": cap["skipped"],
                                   "warnings": warnings}, indent=2))
        return
    console.print(f"[green]✓ saved[/] profile [bold]{name}[/] ({len(seats)} seat(s)) "
                  f"→ {C.get_paths().profiles_file}")
    _render_profile_seats(seats)
    _report_validation([], warnings)
    if cap["skipped"]:
        console.print(f"[yellow]⚠ not captured[/] (no johnny labels — not johnny-managed): "
                      f"{', '.join(cap['skipped'])}")
        console.print("  [dim]adopt with `johnny up <model> --placement <id>` after stopping the original, then re-save.[/]")


@profile_app.command("list")
def profile_list(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """List saved profiles."""
    from .engine import profiles

    profs = profiles.all_profiles()
    if json_output:
        console.print(_json.dumps(profs, indent=2))
        return
    if not profs:
        console.print("[dim]no profiles — capture your running fleet with `johnny profile save <name>`[/]")
        return
    t = Table(pad_edge=False)
    for col in ("PROFILE", "SEATS", "ROLES", "PINNED", "DESCRIPTION"):
        t.add_column(col, no_wrap=(col != "DESCRIPTION"))
    for name, p in profs.items():
        seats = p.get("seats") or []
        t.add_row(name, str(len(seats)),
                  " ".join(str(s.get("role")) for s in seats if s.get("role")) or "—",
                  str(sum(1 for s in seats if s.get("pinned"))),
                  p.get("description") or "—")
    console.print(t)


@profile_app.command("show")
def profile_show(
    name: str = typer.Argument(..., help="Profile name."),
    saint: bool = typer.Option(False, "--saint", help="Print SAINT [backends.*] stanza suggestions for these seats."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show a profile's seats, cross-checked against the registry."""
    from .engine import profiles
    from .hardware import detect as hwd
    from .registry import store

    prof = profiles.get_profile(name)
    if prof is None:
        _emit_err(ValueError(f"no profile '{name}'"), json_output)
    errors, warnings = profiles.validate(prof, store.load(), hwd.detect(), name=name)
    if json_output:
        console.print(_json.dumps({"profile": name, **prof, "errors": errors, "warnings": warnings}, indent=2))
        return
    console.print(f"[bold]{name}[/]" + (f"  [dim]{prof.get('description')}[/]" if prof.get("description") else ""))
    _render_profile_seats(prof.get("seats") or [])
    _report_validation(errors, warnings)
    if saint:
        console.print("\n[dim]# SAINT config.toml suggestions — static fallbacks matching the profile ports.")
        console.print("# The live path stays `johnny resolve <role>` via each backend's johnny_role.[/]")
        for s in prof.get("seats") or []:
            console.print(f"""
\\[backends.local-{s.get('role')}]
provider     = "openai"
base_url     = "http://localhost:{s.get('port')}/v1"   # static fallback — johnny's live seat overrides
model        = "{s.get('model')}"
api_key      = "local"
johnny_role  = "{s.get('role')}\"""")


@profile_app.command("rm")
def profile_rm(
    name: str = typer.Argument(..., help="Profile name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Remove a profile (does not stop any running seats)."""
    from .engine import profiles

    if profiles.get_profile(name) is None:
        _emit_err(ValueError(f"no profile '{name}'"), json_output)
    if not yes and not json_output:
        if not typer.confirm(f"remove profile '{name}'?"):
            raise typer.Exit()
    profiles.remove(name)
    if json_output:
        console.print(_json.dumps({"removed": name}, indent=2))
        return
    console.print(f"[green]✓ removed[/] profile [bold]{name}[/]")


@profile_app.command("up")
def profile_up(
    name: str = typer.Argument(..., help="Profile name."),
    wait: bool = typer.Option(False, "--wait", help="Block until each seat is serving (serial)."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Bring up every seat in a profile (idempotent: running seats are skipped).

    Best-effort — one seat's failure doesn't block the rest. Pinned seats are
    pinned (kept warm / reaper-exempt) even if they were already running.
    """
    from .engine import launch, profiles

    try:
        res = profiles.up_profile(name, wait=wait)
    except launch.PlacementError as e:
        _emit_err(e, json_output)
    if json_output:
        console.print(_json.dumps(res, indent=2))
        if any(s.get("action") == "error" for s in res.get("seats") or []):
            raise typer.Exit(code=1)
        return
    if _render_profile_results(res):
        raise typer.Exit(code=1)


@profile_app.command("down")
def profile_down(
    name: str = typer.Argument(..., help="Profile name."),
    drain: bool = typer.Option(False, "--drain", help="Drain before stopping (no-op for vLLM today)."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Stop every running seat of a profile (clears their pins)."""
    from .engine import launch, profiles

    try:
        res = profiles.down_profile(name, drain=drain)
    except launch.PlacementError as e:
        _emit_err(e, json_output)
    if json_output:
        console.print(_json.dumps(res, indent=2))
        return
    _render_profile_results(res)


@profile_app.command("enable")
def profile_enable(
    name: str = typer.Argument(..., help="Profile name."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the unit + commands without writing/enabling."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Auto-start this profile at boot (user systemd unit).

    Writes ~/.config/systemd/user/johnny-profile@.service and enables this
    profile's instance. True at-boot start (no login) needs lingering:
    `loginctl enable-linger $USER`.
    """
    from . import boot
    from .engine import profiles
    from .util import run as _run

    if profiles.get_profile(name) is None:
        _emit_err(ValueError(f"no profile '{name}' (see `johnny profile list`)"), json_output)
    text = boot.unit_text()
    cmds = boot.enable_commands(name)
    if dry_run:
        console.print(f"[dim]# would write {boot.unit_path()}:[/]\n{text}")
        for c in cmds:
            console.print(f"[dim]# would run:[/] {' '.join(c)}")
        return
    boot.unit_dir().mkdir(parents=True, exist_ok=True)
    boot.unit_path().write_text(text)
    ran = []
    for c in cmds:
        rc, out, errout = _run(c, timeout=1800)
        ran.append({"cmd": " ".join(c), "rc": rc})
        if rc != 0:
            _emit_err(RuntimeError(f"{' '.join(c)} failed: {errout.strip() or out.strip()}"), json_output)
    if json_output:
        console.print(_json.dumps({"enabled": boot.instance(name), "unit": str(boot.unit_path()), "ran": ran}, indent=2))
        return
    console.print(f"[green]✓ enabled[/] [bold]{boot.instance(name)}[/] — starts at boot (unit: {boot.unit_path()})")
    console.print("  [dim]at-boot start needs lingering: loginctl enable-linger $USER[/]")


@profile_app.command("disable")
def profile_disable(
    name: str = typer.Argument(..., help="Profile name."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Stop auto-starting this profile at boot.

    Running seats are left alone (plain `systemctl --user disable`, never
    --now — the unit's ExecStop would down the fleet). Use `johnny profile
    down` to actually stop seats.
    """
    from . import boot
    from .util import run as _run

    ran = []
    for c in boot.disable_commands(name):
        rc, out, errout = _run(c, timeout=60)
        ran.append({"cmd": " ".join(c), "rc": rc})
        if rc != 0:
            _emit_err(RuntimeError(f"{' '.join(c)} failed: {errout.strip() or out.strip()}"), json_output)
    if json_output:
        console.print(_json.dumps({"disabled": boot.instance(name), "ran": ran}, indent=2))
        return
    console.print(f"[green]✓ disabled[/] [bold]{boot.instance(name)}[/] — no longer auto-starts (seats left running)")


provider_app = typer.Typer(add_completion=False, help="Sync an external chat tool's provider config.")
app.add_typer(provider_app, name="provider", rich_help_panel=_P_FLEET)


@provider_app.command("sync")
def provider_sync(
    write: bool = typer.Option(False, "--write", help="Patch the config in place (timestamped backup)."),
    provider: str = typer.Option(None, "--provider", help="Chat provider name (default: config [external].provider)."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Compute the provider's base_url + models catalog from the registry (preview, or --write)."""
    from .external import provider as prov

    res = prov.sync(provider_name=provider, write=write)
    if res.get("error"):
        err.print(f"[red]{res['error']}[/]")
        raise typer.Exit(code=1)
    if json_output:
        console.print(_json.dumps(res, indent=2))
        return
    b = res["block"]
    console.print(f"[bold]{b['name']}[/]  base_url={b['base_url']}  ({len(b['models'])} models)")
    for mid, meta in sorted(b["models"].items()):
        console.print(f"  {mid}: context_length={meta['context_length']}")
    if res["written"]:
        console.print(f"[green]✓ patched[/] {res['path']}  [dim](backup {res['backup']})[/]")
    else:
        console.print(f"[dim]preview only — pass --write to patch {res['path']} (creates a backup).[/]")


# --------------------------------------------------------------------------- lifecycle / cleanup (P8)
@app.command(rich_help_panel=_P_SETUP)
def cleanup(
    apply: bool = typer.Option(False, "--apply", help="Actually delete (default: dry-run preview)."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Surface removal candidates (untracked on disk / unvalidated-here / stale)."""
    from . import lifecycle

    res = lifecycle.cleanup_candidates()
    cands = res["candidates"]
    if json_output:
        console.print(_json.dumps(res, indent=2))
        return
    if not cands:
        console.print(f"[green]nothing to clean up[/]  [dim](fingerprint {res['fingerprint']})[/]")
        return
    t = Table(title="cleanup candidates", title_style="bold")
    for col in ("KIND", "TARGET", "SIZE", "REASON"):
        t.add_column(col)
    style = {"untracked": "yellow", "unvalidated": "cyan", "stale": "dim"}
    for c in cands:
        t.add_row(f"[{style.get(c['kind'], 'white')}]{c['kind']}[/]", c["target"],
                  f"{c['size_gb']}GB" if c.get("size_gb") else "—", c["reason"])
    console.print(t)
    if apply:
        untracked = [c for c in cands if c["kind"] == "untracked"]
        if not untracked:
            console.print("[dim]nothing deletable here (only untracked on-disk dirs); use `johnny rm <model>` for a tracked one.[/]")
            return
        console.print("[dim]confirm each (Ctrl-C to stop):[/]")
        for c in untracked:
            if typer.confirm(f"Delete {c['target']} ({c.get('size_gb')}GB)?", default=False):
                ok = lifecycle.delete_path(c["path"])
                console.print(f"  {'[green]✓ deleted[/]' if ok else '[red]✗ failed[/]'} {c['target']}")
            else:
                console.print(f"  [dim]skipped {c['target']}[/]")
    else:
        console.print("[dim]dry-run — `cleanup --apply` confirms each, or `johnny rm <model>` removes a single one.[/]")


@app.command(name="rm", rich_help_panel=_P_SETUP)
def rm(
    target: str = typer.Argument(..., help="Model id, local path (vendor/name), or directory."),
    registry_only: bool = typer.Option(False, "--registry-only", help="Deregister but keep the weights on disk."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Remove a single model: its on-disk weights and/or its registry entry."""
    from . import lifecycle

    info = lifecycle.resolve_target(target)
    if not info:
        err.print(f"[red]no model[/] '{target}' on disk or in the registry")
        raise typer.Exit(code=1)
    seat = lifecycle.running_seat_for(info)
    if seat:
        err.print(f"[red]'{target}' is serving as seat[/] {seat} — stop it first: `johnny down {seat}`")
        raise typer.Exit(code=1)

    actions = []
    if info.get("model_id"):
        actions.append("deregister from registry")
    if not registry_only and info.get("path"):
        actions.append(f"delete {info['path']} ({info.get('size_gb')}GB)")
    if not actions:
        console.print(f"[yellow]nothing to do[/] for '{target}' (no registry entry and no on-disk path).")
        return

    if not yes and not json_output:
        console.print("will: " + "; ".join(actions))
        if not typer.confirm("Proceed?", default=False):
            raise typer.Exit(code=1)
    res = lifecycle.remove(info, registry_only=registry_only)
    if json_output:
        console.print(_json.dumps(res, indent=2))
        return
    if res["deleted_path"]:
        console.print(f"[green]✓ deleted[/] {res['deleted_path']}")
    if res["deregistered"]:
        console.print(f"[green]✓ deregistered[/] {res['model_id']}")
    if not res["deleted_path"] and not res["deregistered"]:
        console.print("[yellow]nothing removed[/]")


# --------------------------------------------------------------------------- daemon / request plane (P10)
daemon_app = typer.Typer(add_completion=False, help="johnnyd: request-plane API + JIT gateway.")
app.add_typer(daemon_app, name="daemon", rich_help_panel=_P_FLEET)


def _daemon_pidfile(agent: bool = False):
    return C.get_paths().state_dir / ("johnnyd-agent.json" if agent else "johnnyd.json")


@daemon_app.command("up")
def daemon_up(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address for johnnyd (default 127.0.0.1)."),
    port: int = typer.Option(8080, "--port", help="Port for johnnyd to listen on (default 8080)."),
    no_jit: bool = typer.Option(False, "--no-jit", help="Disable load-on-first-request."),
    max_concurrent: int = typer.Option(0, "--max-concurrent", help="Per-seat admission cap (0=unlimited)."),
    agent: bool = typer.Option(False, "--agent", help="Run as a node agent (dial out to a controller)."),
    controller: str = typer.Option(None, "--controller", help="Controller URL (agent mode)."),
    token: str = typer.Option("", "--token", help="Cluster join token (agent mode)."),
    foreground: bool = typer.Option(False, "--foreground", help="Run in this process (don't detach)."),
) -> None:
    """Start johnnyd: controller (request-plane API + JIT gateway) or --agent (node)."""
    import subprocess
    import sys

    if agent:
        if not controller:
            err.print("[red]--agent requires --controller <url>[/]")
            raise typer.Exit(code=1)
        if foreground:
            from .cluster.agent import run_agent

            run_agent(controller, token=token)
            return
        args = [sys.executable, "-m", "johnny", "daemon", "up", "--agent", "--foreground", "--controller", controller]
        if token:
            args += ["--token", token]
        p = subprocess.Popen(args, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pf = _daemon_pidfile(agent=True)
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(_json.dumps({"pid": p.pid, "controller": controller}))
        console.print(f"[green]●[/] johnnyd agent started · pid {p.pid} · → {controller}")
        return

    if foreground:
        from .daemon.server import serve

        serve(host=host, port=port, jit=not no_jit, max_concurrent=max_concurrent)
        return
    args = [sys.executable, "-m", "johnny", "daemon", "up", "--foreground", "--host", host, "--port", str(port)]
    if no_jit:
        args.append("--no-jit")
    if max_concurrent:
        args += ["--max-concurrent", str(max_concurrent)]
    p = subprocess.Popen(args, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pf = _daemon_pidfile()
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(_json.dumps({"pid": p.pid, "host": host, "port": port}))
    console.print(f"[green]●[/] johnnyd started · pid {p.pid} · http://{host}:{port}  "
                  f"[dim](jit={'off' if no_jit else 'on'})[/]")
    console.print(f"  try: curl http://{host}:{port}/v1/fleet  ·  stop: johnny daemon down")


@daemon_app.command("status")
def daemon_status(json_output: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
    """Is johnnyd running + healthy?"""
    import urllib.request

    pf = _daemon_pidfile()
    if not pf.exists():
        console.print("[dim]johnnyd not running (no pidfile).[/]")
        raise typer.Exit(code=1)
    info = _json.loads(pf.read_text())
    healthy = False
    try:
        with urllib.request.urlopen(f"http://{info['host']}:{info['port']}/healthz", timeout=2) as r:
            healthy = _json.loads(r.read()).get("ok", False)
    except Exception:
        healthy = False
    if json_output:
        console.print(_json.dumps({**info, "healthy": healthy}, indent=2))
        return
    console.print(f"{'[green]● healthy[/]' if healthy else '[red]○ unreachable[/]'} "
                  f"johnnyd pid {info['pid']} · http://{info['host']}:{info['port']}")


@daemon_app.command("down")
def daemon_down() -> None:
    """Stop johnnyd (controller and/or agent)."""
    import os
    import signal

    stopped = False
    for agent in (False, True):
        pf = _daemon_pidfile(agent=agent)
        if not pf.exists():
            continue
        info = _json.loads(pf.read_text())
        try:
            os.kill(info["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        pf.unlink(missing_ok=True)
        console.print(f"[green]✓[/] johnnyd {'agent' if agent else 'controller'} stopped (pid {info['pid']})")
        stopped = True
    if not stopped:
        console.print("[dim]johnnyd not running.[/]")


@app.command(rich_help_panel=_P_FLEET)
def nodes(
    controller: str = typer.Option("http://127.0.0.1:8080", "--controller", help="Controller base URL (default http://127.0.0.1:8080)."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """List nodes registered with the controller (multi-machine fleet)."""
    import urllib.request

    try:
        with urllib.request.urlopen(controller.rstrip("/") + "/cluster/nodes", timeout=5) as r:
            data = _json.loads(r.read())
    except Exception as e:
        err.print(f"[red]controller unreachable[/] at {controller}: {e}")
        raise typer.Exit(code=1)
    nodes_list = data.get("nodes", [])
    if json_output:
        console.print(_json.dumps(nodes_list, indent=2))
        return
    if not nodes_list:
        console.print("[dim]no nodes registered.[/] Start an agent: `johnny daemon up --agent --controller <url>`")
        return
    t = Table(title="cluster nodes", title_style="bold")
    for col in ("NODE", "FINGERPRINT", "GPUS", "SEATS", "STATUS"):
        t.add_column(col)
    for n in nodes_list:
        hw = n.get("hardware", {})
        st = n.get("status", "?")
        t.add_row(n.get("node_id", "?"), hw.get("fingerprint", "—"), str(hw.get("gpus", "—")),
                  str(len(n.get("seats", []))), f"[{'green' if st == 'ready' else 'red'}]{st}[/]")
    console.print(t)


# --------------------------------------------------------------------------- TUI (P9)
@app.command(rich_help_panel=_P_OBSERVE)
def tui() -> None:
    """Launch the live Textual dashboard (seats, concurrency, KV — by backend/model)."""
    from .tui.app import run as run_tui

    run_tui()


# --------------------------------------------------------------------------- future stubs
_FUTURE: dict[str, str] = {}


def _make_stub(name: str, phase: str):
    def _cmd():
        err.print(f"[yellow]🚧 `johnny {name}` isn't implemented yet — lands at {phase}.[/] "
                  f"(See PLAN.md §4.)")
        raise typer.Exit(code=1)

    _cmd.__name__ = f"stub_{name}"
    _cmd.__doc__ = f"(stub) lands at {phase}."
    return _cmd


for _name, _phase in _FUTURE.items():
    app.command(name=_name, hidden=True)(_make_stub(_name, _phase))


# --------------------------------------------------------------------------- help order
# Panels otherwise render in command-definition order; force the task-priority order
# (Seats first, Setup last). Stable sort preserves within-panel order. Touches only
# --help rendering / command listing, never dispatch (which is by name).
_PANEL_ORDER = [_P_SEATS, _P_OBSERVE, _P_MODELS, _P_FLEET, _P_SETUP]


def _panel_rank(info) -> int:
    panel = getattr(info, "rich_help_panel", None)
    return _PANEL_ORDER.index(panel) if panel in _PANEL_ORDER else len(_PANEL_ORDER)


app.registered_commands.sort(key=_panel_rank)
app.registered_groups.sort(key=_panel_rank)


# --------------------------------------------------------------------------- root
@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """Bare `johnny` shows status (the old default)."""
    if ctx.invoked_subcommand is None:
        _render_status(json_output=False)
        raise typer.Exit()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
