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
import time

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


# --------------------------------------------------------------------------- status
def _seat_image(s) -> str:
    return (s.extra or {}).get("image") or "—"


def _seats_as_dicts(seats) -> list[dict]:
    return [
        {"seat": s.name, "backend": s.backend, "port": s.port, "model": s.model,
         "state": s.state, "gpus": s.gpus, "image": _seat_image(s)}
        for s in seats
    ]


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
    table = Table(title="johnny — seats", title_style="bold")
    for col, style in (("SEAT", "bold"), ("BACKEND", "dim"), ("PORT", None),
                       ("MODEL", "cyan"), ("STATE", None), ("GPUS", None), ("IMAGE", "dim")):
        table.add_column(col, style=style)
    for s in seats:
        gpus = ",".join(map(str, s.gpus)) if s.gpus else "—"
        table.add_row(s.name, s.backend, str(s.port or "—"), s.model or "—",
                      f"[{_STATE_STYLE.get(s.state, 'white')}]{s.state}[/]", gpus, _seat_image(s))
    console.print(table)


@app.command()
def status(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
    watch: bool = typer.Option(False, "--watch", help="Refresh live (basic; full TUI is P9)."),
) -> None:
    """Show running inference seats (docker + endpoint probe)."""
    if watch and not json_output:
        try:
            from rich.live import Live

            with Live(console=console, refresh_per_second=4, screen=True) as live:
                while True:
                    table = _build_status_renderable()
                    live.update(table)
                    time.sleep(2)
        except KeyboardInterrupt:
            return
    else:
        _render_status(json_output=json_output)


def _build_status_renderable():
    """A Rich renderable of current seats (used by --watch)."""
    from . import engine

    table = Table(title="johnny — seats (live)", title_style="bold")
    for col, style in (("SEAT", "bold"), ("BACKEND", "dim"), ("PORT", None),
                       ("MODEL", "cyan"), ("STATE", None), ("GPUS", None), ("IMAGE", "dim")):
        table.add_column(col, style=style)
    for s in engine.all_seats():
        gpus = ",".join(map(str, s.gpus)) if s.gpus else "—"
        table.add_row(s.name, s.backend, str(s.port or "—"), s.model or "—",
                      f"[{_STATE_STYLE.get(s.state, 'white')}]{s.state}[/]", gpus, _seat_image(s))
    return table


# --------------------------------------------------------------------------- doctor
@app.command()
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
@app.command()
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
@app.command()
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
@app.command()
def version(json_output: bool = typer.Option(False, "--json")) -> None:
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
@app.command()
def gpu(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
    refresh: bool = typer.Option(False, "--refresh", help="Re-run the dtype ISA probe (ignore cache)."),
) -> None:
    """Detect GPUs: vendor, count, per-GPU VRAM, arch, and natively-accelerated dtypes."""
    from dataclasses import asdict

    from .hardware import detect as hwdetect

    hw = hwdetect.detect(refresh=refresh)
    if json_output:
        console.print(_json.dumps(asdict(hw), indent=2))
        return

    if not hw.gpus:
        console.print(
            f"[yellow]no GPUs detected[/] (vendor={hw.vendor or 'none'}); "
            f"host RAM {hw.host_ram_gb:.0f} GB — CPU / LM Studio / Ollama only."
        )
        return

    het = "" if hw.homogeneous else "  [yellow](heterogeneous)[/]"
    console.print(
        f"[bold]{hw.vendor}[/] · {len(hw.gpus)} GPU(s) · {hw.total_vram_gb:.0f} GB VRAM · "
        f"host RAM {hw.host_ram_gb:.0f} GB · fingerprint [cyan]{hw.fingerprint}[/]{het}"
    )
    for g in hw.groups:
        dl = ", ".join(g.native_dtypes) or "—"
        console.print(
            f"  [bold]{g.arch}[/] ×{g.count} @ {g.vram_gb:.0f}GB — native dtypes: "
            f"[green]{dl}[/] [dim](source: {hw.dtype_source})[/]"
        )

    table = Table(title="GPUs", title_style="bold")
    table.add_column("IDX")
    table.add_column("NAME")
    table.add_column("ARCH", style="cyan")
    table.add_column("VRAM (GB)", justify="right")
    for g in hw.gpus:
        table.add_row(str(g.index), g.name, g.arch, f"{g.vram_gb:.0f}")
    console.print(table)

    nd = set(hw.native_dtypes)
    fp8 = "[green]✓[/]" if "fp8" in nd else "[red]✗[/]"
    fp4 = "[green]✓[/]" if "fp4" in nd else "[red]✗[/]"
    console.print(f"  fp8 native: {fp8}    fp4 native: {fp4}")


# --------------------------------------------------------------------------- registry
registry_app = typer.Typer(add_completion=False, help="Inspect / seed / validate the model registry.")
app.add_typer(registry_app, name="registry")


@registry_app.command("show")
def registry_show(
    model: str = typer.Argument(None, help="Model id to detail; omit to list all."),
    compact: bool = typer.Option(False, "--compact", "-c", help="Terse one-row-per-model index (omit placements)."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List registry models with their placements (or detail one model).

    Placements are what you load (`up --placement <id>`) and prune
    (`registry delete <model> <id>`), so they're shown by default; -c for the index.
    """
    from .registry import store

    reg = store.load()
    models = store.models(reg)
    if model:
        m = models.get(model)
        if not m:
            err.print(f"[red]no model[/] '{model}' in the registry")
            raise typer.Exit(code=1)
        if json_output:
            console.print(_json.dumps(m, indent=2))
            return
        ident = m.get("identity", {})
        console.print(f"[bold]{model}[/]  [dim]{ident.get('local_path') or ident.get('repo_id')}[/]")
        console.print(f"  arch={ident.get('arch')} quant={ident.get('quant')} ctx={m.get('capabilities',{}).get('native_context')}")
        t = Table(title="placements")
        for col in ("ID", "BACKEND", "USE", "TP", "MML", "GMU", "MTP", "KV", "SOURCE", "FINGERPRINT"):
            t.add_column(col)
        for p in m.get("placements", []):
            k = p.get("knobs", {})
            mtp = "on" if (k.get("mtp") or {}).get("enabled") else "—"
            t.add_row(p.get("id", ""), p.get("backend", ""), str(p.get("use_case") or "—"),
                      str(k.get("tensor_parallel_size") or "—"), str(k.get("max_model_len") or "—"),
                      str(k.get("gpu_memory_util") or "—"), mtp, str(k.get("kv_cache_dtype") or "—"),
                      p.get("source", ""), p.get("validation_key", {}).get("hardware_fingerprint", "—"))
        console.print(t)
        return

    if json_output:
        console.print(_json.dumps(reg, indent=2))
        return
    if not models:
        console.print("[dim]registry is empty.[/] Seed it with [bold]johnny registry import[/].")
        return
    if compact:
        t = Table(title=f"registry — {len(models)} model(s)", title_style="bold")
        for col in ("MODEL", "ARCH", "QUANT", "CTX", "#PLACEMENTS", "BACKENDS"):
            t.add_column(col)
        for mid, m in sorted(models.items()):
            ident = m.get("identity", {})
            pls = m.get("placements", [])
            backends = sorted({p.get("backend", "?") for p in pls})
            t.add_row(mid, str(ident.get("arch") or "—"), str(ident.get("quant") or "—"),
                      str(m.get("capabilities", {}).get("native_context") or "—"), str(len(pls)), ", ".join(backends))
        console.print(t)
        return

    # Default: full picture — every model with its placements inline.
    console.print(f"[bold]registry — {len(models)} model(s)[/]  "
                  f"[dim]load: `johnny up <model> --placement <id>`  ·  -c for the terse index[/]")
    for mid, m in sorted(models.items()):
        ident = m.get("identity", {})
        pls = m.get("placements", [])
        cap = m.get("capabilities", {})
        meta = " · ".join(x for x in [ident.get("arch"), ident.get("quant"),
                          f"ctx {cap.get('native_context')}" if cap.get("native_context") else None] if x)
        console.print(f"\n[bold]{mid}[/]  [dim]{meta}[/]")
        if not pls:
            console.print("  [dim](no placements — `johnny induct <model>` or `registry import`)[/]")
            continue
        for p in pls:
            k = p.get("knobs", {})
            perf = p.get("perf", {})
            peak, single = perf.get("peak_tok_s"), perf.get("single_stream_tok_s")
            perf_s = f" · {peak}/{single} tok/s" if peak else ""
            console.print(
                f"  • [cyan]{p.get('id')}[/]  [dim]tp{k.get('tensor_parallel_size') or '—'} · "
                f"{p.get('use_case') or '—'} · mml{k.get('max_model_len') or '—'}{perf_s} · {p.get('source')}[/]"
            )


@registry_app.command("import")
def registry_import(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be imported; write nothing."),
    json_output: bool = typer.Option(False, "--json"),
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
def registry_validate(json_output: bool = typer.Option(False, "--json")) -> None:
    """Validate the registry against the schema."""
    from .registry import schema, store

    errors = schema.validate(store.load())
    if json_output:
        console.print(_json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return
    if not errors:
        console.print("[green]✓ registry is valid[/]")
        return
    for e in errors:
        err.print(f"[red]✗[/] {e}")
    raise typer.Exit(code=1)


@registry_app.command("delete")
def registry_delete(
    model: str = typer.Argument(..., help="Model id (see `johnny registry show`)."),
    placement: str = typer.Argument(None, help="Placement id to delete — exact or a unique substring (e.g. 'tp2'). Omit with --all."),
    all_placements: bool = typer.Option(False, "--all", help="Delete ALL placements for the model (keeps the model entry)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Delete a placement (or all) from a model — placement-level pruning.

    Keeps the model and its other placements. To remove a whole model (and optionally
    its weights), use `johnny rm <model>`.
    """
    from .registry import store

    reg = store.load()
    m = store.get(reg, model)
    if not m:
        err.print(f"[red]no model[/] '{model}' in the registry")
        raise typer.Exit(code=1)
    pls = m.get("placements") or []
    if not pls:
        console.print(f"[yellow]'{model}' has no placements.[/]")
        return

    if all_placements:
        targets = list(pls)
    elif placement:
        exact = [p for p in pls if p.get("id") == placement]
        subs = [p for p in pls if placement in (p.get("id") or "")]
        targets = exact or subs
        if not targets:
            err.print(f"[red]no placement[/] matching '{placement}' in '{model}'.")
            console.print("  known: " + ", ".join(p.get("id", "") for p in pls))
            raise typer.Exit(code=1)
        if len(targets) > 1:
            err.print(f"[red]'{placement}' is ambiguous[/] — matches {len(targets)}:")
            for p in targets:
                console.print(f"  • {p.get('id')}")
            raise typer.Exit(code=1)
    else:
        console.print(f"[yellow]specify a placement id to delete (or --all).[/] '{model}' has:")
        for p in pls:
            k = p.get("knobs", {})
            console.print(f"  • [cyan]{p.get('id')}[/]  [dim](tp{k.get('tensor_parallel_size') or '—'} · "
                          f"{p.get('use_case') or '—'} · {p.get('source')})[/]")
        raise typer.Exit(code=1)

    ids = [p.get("id", "") for p in targets]
    if not yes and not json_output:
        console.print(f"will delete from [bold]{model}[/]: " + ", ".join(ids))
        if not typer.confirm("Proceed?", default=False):
            raise typer.Exit(code=1)
    for pid in ids:
        store.delete_placement(reg, model, pid)
    store.save(reg)
    remaining = len(m.get("placements") or [])
    if json_output:
        console.print(_json.dumps({"model": model, "deleted": ids, "remaining": remaining}, indent=2))
    else:
        console.print(f"[green]✓ deleted[/] {len(ids)} placement(s) from {model}  [dim]({remaining} remaining)[/]")


# --------------------------------------------------------------------------- seat lifecycle (P3)
def _emit_err(e: Exception, json_output: bool):
    if json_output:
        console.print(_json.dumps({"error": str(e)}, indent=2))
    else:
        err.print(f"[red]{e}[/]")
    raise typer.Exit(code=1)


def _render_pick(it: dict) -> str:
    """One placement line for the picker: model + id + key knobs + perf."""
    p = it["p"]
    k = p.get("knobs", {})
    peak = (p.get("perf") or {}).get("peak_tok_s")
    perf_s = f" · {peak} tok/s" if peak else ""
    return (f"[bold]{it['model']}[/]  [cyan]{p.get('id')}[/]  "
            f"[dim]tp{k.get('tensor_parallel_size') or '—'} · {p.get('use_case') or '—'} · "
            f"mml{k.get('max_model_len') or '—'}{perf_s}[/]")


def _pick_placement_interactive(json_output: bool) -> tuple[str, str]:
    """Open the placement picker over the whole registry; return (model, placement_id)
    or exit (cancel / nothing to pick / --json with no model)."""
    from .external import picker
    from .registry import store

    if json_output:
        _emit_err(ValueError("`up` needs a model id with --json (the picker needs a TTY)"), True)
    models = store.models(store.load())
    items = [{"model": mid, "p": p}
             for mid, m in sorted(models.items())
             for p in (m.get("placements") or [])]
    if not items:
        err.print("[yellow]no placements in the registry[/] — run `johnny induct <model>` first.")
        raise typer.Exit(code=1)
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


@app.command()
def up(
    model: str = typer.Argument(None, help="Registry model id. Omit to pick a placement interactively."),
    placement: str = typer.Option(None, "--placement", help="Placement id or unique substring (e.g. 'tp4'); else best fit for this hardware."),
    port: int = typer.Option(None, "--port"),
    swap: str = typer.Option(None, "--swap", help="Seat to evict to free its GPUs/port."),
    force: bool = typer.Option(False, "--force", help="Place even if GPUs are busy."),
    wait: bool = typer.Option(False, "--wait", help="Block until the seat is serving."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Bring up a model seat (spawn on free GPUs, or swap a named seat).

    With no model id, opens an interactive picker over every registry placement —
    ↑/↓ to choose, enter to load.
    """
    from .engine import launch

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


@app.command()
def down(
    seat: str = typer.Argument(None, help="Seat/container name (or model id). Omit to pick interactively."),
    drain: bool = typer.Option(False, "--drain", help="Graceful drain (no-op without a router)."),
    json_output: bool = typer.Option(False, "--json"),
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


@app.command()
def swap(
    seat: str = typer.Argument(..., help="Running seat to replace."),
    model: str = typer.Argument(..., help="Model to launch in its place."),
    wait: bool = typer.Option(False, "--wait"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Replace one seat in place (same cards/port)."""
    from .engine import launch

    try:
        res = launch.swap(seat, model, wait=wait)
    except Exception as e:
        _emit_err(e, json_output)
    console.print(_json.dumps(res, indent=2) if json_output else
                  f"[green]●[/] swapped {seat} → [bold]{res['seat']}[/] (state {res.get('state')})")


@app.command()
def reap(
    idle_ttl: int = typer.Option(None, "--idle-ttl", help="Idle seconds before reaping (default 1800)."),
    dry_run: bool = typer.Option(False, "--dry-run"),
    json_output: bool = typer.Option(False, "--json"),
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


@app.command()
def pin(
    seat: str = typer.Argument(...),
    ttl: int = typer.Option(None, "--ttl", help="Seconds; omit for indefinite."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Exempt a seat from the reaper (ephemeral pin in the telemetry SQLite)."""
    from .telemetry import collect

    collect.add_pin(seat, ttl_s=ttl)
    msg = {"pinned": seat, "ttl_s": ttl}
    console.print(_json.dumps(msg, indent=2) if json_output else
                  f"[green]✓[/] pinned {seat}" + (f" for {ttl}s" if ttl else " (indefinite)"))


@app.command()
def unpin(seat: str = typer.Argument(...), json_output: bool = typer.Option(False, "--json")) -> None:
    """Remove a seat's reaper exemption."""
    from .telemetry import collect

    collect.remove_pin(seat)
    console.print(_json.dumps({"unpinned": seat}, indent=2) if json_output else f"[green]✓[/] unpinned {seat}")


@app.command()
def resolve(
    target: str = typer.Argument(..., help="Role, seat, or model id."),
    json_output: bool = typer.Option(False, "--json"),
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


@app.command()
def logs(
    seat: str = typer.Argument(...),
    follow: bool = typer.Option(False, "-f", "--follow"),
    tail: int = typer.Option(200, "--tail"),
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


@app.command()
def metrics(
    seat: str = typer.Argument(...),
    history: bool = typer.Option(False, "--history", help="Aggregate trends from the telemetry SQLite."),
    since: int = typer.Option(None, "--since", help="History window in seconds (default: all)."),
    json_output: bool = typer.Option(False, "--json"),
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
        for col in ("TP", "QUANT", "GB/GPU", "KV-CEILING CTX"):
            vt.add_column(col)
        for v in pl["viable"]:
            vt.add_row(str(v["tp"]), str(v.get("quant")), str(v.get("per_gpu_gb")), str(v.get("kv_ceiling_ctx")))
        console.print(vt)
        if pl["pruned"]:
            console.print("[dim]pruned:[/]")
            for p in pl["pruned"]:
                console.print(f"  [yellow]✗ tp={p.get('tp')}[/] — {p.get('reason')}")
    if pl.get("arch_supported") is False:
        console.print(f"[red]✗ unsupported architecture[/] — {pl.get('arch_warning')}")
        console.print("[dim]nothing to sweep; induct would abort before launching any seat.[/]")
        return
    console.print(f"[bold]{len(pl['points'])}[/] candidate config point(s) to sweep "
                  f"[dim](seeded search, not a brute grid)[/]")


@app.command()
def induct(
    model: str = typer.Argument(..., help="HF id, registry id, or local path."),
    use_case: str = typer.Option(None, "--use-case", help="Winner pick: throughput (max peak tok/s under concurrency) | latency (fastest single-stream tok/s) | context (largest usable context)"),
    device: str = typer.Option("auto", "--device", help="gpu | cpu | auto (auto falls back to CPU if no GPU fits)."),
    tp: int = typer.Option(None, "--tp", help="Force tensor-parallel size: sweep only this TP (must be a viable placement). Overrides the auto winner's TP."),
    embeddings: bool = typer.Option(None, "--embeddings/--no-embeddings", help="Force embeddings vs generative bench (default: auto-detect)."),
    bench: bool = typer.Option(False, "--bench", help="Also run the quality harness (heavy/opt-in)."),
    plan: bool = typer.Option(False, "--plan", help="Dry preview: viable placements + candidate grid, no launches."),
    resume: bool = typer.Option(False, "--resume", help="Continue a previous run, skipping done points."),
    max_points: int = typer.Option(None, "--max-points", help="Cap candidate points (bounded runs)."),
    yes: bool = typer.Option(False, "--yes", help="Skip the pre-sweep confirmation."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Auto-tune a model into an optimal placement (tuning by default; GPU or CPU)."""
    from .induct import pipeline

    if plan:
        try:
            pl = pipeline.plan(model, max_points=max_points, device=device, embeddings=embeddings, tp=tp)
        except Exception as e:
            _emit_err(e, json_output)
        if json_output:
            console.print(_json.dumps(pl, indent=2))
        else:
            _render_plan(pl)
        return

    if not yes and not json_output:
        try:
            pl = pipeline.plan(model, max_points=max_points, device=device, embeddings=embeddings, tp=tp)
        except Exception as e:
            _emit_err(e, json_output)
        _render_plan(pl)
        if not pl["points"]:
            raise typer.Exit(code=1)
        where = "CPU" if pl.get("device") == "cpu" else "GPU"
        if not typer.confirm(f"Launch {len(pl['points'])} {where} tuning seat(s)? (each is a real load + bench)"):
            raise typer.Exit(code=1)

    prog = None if json_output else (lambda m: console.print(f"[dim]· {m}[/]"))
    try:
        res = pipeline.run(model, use_case=use_case, bench=bench, resume=resume, max_points=max_points,
                           progress=prog, device=device, embeddings=embeddings, tp=tp)
    except Exception as e:
        _emit_err(e, json_output)
    if json_output:
        console.print(_json.dumps(res, indent=2, default=str))
        return
    if res.get("error"):
        err.print(f"[red]{res['error']}[/]")
        raise typer.Exit(code=1)
    w = res.get("winner")
    if w:
        wp = w["point"]
        console.print(f"[green]✓ winner[/] TP={wp.get('tp')} gmu={wp.get('gpu_memory_util')} "
                      f"mml={wp.get('max_model_len')} → peak {w.get('peak_tok_s')} tok/s, single {w.get('single_tok_s')} tok/s")
        console.print(f"  wrote placement [bold]{res['placement_id']}[/] to the registry")
    else:
        console.print("[yellow]no winning config[/] (all points failed — see the report)")
    console.print(f"  report: {res['report']}  ·  bench: {res['bench']}")


@app.command()
def tune(
    model: str = typer.Argument(..., help="Registry id or local path."),
    use_case: str = typer.Option(None, "--use-case", help="Winner pick: throughput (max peak tok/s under concurrency) | latency (fastest single-stream tok/s) | context (largest usable context)"),
    resume: bool = typer.Option(False, "--resume"),
    max_points: int = typer.Option(None, "--max-points"),
    yes: bool = typer.Option(False, "--yes"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Re-tune an existing model (induction, tuning-only)."""
    induct(model=model, use_case=use_case, bench=False, plan=False, resume=resume,
           max_points=max_points, yes=yes, json_output=json_output)


# --------------------------------------------------------------------------- discovery (P5)
_VERDICT_STYLE = {"fits": "green", "tight": "yellow", "wont-fit": "red", "unknown": "dim"}


def _dtype_cell(d: dict) -> str:
    """Render a dtype-fit verdict: native ✓ / not-native ✗ / unknown —."""
    ok = (d or {}).get("ok")
    need = (d or {}).get("need")
    if ok is True:
        return f"[green]✓ {need}[/]"
    if ok is False:
        return f"[red]✗ {need or (d or {}).get('detail', '')}[/]"
    return "[dim]—[/]"


@app.command()
def search(
    query: str = typer.Argument(..., help="HF search query, or a base model id with --quants."),
    quants: bool = typer.Option(False, "--quants", "-q",
                                help="List quantizations of QUERY (a base model id) with a dtype-fit verdict."),
    limit: int = typer.Option(10, "--limit"),
    json_output: bool = typer.Option(False, "--json"),
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
        for col in ("MODEL", "QUANT", "DTYPE", "SIZE", "FIT"):
            t.add_column(col)
        for r in res["results"]:
            v = r["fit"]
            verdict = f"[{_VERDICT_STYLE.get(v['verdict'], 'white')}]{v['verdict']}[/]"
            label = r["id"] + ("  [dim](base)[/]" if r.get("base") else "")
            t.add_row(label, str(r.get("quant") or "—"), _dtype_cell(r["dtype"]),
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
    for col in ("MODEL", "DOWNLOADS", "GATED", "SIZE", "DTYPE", "FIT", "BADGES"):
        t.add_column(col)
    for r in res["results"]:
        v = r["fit"]
        verdict = f"[{_VERDICT_STYLE.get(v['verdict'], 'white')}]{v['verdict']}[/]"
        t.add_row(r["id"], str(r.get("downloads") or "—"), "🔒" if r["gated"] else "",
                  f"{r['size_gb']}GB" if r["size_gb"] else "—", _dtype_cell(r.get("dtype")),
                  f"{verdict} {v.get('detail', '')}", ", ".join(r["badges"]) or "—")
    console.print(t)


@app.command()
def download(repo: str = typer.Argument(...), json_output: bool = typer.Option(False, "--json")) -> None:
    """Download a model into the models dir (gated models need `johnny login`)."""
    from .discover import search as dsearch

    cfg = C.load_yaml(C.get_paths().config_file) or {}
    models_dir = (cfg.get("roots") or {}).get("models_dir")
    if not models_dir:
        err.print("[red]no models_dir in config[/] — run `johnny init`.")
        raise typer.Exit(code=1)
    console.print(f"[dim]downloading {repo} → {models_dir}/{repo} … (large; ^C to abort)[/]")
    res = dsearch.acquire(repo, models_dir)
    if res.get("error"):
        err.print(f"[red]{res['error']}[/]")
        raise typer.Exit(code=1)
    console.print(_json.dumps(res, indent=2) if json_output else f"[green]✓ downloaded[/] {repo} → {res['path']}")


@app.command()
def login(
    token: str = typer.Option(None, "--token", help="HF token; omit to show status."),
    json_output: bool = typer.Option(False, "--json"),
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
@app.command()
def alive(
    model: str = typer.Option(None, "--model", help="Target a specific model."),
    seat: str = typer.Option(None, "--seat", help="Target a specific seat."),
    role: str = typer.Option("orchestrator", "--role", help="Target seat by role (default)."),
    no_wait: bool = typer.Option(False, "--no-wait", help="Don't wait for a loading seat."),
    timeout: int = typer.Option(900, "--timeout"),
    no_attach: bool = typer.Option(False, "--no-attach", help="Start detached (don't attach the tmux session)."),
    session: str = typer.Option(None, "--session"),
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


provider_app = typer.Typer(add_completion=False, help="Sync an external chat tool's provider config.")
app.add_typer(provider_app, name="provider")


@provider_app.command("sync")
def provider_sync(
    write: bool = typer.Option(False, "--write", help="Patch the config in place (timestamped backup)."),
    provider: str = typer.Option(None, "--provider", help="Chat provider name (default: config [external].provider)."),
    json_output: bool = typer.Option(False, "--json"),
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
@app.command()
def cleanup(
    apply: bool = typer.Option(False, "--apply", help="Actually delete (default: dry-run preview)."),
    json_output: bool = typer.Option(False, "--json"),
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


@app.command(name="rm")
def rm(
    target: str = typer.Argument(..., help="Model id, local path (vendor/name), or directory."),
    registry_only: bool = typer.Option(False, "--registry-only", help="Deregister but keep the weights on disk."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
    json_output: bool = typer.Option(False, "--json"),
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
app.add_typer(daemon_app, name="daemon")


def _daemon_pidfile(agent: bool = False):
    return C.get_paths().state_dir / ("johnnyd-agent.json" if agent else "johnnyd.json")


@daemon_app.command("up")
def daemon_up(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
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
def daemon_status(json_output: bool = typer.Option(False, "--json")) -> None:
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


@app.command()
def nodes(
    controller: str = typer.Option("http://127.0.0.1:8080", "--controller"),
    json_output: bool = typer.Option(False, "--json"),
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
@app.command()
def tui() -> None:
    """Launch the live Textual dashboard (seats, concurrency, KV — by backend/model)."""
    from .tui.app import run as run_tui

    run_tui()


# --------------------------------------------------------------------------- future stubs
_FUTURE = {
    "bench": "P4",  # quality-eval harness orchestration (heavy/opt-in); wired via `induct --bench`
}


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
