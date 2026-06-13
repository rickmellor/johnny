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
def _render_status(json_output: bool = False) -> None:
    if not probe.docker_available():
        if json_output:
            console.print(_json.dumps({"docker": False, "seats": []}, indent=2))
        else:
            err.print("[red]docker is not reachable[/] — is the daemon running? (`johnny doctor`)")
        return

    seats = probe.list_seats()
    if json_output:
        console.print(_json.dumps({"docker": True, "seats": seats}, indent=2))
        return

    if not seats:
        console.print("[dim]no inference seats running.[/] Bring one up once the engine lands (P3); for now use your existing tooling.")
        return

    table = Table(title="johnny — seats (P0: derived from docker + /v1/models)", title_style="bold")
    table.add_column("SEAT", style="bold")
    table.add_column("PORT")
    table.add_column("SERVED MODEL", style="cyan")
    table.add_column("STATE")
    table.add_column("IMAGE", style="dim")
    for s in seats:
        st = s["state"]
        table.add_row(
            s["seat"],
            str(s["port"] or "—"),
            s["model"] or "—",
            f"[{_STATE_STYLE.get(st, 'white')}]{st}[/]",
            s["image"],
        )
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
    table = Table(title="johnny — seats (live)", title_style="bold")
    for col, style in (("SEAT", "bold"), ("PORT", None), ("SERVED MODEL", "cyan"), ("STATE", None), ("IMAGE", "dim")):
        table.add_column(col, style=style)
    if not probe.docker_available():
        table.add_row("[red]docker unreachable[/]", "—", "—", "—", "—")
        return table
    for s in probe.list_seats():
        st = s["state"]
        table.add_row(s["seat"], str(s["port"] or "—"), s["model"] or "—",
                      f"[{_STATE_STYLE.get(st, 'white')}]{st}[/]", s["image"])
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


# --------------------------------------------------------------------------- future stubs
_FUTURE = {
    "up": "P3", "down": "P3", "swap": "P3", "reap": "P3", "resolve": "P3",
    "pin": "P3", "unpin": "P3", "logs": "P3", "metrics": "P3",
    "registry": "P2", "induct": "P4", "tune": "P4", "bench": "P4",
    "search": "P5", "download": "P5", "login": "P5", "alive": "P6",
    "provider": "P6", "cleanup": "P8", "nodes": "P11",
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
