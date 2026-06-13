"""johnny live dashboard (Textual).

A presentation layer over the engine + telemetry: a live seats table broken out by
backend/model with live concurrency (running/waiting) + KV utilization pulled from
each seat's metrics, auto-refreshing. No domain logic lives here — it calls the same
engine functions the CLI does.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

from .. import engine
from ..telemetry import sources


class JohnnyTUI(App):
    TITLE = "johnny — fleet"
    BINDINGS = [("q", "quit", "Quit"), ("r", "refresh", "Refresh")]
    CSS = """
    #status { padding: 0 1; color: $text-muted; }
    DataTable { height: 1fr; }
    """

    REFRESH_SECONDS = 2.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield DataTable(id="seats", zebra_stripes=True)
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#seats", DataTable)
        table.add_columns("SEAT", "BACKEND", "MODEL", "STATE", "PORT", "GPUS", "RUN", "WAIT", "KV%")
        self.refresh_data()
        self.set_interval(self.REFRESH_SECONDS, self.refresh_data)

    def action_refresh(self) -> None:
        self.refresh_data()

    def _metrics(self, seat) -> dict:
        if not seat.port or seat.state != "ready":
            return {}
        try:
            return sources.metrics_for_port(seat.port)
        except Exception:
            return {}

    def refresh_data(self) -> None:
        table = self.query_one("#seats", DataTable)
        table.clear()
        try:
            seats = engine.all_seats()
        except Exception as e:  # never let a transient docker hiccup crash the UI
            self.query_one("#status", Static).update(f"[red]error: {e}[/]")
            return
        for s in sorted(seats, key=lambda x: (x.backend, x.name)):
            m = self._metrics(s)
            kv = m.get("kv_util")
            table.add_row(
                s.name, s.backend, s.model or "—", s.state, str(s.port or "—"),
                ",".join(map(str, s.gpus)) or "—",
                str(m.get("running", "—")), str(m.get("waiting", "—")),
                f"{kv * 100:.0f}" if isinstance(kv, (int, float)) else "—",
            )
        self.query_one("#status", Static).update(
            f"{len(seats)} seat(s) · refresh {self.REFRESH_SECONDS:.0f}s · [r]efresh · [q]uit"
        )


def run() -> None:
    JohnnyTUI().run()
