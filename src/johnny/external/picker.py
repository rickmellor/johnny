"""Inline pickers — arrow keys (↑/↓ or k/j), Enter to confirm, q/Esc to cancel.
`select` is single-choice; `multi_select` adds space-to-toggle checkboxes ('a' toggles
all). Stdlib-only (termios/tty) with a numbered-prompt fallback for non-TTY stdin or
platforms without termios (Windows). No extra dependency; renders via rich.

Used by `johnny up` (no model arg) to choose a placement visually, `johnny down`, and
induct's end-of-sweep seat picker.
"""

from __future__ import annotations

import sys
from typing import Callable, Sequence

from rich.console import Console, Group
from rich.text import Text

console = Console()


def _interactive_capable() -> bool:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    try:
        import termios  # noqa: F401
        import tty  # noqa: F401
    except Exception:
        return False
    return True


def _decode(data: bytes) -> str:
    """Pure: raw key bytes → 'up'|'down'|'enter'|'toggle'|'all'|'cancel'|'other'.
    Handles both ESC[ and ESCO cursor modes, Enter, Ctrl-C, lone Esc, j/k/q, and
    space/'a' (checkbox toggles — no-ops in the single-select loop)."""
    if data in (b"\x1b[A", b"\x1bOA"):
        return "up"
    if data in (b"\x1b[B", b"\x1bOB"):
        return "down"
    if data == b"\x1b":
        return "cancel"
    first = data[:1]
    if first in (b"\r", b"\n"):
        return "enter"
    if first == b"\x03":  # Ctrl-C
        return "cancel"
    return {b"q": "cancel", b"k": "up", b"j": "down",
            b" ": "toggle", b"a": "all"}.get(first.lower(), "other")


def _read_key() -> str:
    """One keypress → an action. Raw mode per call so the terminal is always restored.

    Reads raw bytes via os.read (NOT sys.stdin.read, whose text buffer would swallow
    the rest of an arrow escape sequence and make a select() on the fd look empty —
    the bug that turned ↑/↓ into 'cancel')."""
    import os
    import select as _select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        data = os.read(fd, 3)  # arrow keys arrive as ESC[A / ESC[B in one read
        if data == b"\x1b":  # lone ESC so far — maybe a split escape sequence
            ready, _, _ = _select.select([fd], [], [], 0.05)
            if ready:
                data += os.read(fd, 2)
        return _decode(data)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select(
    items: Sequence,
    render: Callable[[object], str],
    title: str = "select",
    hint: str = "↑/↓ move · enter select · q cancel",
) -> int | None:
    """Show items, return the chosen index, or None if cancelled/empty.

    render(item) returns a rich-markup string (the picker adds the cursor/highlight).
    Falls back to a numbered prompt when stdin/stdout isn't an interactive TTY.
    """
    if not items:
        return None
    if not _interactive_capable():
        return _numbered(items, render, title)

    from rich.live import Live

    idx = 0

    def frame():
        rows = [Text.from_markup(f"[bold]{title}[/]  [dim]{hint}[/]"), Text("")]
        for i, it in enumerate(items):
            line = render(it)
            rows.append(Text.from_markup(f"[reverse]❯ {line}[/reverse]" if i == idx else f"  {line}"))
        return Group(*rows)

    with Live(frame(), console=console, auto_refresh=False) as live:
        while True:
            key = _read_key()
            if key == "up":
                idx = (idx - 1) % len(items)
            elif key == "down":
                idx = (idx + 1) % len(items)
            elif key == "enter":
                return idx
            elif key == "cancel":
                return None
            live.update(frame())
            live.refresh()


def _numbered(items: Sequence, render: Callable[[object], str], title: str) -> int | None:
    console.print(f"[bold]{title}[/]  [dim](not a TTY — numbered fallback)[/]")
    for i, it in enumerate(items):
        console.print(f"  [bold]{i + 1}[/]) " + render(it))
    try:
        raw = input("select # (blank to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(items) else None


def multi_select(
    items: Sequence,
    render: Callable[[object], str],
    title: str = "select",
    hint: str = "↑/↓ move · space toggle · a all · enter accept · q cancel",
    preselected: set[int] | None = None,
) -> list[int] | None:
    """Checkbox picker: return the chosen indices sorted in row order, or None if
    cancelled/empty. Enter accepts the current toggle set (which may be empty — the
    caller decides what an empty pick means). Falls back to a comma-separated numbered
    prompt when stdin/stdout isn't an interactive TTY (blank keeps the preselection)."""
    if not items:
        return None
    chosen = set(preselected or ())
    if not _interactive_capable():
        return _numbered_multi(items, render, title, chosen)

    from rich.live import Live

    idx = 0

    def frame():
        rows = [Text.from_markup(f"[bold]{title}[/]  [dim]{hint}[/]"), Text("")]
        for i, it in enumerate(items):
            line = f"{'[green]◉[/]' if i in chosen else '[dim]○[/]'} {render(it)}"
            rows.append(Text.from_markup(f"[reverse]❯ {line}[/reverse]" if i == idx else f"  {line}"))
        return Group(*rows)

    with Live(frame(), console=console, auto_refresh=False) as live:
        while True:
            key = _read_key()
            if key == "up":
                idx = (idx - 1) % len(items)
            elif key == "down":
                idx = (idx + 1) % len(items)
            elif key == "toggle":
                chosen.symmetric_difference_update({idx})
            elif key == "all":
                chosen = set() if len(chosen) == len(items) else set(range(len(items)))
            elif key == "enter":
                return sorted(chosen)
            elif key == "cancel":
                return None
            live.update(frame())
            live.refresh()


def parse_picks(raw: str, n: int) -> list[int] | None:
    """'3', '1,7 12' → sorted unique 1-based numbers within [1, n]; 'a'/'all' → all;
    None on anything unparsable or out of range (the caller re-prompts or cancels)."""
    s = raw.strip().lower()
    if s in ("a", "all"):
        return list(range(1, n + 1))
    try:
        picks = sorted({int(tok) for tok in s.replace(",", " ").split()})
    except ValueError:
        return None
    if not picks or picks[0] < 1 or picks[-1] > n:
        return None
    return picks


def _numbered_multi(items: Sequence, render: Callable[[object], str], title: str,
                    preselected: set[int]) -> list[int] | None:
    console.print(f"[bold]{title}[/]  [dim](not a TTY — numbered fallback)[/]")
    for i, it in enumerate(items):
        mark = "◉" if i in preselected else "○"
        console.print(f"  [bold]{i + 1}[/]) {mark} " + render(it))
    try:
        raw = input("select #s (comma-separated, 'a'=all, blank=keep marked): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return sorted(preselected)
    picks = parse_picks(raw, len(items))
    return None if picks is None else [p - 1 for p in picks]
