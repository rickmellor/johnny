"""Boot integration — a user systemd template unit that brings a profile up at boot.

Pure text/path helpers (no side effects) so the unit is unit-testable; the CLI
(`johnny profile enable/disable`) does the writing and systemctl calls.

Why a *user* unit: no sudo, docker access via the user's group, and the user's
HOME/XDG so johnny finds its config — it just needs lingering enabled
(`loginctl enable-linger <user>`) to start at boot without a login.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .util import which

UNIT_NAME = "johnny-profile@.service"


def unit_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def unit_path() -> Path:
    return unit_dir() / UNIT_NAME


def instance(profile: str) -> str:
    return f"johnny-profile@{profile}.service"


def johnny_cmd() -> str:
    """Absolute command systemd should run (units don't get the user's PATH)."""
    exe = which("johnny")
    if exe:
        return exe
    local = Path.home() / ".local" / "bin" / "johnny"
    if local.exists():
        return str(local)
    return f"{sys.executable} -m johnny"


def unit_text(johnny: str | None = None) -> str:
    johnny = johnny or johnny_cmd()
    return f"""\
[Unit]
Description=johnny run profile %i

[Service]
Type=oneshot
RemainAfterExit=yes
# A user unit can't order After= the system docker.service; wait for the socket.
ExecStartPre=/bin/sh -c 'until docker info >/dev/null 2>&1; do sleep 2; done'
ExecStart={johnny} profile up %i --wait
ExecStop={johnny} profile down %i
# Serial --wait cold starts (a 27B FP8 load is minutes); give the fleet room.
TimeoutStartSec=1800

[Install]
WantedBy=default.target
"""


def enable_commands(profile: str) -> list[list[str]]:
    return [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", instance(profile)],
    ]


def disable_commands(profile: str) -> list[list[str]]:
    # Plain disable (no --now): the unit's ExecStop runs `profile down`, so
    # --now would stop the running fleet when the user only meant "stop
    # auto-starting". `johnny profile down` is the explicit stop.
    return [["systemctl", "--user", "disable", instance(profile)]]
