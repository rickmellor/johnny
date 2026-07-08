"""Tiny shared helpers (subprocess + PATH lookup), dependency-light on purpose."""

from __future__ import annotations

import shutil
import subprocess


def which(name: str) -> str | None:
    """Absolute path to an executable on PATH, or None."""
    return shutil.which(name)


def run(cmd: list[str], timeout: float = 10.0, env: dict | None = None) -> tuple[int, str, str]:
    """Run a command, capturing output. Never raises.

    Returns (returncode, stdout, stderr). 127 = not found, 124 = timeout.
    `env`, when given, fully replaces the child environment (pass {**os.environ, ...}).
    """
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as e:  # pragma: no cover - defensive
        return 1, "", str(e)
