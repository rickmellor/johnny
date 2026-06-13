"""Launch a chat TUI against a johnny seat (generalized `alive`, §3.9).

Resolve a seat by role (default chat/orchestrator) or explicit model/seat, wait
for it to be ready, then launch a TUI in tmux. A small adapter keeps this
provider-agnostic; the built-in Hermes adapter reproduces today's behavior
(SSH_* stripping for local audio + the tmux session + spoken readiness).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time

from ..engine import load_config
from ..engine import service
from ..runtime import probe
from ..util import which


def _strip_ssh_env() -> dict:
    e = dict(os.environ)
    for k in ("SSH_CLIENT", "SSH_TTY", "SSH_CONNECTION"):
        e.pop(k, None)
    return e


def _say(msg: str) -> None:
    if which("spd-say"):
        try:
            subprocess.Popen(["spd-say", "-r", "-20", "-p", "-20", msg])
        except OSError:
            pass


class HermesAdapter:
    name = "hermes"

    def command(self, served_model: str, provider: str = "johnny", extra=None) -> list[str]:
        cmd = ["env", "-u", "SSH_CLIENT", "-u", "SSH_TTY", "-u", "SSH_CONNECTION",
               "hermes", "chat", "--tui", "-m", served_model, "--provider", provider]
        if extra:
            cmd += list(extra)
        return cmd


class GenericAdapter:
    """A user-configured command template with {model}/{port}/{base_url} substitution."""

    name = "generic"

    def __init__(self, template: list[str]):
        self.template = template

    def command(self, served_model: str, base_url: str | None = None, port=None, extra=None) -> list[str]:
        subst = {"model": served_model, "base_url": base_url or "", "port": str(port or "")}
        cmd = [part.format(**subst) for part in self.template]
        if extra:
            cmd += list(extra)
        return cmd


def _tmux_has(session: str) -> bool:
    rc = subprocess.call(["tmux", "has-session", "-t", session],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rc == 0


def alive(target: str | None = None, role: str = "orchestrator", wait: bool = True, timeout: float = 900,
          attach: bool = True, session: str | None = None, provider: str | None = None,
          extra_args=None, cfg: dict | None = None) -> dict:
    cfg = cfg if cfg is not None else load_config()
    ext = cfg.get("external") or {}
    provider = provider or ext.get("provider") or "johnny"
    session = session or ext.get("tmux_session") or "johnny"
    adapter_name = ext.get("adapter", "hermes")
    tgt = target or role
    res = service.resolve(tgt, cfg)

    if res["state"] == "absent":
        return {"error": f"no seat for '{tgt}' (absent). Start one: `johnny up <model>`"}
    if res["state"] in ("loading",):
        if not wait:
            return {"error": f"seat '{res['seat']}' is loading (eta_s={res.get('eta_s')}); --no-wait set"}
        deadline = time.time() + timeout
        while time.time() < deadline:
            r2 = service.resolve(tgt, cfg)
            if r2["state"] == "ready":
                res = r2
                break
            time.sleep(3)
        if res["state"] != "ready":
            return {"error": f"timed out waiting for '{tgt}' to become ready"}

    served = res["model"]
    if not which("tmux"):
        return {"error": "tmux not installed (needed to host the TUI session)"}

    if adapter_name == "generic" and ext.get("command_template"):
        inner = GenericAdapter(ext["command_template"]).command(served, base_url=res.get("endpoint"), extra=extra_args)
    else:
        inner = HermesAdapter().command(served, provider=provider, extra=extra_args)
    fresh = False
    if not _tmux_has(session):
        subprocess.run(["tmux", "new-session", "-d", "-s", session,
                        " ".join(shlex.quote(c) for c in inner)], env=_strip_ssh_env())
        fresh = True
        _say("Hermes is alive and ready.")
    return {
        "action": "attach" if attach else "detached",
        "session": session, "seat": res["seat"], "model": served,
        "endpoint": res.get("endpoint"), "fresh": fresh, "command": inner,
    }
