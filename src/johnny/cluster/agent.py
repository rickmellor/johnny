"""Node agent: dial out to the controller, register, heartbeat inventory, run commands.

Single direction (agent -> controller), so it's NAT/firewall-friendly. Commands are
delivered as the heartbeat response (poll-on-heartbeat); the agent executes them
locally via the engine and posts results back.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.request

from .. import engine
from ..backends import available_drivers
from ..hardware import detect as hwd
from ..util import run


def _post(url: str, obj: dict, timeout: float = 15) -> dict:
    data = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (trusted LAN/controller)
        return json.loads(r.read() or b"{}")


def _inventory(token: str, node_id: str, hw) -> dict:
    rc, dv, _ = run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=5)
    seats = [
        {"seat": s.name, "model": s.model, "state": s.state, "port": s.port, "gpus": s.gpus, "backend": s.backend}
        for s in engine.all_seats()
    ]
    return {
        "node_id": node_id,
        "token": token,
        "hardware": {"vendor": hw.vendor, "fingerprint": hw.fingerprint, "gpus": len(hw.gpus),
                     "vram_gb": hw.total_vram_gb, "native_dtypes": hw.native_dtypes},
        "software": {"docker": dv.strip() if rc == 0 else None, "backends": available_drivers()},
        "seats": seats,
    }


def node_id_for(hw) -> str:
    return f"{socket.gethostname()}::{hw.fingerprint}"


def run_agent(controller_url: str, token: str = "", interval: int = 10) -> None:
    hw = hwd.detect()
    node_id = node_id_for(hw)
    base = controller_url.rstrip("/")
    try:
        _post(base + "/cluster/register", _inventory(token, node_id, hw))
    except Exception as e:  # keep trying via heartbeats
        print(f"[agent] register failed: {e}")
    while True:
        try:
            resp = _post(base + "/cluster/heartbeat", _inventory(token, node_id, hw))
            for cmd in resp.get("commands", []):
                _execute(cmd, base, node_id, token)
        except Exception:
            pass  # controller down → keep serving locally, retry next tick
        time.sleep(interval)


def _execute(cmd: dict, base: str, node_id: str, token: str) -> None:
    from ..engine import launch

    res = {"node_id": node_id, "token": token, "id": cmd.get("id"), "action": cmd.get("action")}
    try:
        if cmd.get("action") == "up":
            res["result"] = launch.up(cmd["model"], placement_id=cmd.get("placement"),
                                      force=cmd.get("force", False), wait=False)
        elif cmd.get("action") == "down":
            res["result"] = launch.down(cmd["seat"])
        else:
            res["error"] = f"unknown action {cmd.get('action')}"
    except Exception as e:
        res["error"] = str(e)
    try:
        _post(base + "/cluster/result", res)
    except Exception:
        pass
