"""Controller-side node registry (in-memory) + per-node command queue."""

from __future__ import annotations

import threading
import time
from collections import defaultdict

HEARTBEAT_TTL = 45  # seconds before a silent node is marked NotReady


class NodeRegistry:
    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._cmds: dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
        self._seq = 0

    def register(self, node: dict) -> str:
        nid = node["node_id"]
        with self._lock:
            self._nodes[nid] = {**node, "last_seen": time.time()}
        return nid

    def heartbeat(self, node_id: str, payload: dict) -> list:
        """Update inventory + last_seen; return (and clear) any queued commands."""
        with self._lock:
            entry = self._nodes.get(node_id, {"node_id": node_id})
            entry.update(payload)
            entry["last_seen"] = time.time()
            self._nodes[node_id] = entry
            cmds = self._cmds.pop(node_id, [])
        return cmds

    def enqueue(self, node_id: str, cmd: dict) -> int:
        with self._lock:
            self._seq += 1
            cmd = {**cmd, "id": self._seq}
            self._cmds[node_id].append(cmd)
        return cmd["id"]

    def nodes(self) -> list[dict]:
        now = time.time()
        with self._lock:
            out = []
            for nid, n in self._nodes.items():
                status = "ready" if (now - n.get("last_seen", 0)) < HEARTBEAT_TTL else "NotReady"
                out.append({**{k: v for k, v in n.items() if k != "token"}, "status": status})
            return out


REGISTRY = NodeRegistry()
