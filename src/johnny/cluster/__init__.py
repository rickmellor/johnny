"""Multi-machine (§3.12 / P11) — minimal controller + per-node agents.

Deliberately dumb: agents dial OUT to the controller (NAT-friendly), register +
heartbeat their inventory, and poll for commands on the heartbeat (no websocket
needed). Explicit/locality placement; no reconciler/affinity engine. docker stays
each node's source of truth, so seats survive controller loss.
"""
