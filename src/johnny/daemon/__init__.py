"""johnnyd — the optional soft daemon (§3.11 / P10).

Hosts the request-plane HTTP API (the §3.13 contract: /v1/fleet, /resolve, ensure-
loaded, leases) and a minimal OpenAI-compatible JIT gateway (load-on-first-request,
concurrency caps) for users without an external router. Built on stdlib http.server
to stay dependency-light. SAINT remains the reference *external* consumer.
"""
