"""johnnyd HTTP server: request-plane API + OpenAI-compatible JIT gateway.

Read API (the §3.13 contract over HTTP):
  GET  /healthz
  GET  /v1/fleet                 -> all seats
  GET  /v1/models                -> served models across seats (OpenAI-shaped)
  GET  /resolve?target=<x>       -> resolve (the SAINT hot path)
  POST /up      {model,...}       -> ensure-loaded (non-blocking)
  POST /pin     {seat,ttl?}       -> lease;   POST /unpin {seat}

Gateway (for users without an external router; SAINT is the reference external one):
  POST /v1/chat/completions | /v1/completions | /v1/embeddings
    -> resolve the body's "model"; if ready, proxy (streaming) to the seat; if not,
       JIT-trigger a load (when enabled) and return 503 {state:loading, eta_s} —
       never block the triggering request (cold start is minutes). A per-seat
       concurrency cap returns 429 when exceeded.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .. import __version__
from .. import engine
from ..engine import launch, service

_HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding"}
_inflight: dict[str, int] = defaultdict(int)
_inflight_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = f"johnnyd/{__version__}"
    jit = True
    max_concurrent = 0  # 0 = unlimited

    def log_message(self, *a):  # quiet
        pass

    # --- helpers ---
    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(n) if n else b""

    # --- routing ---
    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/healthz":
            return self._json(200, {"ok": True, "version": __version__})
        if u.path == "/v1/fleet":
            seats = engine.all_seats()
            return self._json(200, {"seats": [self._seat(s) for s in seats]})
        if u.path == "/v1/models":
            seats = engine.all_seats()
            data = [{"id": s.model, "object": "model", "owned_by": s.backend}
                    for s in seats if s.model and s.state == "ready"]
            return self._json(200, {"object": "list", "data": data})
        if u.path == "/resolve":
            q = parse_qs(u.query)
            target = (q.get("target") or [None])[0]
            if not target:
                return self._json(400, {"error": "missing ?target="})
            return self._json(200, service.resolve(target))
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        u = urlparse(self.path)
        if u.path == "/up":
            return self._post_up()
        if u.path in ("/pin", "/unpin"):
            return self._post_pin(u.path == "/pin")
        if u.path in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings"):
            return self._gateway(u.path)
        self._json(404, {"error": "not found"})

    @staticmethod
    def _seat(s) -> dict:
        return {"seat": s.name, "backend": s.backend, "model": s.model, "port": s.port,
                "state": s.state, "gpus": s.gpus}

    def _post_up(self) -> None:
        try:
            req = json.loads(self._body() or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad json"})
        model = req.get("model")
        if not model:
            return self._json(400, {"error": "missing model"})
        try:
            res = launch.up(model, placement_id=req.get("placement"), force=req.get("force", False), wait=False)
            return self._json(200, res)
        except Exception as e:
            return self._json(409, {"error": str(e)})

    def _post_pin(self, pin: bool) -> None:
        from ..telemetry import collect

        try:
            req = json.loads(self._body() or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad json"})
        seat = req.get("seat")
        if not seat:
            return self._json(400, {"error": "missing seat"})
        if pin:
            collect.add_pin(seat, ttl_s=req.get("ttl"))
        else:
            collect.remove_pin(seat)
        return self._json(200, {"seat": seat, "pinned": pin})

    def _gateway(self, path: str) -> None:
        body = self._body()
        try:
            req = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad json"})
        model = req.get("model")
        if not model:
            return self._json(400, {"error": "missing model"})
        res = service.resolve(model)
        state = res["state"]
        if state != "ready" or not res.get("endpoint"):
            if self.jit and state in ("absent", "loading"):
                try:
                    launch.up(model, wait=False)  # non-blocking JIT trigger
                except Exception:
                    pass
            return self._json(503, {"error": "model not ready", "state": state, "eta_s": res.get("eta_s")})

        seat = res["seat"]
        if self.max_concurrent:
            with _inflight_lock:
                if _inflight[seat] >= self.max_concurrent:
                    return self._json(429, {"error": "seat at concurrency cap", "seat": seat})
                _inflight[seat] += 1
        try:
            self._proxy(res["endpoint"] + path[len("/v1"):], body)
        finally:
            if self.max_concurrent:
                with _inflight_lock:
                    _inflight[seat] -= 1

    def _proxy(self, upstream: str, body: bytes) -> None:
        req = urllib.request.Request(upstream, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in _HOP:
                        self.send_header(k, v)
                self.end_headers()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except Exception as e:
            self._json(502, {"error": f"upstream proxy failed: {e}"})


def serve(host: str = "127.0.0.1", port: int = 8080, jit: bool = True, max_concurrent: int = 0) -> None:
    Handler.jit = jit
    Handler.max_concurrent = max_concurrent
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.serve_forever()
