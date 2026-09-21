"""Endpoint mode + the client-suite plumbing in `johnny bench` — no docker, no real seats.

Covers the URL helper every suite now routes through, the final-result-line contract with
the bundled scripts, registry `quality` shaping for hardcode/load/depthprobe, the report
rendering, and `run_endpoint` guard rails + a real hardcode run against a fake server.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from johnny import bench as B


def test_v1_accepts_port_and_urls():
    assert B._v1(8002) == "http://127.0.0.1:8002/v1"
    assert B._v1("8002") == "http://127.0.0.1:8002/v1"
    assert B._v1("http://host:8124") == "http://host:8124/v1"
    assert B._v1("http://host:8124/") == "http://host:8124/v1"
    assert B._v1("http://host:8124/v1") == "http://host:8124/v1"
    assert B._v1("http://host:8124/v1/") == "http://host:8124/v1"


def test_result_line_takes_last_matching_line():
    out = 'noise\nHARDCODE_RESULT {"passed": 1}\nmore\nHARDCODE_RESULT {"passed": 2, "total": 20}\n'
    assert B._result_line(out, "HARDCODE_RESULT") == {"passed": 2, "total": 20}
    assert B._result_line("nothing here", "LOAD_RESULT") is None
    assert B._result_line("LOAD_RESULT {not json", "LOAD_RESULT") is None
    # a tag that merely prefixes another tag's line must not match
    assert B._result_line('LOAD_RESULTS {"x": 1}', "LOAD_RESULT") is None


def test_new_suites_registered_and_endpoint_subset():
    for s in ("hardcode", "load", "depthprobe"):
        assert s in B.SUITES and s in B.ENDPOINT_SUITES
    assert "perf" not in B.ENDPOINT_SUITES and "ctxsafe" not in B.ENDPOINT_SUITES
    assert set(B.ENDPOINT_SUITES) <= set(B.SUITES)


def test_client_suite_quality_shapes_are_compact():
    results = {
        "hardcode": {"ok": True, "passed": 15, "total": 20, "pass_rate_pct": 75.0, "failed": ["h18"],
                     "thinking": False, "limit": None, "samples": "/x", "api_errors": 0},
        "load": {"ok": True, "input_tokens": 2800, "output_tokens": 250, "tokens_per_word": 1.3,
                 "best": {"concurrency": 16, "req_per_s": 2.2, "total_tok_s": 6700},
                 "levels": [{"concurrency": 16, "req_per_s": 2.2, "total_tok_s": 6700, "ttft_ms_p50": 900,
                             "tpot_ms_p50": 44, "e2e_s_p50": 13.5, "e2e_s_p99": 16.3, "failed": 0,
                             "prefill_tok_s": 6100, "completed": 160}]},
        "depthprobe": {"ok": False, "error": "nope"},
    }
    q = B._client_suite_quality(results, "2026-09-21")
    assert q["hardcode"] == {"passed": 15, "total": 20, "pass_rate_pct": 75.0, "failed": ["h18"],
                             "thinking": False, "limit": None, "date": "2026-09-21"}
    assert q["load"]["best"]["req_per_s"] == 2.2 and "prefill_tok_s" not in q["load"]["levels"][0]
    assert "depthprobe" not in q  # failed suites never reach the registry


def test_report_renders_new_suites(tmp_path):
    results = {
        "hardcode": {"ok": True, "passed": 18, "total": 20, "pass_rate_pct": 90.0, "failed": ["h18", "h19"],
                     "api_errors": 0, "mean_completion_tokens": 383, "thinking": False},
        "load": {"ok": True, "input_tokens": 2800, "output_tokens": 250, "tokens_per_word": 1.31,
                 "levels": [{"concurrency": 8, "completed": 80, "failed": 0, "req_per_s": 0.87, "prefill_tok_s": 2440,
                             "output_tok_s": 218, "total_tok_s": 2661, "ttft_ms_p50": 1598, "ttft_ms_p99": 4000,
                             "tpot_ms_p50": 26.8, "e2e_s_p50": 8.1, "e2e_s_p99": 11.0}],
                 "best": {"concurrency": 8, "req_per_s": 0.87, "total_tok_s": 2661}},
        "depthprobe": {"ok": True, "points": [
            {"target_tokens": 2000, "prompt_tokens": 2415, "ttft_s": 1.3, "prefill_tok_s": 1915.6,
             "decode_tok_s": 66.8, "completion_tokens": 64},
            {"target_tokens": 300000, "error": "HTTP 400: context too small"}]},
    }
    text = B.write_report(tmp_path, "coder", "endpoint http://h:1/v1", results).read_text()
    assert "Hard coding set 18/20 (90.0%)" in text and "h18, h19" in text and "±2 tasks is noise" in text
    assert "| 8 | 80 | 0 | 0.87 |" in text and "best: 0.87 req/s" in text
    assert "2415 tok prompt: prefill 1915.6 tok/s" in text and "HTTP 400" in text


def test_run_endpoint_refuses_managed_only_suites(monkeypatch):
    monkeypatch.setattr(B, "load_config", lambda: {})
    res = B.run_endpoint("http://127.0.0.1:9", None, ["perf", "hardcode"], cfg={})
    assert "perf" in res["error"] and "load" in res["error"]
    res = B.run_endpoint("http://127.0.0.1:9", None, ["ctxsafe"], cfg={})
    assert "ctxsafe" in res["error"]


def test_run_endpoint_unreachable_is_a_clean_error():
    res = B.run_endpoint("http://127.0.0.1:9", None, ["hardcode"], cfg={})
    assert "unreachable" in res["error"]


class _Fake(BaseHTTPRequestHandler):
    """Serves /v1/models and answers every chat request with a fixed (wrong) code block."""

    def log_message(self, *a):  # quiet
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "fake-model"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._send({"choices": [{"message": {"content": "```python\nx = 1\n```"}}],
                    "usage": {"completion_tokens": 7}})


@pytest.fixture()
def fake_endpoint():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_run_endpoint_hardcode_end_to_end(fake_endpoint, tmp_path, monkeypatch):
    class _Paths:
        runs_dir = tmp_path

    monkeypatch.setattr(B.C, "get_paths", lambda: _Paths)
    assert B.endpoint_models(fake_endpoint) == ["fake-model"]
    msgs: list[str] = []
    res = B.run_endpoint(fake_endpoint, None, ["hardcode"], cfg={}, limit=2, progress=msgs.append)
    r = res["results"]["hardcode"]
    assert res["registry_updated"] is False and res["placement_id"] is None and res["model_id"] == "fake-model"
    assert r["ok"] and r["total"] == 2 and r["passed"] == 0 and sorted(r["failed"]) == ["h01", "h02"]
    assert any("using fake-model" in m for m in msgs)
    report = (tmp_path / next(p.name for p in tmp_path.iterdir()) / "BENCH_REPORT.md").read_text()
    assert "Hard coding set 0/2" in report and "endpoint " + fake_endpoint in report


def test_run_load_treats_best_null_as_failure(monkeypatch, tmp_path):
    line = 'LOAD_RESULT {"input_tokens":2800,"output_tokens":250,"tokens_per_word":1.35,"levels":[{"concurrency":2,"completed":0,"failed":4}],"best":null}'
    monkeypatch.setattr(B, "_stream_run", lambda cmd, timeout, progress, env=None: (1, "table\n" + line))
    r = B._run_load(8002, "m", tmp_path, {}, {"input_tokens": 2800}, lambda *_: None)
    assert r["ok"] is False and "no request completed" in r["error"]
    good = line.replace('"best":null', '"best":{"concurrency":2,"req_per_s":1.5,"total_tok_s":4000}')
    monkeypatch.setattr(B, "_stream_run", lambda cmd, timeout, progress, env=None: (0, good))
    r = B._run_load(8002, "m", tmp_path, {}, None, lambda *_: None)
    assert r["ok"] and r["best"]["req_per_s"] == 1.5


def test_humaneval_parser_reads_rescued_count():
    out = "HumanEval pass@1: 150/164 = 91.46%\nImport mode: prompt imports restored; rescued by restored imports: 12\n"
    assert B.parse_humaneval_score(out) == {"passed": 150, "total": 164, "pass_at_1_pct": 91.46, "imports_rescued": 12}
    assert "imports_rescued" not in B.parse_humaneval_score("HumanEval pass@1: 1/2 = 50.00%\n")
