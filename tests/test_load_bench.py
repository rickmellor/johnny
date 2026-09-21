"""load_bench: calibration, closed-loop sweep, prompt uniqueness, failure handling, percentiles.

Everything runs against a fake OpenAI-compatible server started in-process -- never a real seat."""
import importlib.util
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "load_bench", Path(__file__).parent.parent / "src/johnny/scripts/load_bench.py")
lb = importlib.util.module_from_spec(spec); sys.modules["load_bench"] = lb; spec.loader.exec_module(lb)


def fake_tokens(prompt: str) -> int:
    return round(len(prompt.split()) * 1.3)


class FakeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, always_500: bool = False):
        super().__init__(("127.0.0.1", 0), Handler)
        self.always_500 = always_500
        self.lock = threading.Lock()
        self.prompts: list[tuple[bool, str]] = []   # (stream, prompt)
        self.in_flight = 0
        self.max_in_flight = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"   # close-delimited bodies: no chunked encoding needed for SSE

    def log_message(self, *args):  # silence
        pass

    def _send(self, code: int, ctype: str, body: bytes = b"") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_POST(self):
        srv: FakeServer = self.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if srv.always_500:
            return self._send(500, "application/json", b'{"error":{"message":"boom"}}')
        if self.path == "/v1/completions":
            chat, prompt = False, body["prompt"]
        elif self.path == "/v1/chat/completions":
            chat, prompt = True, body["messages"][0]["content"]
        else:
            return self._send(404, "application/json", b"{}")
        stream, n = bool(body.get("stream")), int(body["max_tokens"])
        usage = {"prompt_tokens": fake_tokens(prompt), "completion_tokens": n,
                 "total_tokens": fake_tokens(prompt) + n}
        with srv.lock:
            srv.prompts.append((stream, prompt))
            srv.in_flight += 1
            srv.max_in_flight = max(srv.max_in_flight, srv.in_flight)
        try:
            if not stream:
                choice = {"message": {"role": "assistant", "content": "x"}} if chat else {"text": "x"}
                return self._send(200, "application/json",
                                  json.dumps({"choices": [dict(choice, index=0)], "usage": usage}).encode())
            self._send(200, "text/event-stream")
            time.sleep(0.01)  # "prefill"
            if chat:  # role-only first chunk, like real servers: must not count as the first token
                self._event({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})
            for _ in range(n):
                time.sleep(0.002)
                self._event({"choices": [{"index": 0, "delta": {"content": "tok "}} if chat
                                         else {"index": 0, "text": "tok "}]})
            self.wfile.write(b": keep-alive comment\n\n")
            self._event({"choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
        finally:
            with srv.lock:
                srv.in_flight -= 1

    def _event(self, obj) -> None:
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()


@pytest.fixture
def server():
    yield from _serve(FakeServer())


@pytest.fixture
def broken_server():
    yield from _serve(FakeServer(always_500=True))


def _serve(srv):
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown(); srv.server_close()


def run_cli(capsys, *argv) -> tuple[int, dict, str]:
    code = lb.main(list(argv))
    out = capsys.readouterr().out
    last = out.strip().splitlines()[-1]
    assert last.startswith("LOAD_RESULT ")
    return code, json.loads(last[len("LOAD_RESULT "):]), out


def test_calibration_and_prompt_sizing(server):
    target = lb.Target(server.base_url, "fake", "completions", "EMPTY", 10)
    rng = random.Random(1)
    tpw, _note, calibrated = lb.calibrate(target, rng, 2800, "words")
    assert calibrated and tpw == pytest.approx(1.3, abs=0.01)
    assert server.prompts[0][0] is False  # calibration is the one non-streamed request
    for wanted in (200, 2800):
        prompt = lb.build_prompt(rng, lb.words_for_tokens(wanted, tpw), "words", 1)
        assert abs(fake_tokens(prompt) - wanted) / wanted < 0.05
        assert abs(fake_tokens(lb.build_prompt(rng, lb.words_for_tokens(wanted, tpw), "random", 2)) - wanted) / wanted < 0.05


def test_small_sweep(server, capsys, tmp_path):
    out_path = tmp_path / "full.json"
    code, result, out = run_cli(
        capsys, "--base-url", server.base_url, "--model", "fake", "--concurrency", "2,4",
        "--requests-per-level", "8", "--input-tokens", "200", "--output-tokens", "16",
        "--warmup", "2", "--seed", "7", "--out", str(out_path))
    assert code == 0
    assert result["input_tokens"] == 200 and result["output_tokens"] == 16
    assert result["tokens_per_word"] == pytest.approx(1.3, abs=0.02)
    assert [lv["concurrency"] for lv in result["levels"]] == [2, 4]
    for lv in result["levels"]:
        assert lv["completed"] == 8 and lv["failed"] == 0
        assert lv["req_per_s"] > 0 and lv["output_tok_s"] > 0
        assert lv["total_tok_s"] == pytest.approx(lv["prefill_tok_s"] + lv["output_tok_s"], abs=0.2)
        assert 0 < lv["ttft_ms_p50"] / 1000 < lv["e2e_s_p50"]
        assert lv["ttft_ms_p50"] <= lv["ttft_ms_p90"] <= lv["ttft_ms_p99"]
        assert lv["tpot_ms_p50"] > 0
        assert abs(lv["prefill_tok_s"] * lv["duration_s"] / 8 - 200) / 200 < 0.05  # server-counted prompt size
    best = max(result["levels"], key=lambda lv: lv["req_per_s"])
    assert result["best"] == {k: best[k] for k in ("concurrency", "req_per_s", "total_tok_s")}
    assert server.max_in_flight <= 4                      # closed loop never exceeds the level
    assert len(server.prompts) == 1 + 2 + 8 + 8           # calibration + warmup + two levels
    full = json.loads(out_path.read_text())
    assert [len(lv["requests"]) for lv in full["requests"]] == [8, 8]
    assert "conc" in out and "req/s" in out


def test_chat_endpoint(server, capsys):
    code, result, _ = run_cli(
        capsys, "--base-url", server.base_url + "/", "--model", "fake", "--endpoint-kind", "chat",
        "--concurrency", "3", "--requests-per-level", "6", "--input-tokens", "100",
        "--output-tokens", "8", "--warmup", "0")
    assert code == 0 and result["levels"][0]["completed"] == 6
    assert result["levels"][0]["ttft_ms_p50"] >= 10      # the empty role chunk did not count as first token


def test_prompts_have_distinct_first_lines(server, capsys):
    run_cli(capsys, "--base-url", server.base_url, "--model", "fake", "--concurrency", "4",
            "--requests-per-level", "12", "--input-tokens", "120", "--output-tokens", "4", "--warmup", "2")
    first_lines = [prompt.splitlines()[0] for _stream, prompt in server.prompts]
    assert len(first_lines) == 15 and len(set(first_lines)) == 15
    assert len({line.split()[0] for line in first_lines}) == 15  # they differ from the very first word


def test_all_requests_failing_is_a_clean_exit_1(broken_server, capsys):
    code, result, out = run_cli(
        capsys, "--base-url", broken_server.base_url, "--model", "fake", "--concurrency", "2,4",
        "--requests-per-level", "8", "--input-tokens", "200", "--output-tokens", "16")
    assert code == 1
    assert len(result["levels"]) == 1                     # sweep stopped after the first level
    level = result["levels"][0]
    assert level["completed"] == 0 and 4 < level["failed"] <= 8
    assert level["req_per_s"] == 0 and level["ttft_ms_p50"] is None
    assert result["best"] is None
    assert "FALLING BACK" in out and "HTTP 500" in out and "Traceback" not in out


def test_unreachable_server_is_a_clean_exit_1(capsys):
    code, result, _ = run_cli(capsys, "--base-url", "http://127.0.0.1:9/v1", "--model", "fake",
                              "--concurrency", "2", "--requests-per-level", "4", "--timeout", "2", "--warmup", "0")
    assert code == 1 and result["levels"][0]["completed"] == 0


def test_usage_errors_exit_2():
    for argv in (["--model", "m"], ["--base-url", "http://x/v1", "--model", "m", "--concurrency", "0,x"]):
        with pytest.raises(SystemExit) as exc:
            lb.main(argv)
        assert exc.value.code == 2


def test_percentile():
    data = [15, 20, 35, 40, 50]
    assert lb.percentile(data, 0) == 15 and lb.percentile(data, 100) == 50
    assert lb.percentile(data, 50) == 35
    assert lb.percentile(data, 40) == pytest.approx(29.0)
    assert lb.percentile(list(range(1, 101)), 90) == pytest.approx(90.1)
    assert lb.percentile(list(range(1, 101)), 99) == pytest.approx(99.01)
    assert lb.percentile([7.5], 99) == 7.5
    assert lb.percentile(reversed(data), 50) == 35        # unsorted / iterator input
    assert lb.percentile([], 50) is None


def test_sse_parsing_is_tolerant():
    lines = [b"\n", b": comment\n", b"event: message\n", b"data: not json\n", b"data:\n",
             b'data: {"choices": []}\n', b'data:{"choices":[{"text":"a"}]}\r\n',
             b'data: {"choices":[{"delta":{"content":"b"}}]}\n', b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
             b"data: [DONE]\n", b'data: {"choices":[{"text":"after done"}]}\n']
    assert [lb.delta_text(e) for e in lb.iter_sse(lines)] == ["", "a", "b", ""]
