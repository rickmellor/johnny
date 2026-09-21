"""depth_probe.py against a fake streaming OpenAI-compatible server (no real seat touched)."""
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "src/johnny/scripts/depth_probe.py"
MAX_WORDS = 3000  # fake context limit, in whitespace-separated words


class _Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        prompt = body["messages"][0]["content"]
        words = len(prompt.split())
        if words > MAX_WORDS:
            payload = json.dumps({"error": {"message": f"prompt of {words} exceeds context"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj):
            self.wfile.write(b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n")
            self.wfile.flush()

        send({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})
        n = body["max_tokens"]
        for i in range(n):
            time.sleep(0.005)
            send({"choices": [{"index": 0, "delta": {"content": f"w{i} "}}]})
        send({"choices": [], "usage": {"prompt_tokens": words, "completion_tokens": n,
                                       "total_tokens": words + n}})
        send(b"[DONE]")


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    srv.bodies = []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _run(server, depths, *extra):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
         "--model", "fake", "--depths", depths, "--max-tokens", "8", "--timeout", "20", *extra],
        capture_output=True, text=True, timeout=60)
    return r


def _result(stdout):
    last = stdout.strip().splitlines()[-1]
    assert last.startswith("DEPTHPROBE_RESULT ")
    return json.loads(last[len("DEPTHPROBE_RESULT "):])


def test_probe_depths_and_oversize_error(server, tmp_path):
    out = tmp_path / "probe.json"
    r = _run(server, "300,1500,600000", "--out", str(out))
    assert r.returncode == 0, r.stderr
    points = _result(r.stdout)["points"]
    assert [p["target_tokens"] for p in points] == [300, 1500, 600000]
    small, big, over = points
    assert big["prompt_tokens"] > small["prompt_tokens"] > 0
    for p in (small, big):
        assert set(p) == {"target_tokens", "prompt_tokens", "ttft_s", "prefill_tok_s",
                          "decode_tok_s", "completion_tokens"}
        assert p["prefill_tok_s"] > 0 and p["decode_tok_s"] > 0 and p["ttft_s"] > 0
        assert p["completion_tokens"] == 8
    assert set(over) == {"target_tokens", "error"} and "400" in over["error"]
    assert json.loads(out.read_text()) == {"points": points}
    assert len(r.stdout.strip().splitlines()) == 4  # one line per depth + result line

    body = server.bodies[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["cache_prompt"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_same_depth_prompts_are_unique_and_thinking_flag(server):
    r = _run(server, "300,300", "--thinking")
    assert r.returncode == 0, r.stderr
    assert all("error" not in p for p in _result(r.stdout)["points"])
    a, b = (body["messages"][0]["content"] for body in server.bodies)
    assert a.splitlines()[0] != b.splitlines()[0]
    assert a.splitlines()[1:] == b.splitlines()[1:]
    assert all("chat_template_kwargs" not in body for body in server.bodies)
