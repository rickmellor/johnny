"""hardcode suite: references, code extraction, scoring, and the HARDCODE_RESULT contract. No network."""
import importlib.util, json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "hardcode_eval", Path(__file__).parent.parent / "src/johnny/scripts/hardcode_eval.py")
hc = importlib.util.module_from_spec(spec); sys.modules["hardcode_eval"] = hc; spec.loader.exec_module(hc)


def test_tasks_shape():
    assert [t["id"] for t in hc.TASKS] == [f"h{i:02d}" for i in range(1, 25)]
    for t in hc.TASKS:
        assert set(t) == {"id", "prompt", "reference", "tests"}
        assert t["prompt"].endswith("Return only one ```python code block, no explanation.")


def test_self_test_all_references_pass():
    assert hc.self_test() == []
    assert hc.main(["--self-test"]) == 0


def test_extract_longest_python_block():
    text = ("Short:\n```python\nx=1\n```\nFull:\n```python\ndef f():\n    return 42\n```\n"
            "```text\n" + "this non-python block is the longest one here " * 3 + "\n```\n")
    assert hc.extract_code(text) == "def f():\n    return 42\n"


def test_extract_fallbacks():
    assert hc.extract_code("here\n```\ndef g(): pass\n```\n") == "def g(): pass\n"
    assert hc.extract_code("def h(): return 1") == "def h(): return 1"


def test_scoring_correct_and_wrong_candidate():
    task = next(t for t in hc.TASKS if t["id"] == "h16")
    ok, err = hc.run_tests(task["reference"], task["tests"])
    assert ok and err == ""
    ok, err = hc.run_tests("def schedule_meetings(meetings):\n    return len(meetings)", task["tests"])
    assert not ok and "AssertionError" in err
    ok, err = hc.run_tests("while True: pass", task["tests"], timeout=1)
    assert not ok and "timeout" in err


def test_src_variable_is_visible_to_tests():
    assert hc.run_tests("x = 1", "assert 'x = 1' in __SRC__")[0]


class _FakeOpenAI(BaseHTTPRequestHandler):
    """Answers h01 with its reference solution and everything else with garbage."""
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, self.headers.get("Authorization"), body))
        prompt = body["messages"][0]["content"]
        h01 = hc.TASKS[0]
        answer = f"Sure:\n```python\n{h01['reference']}\n```" if prompt == h01["prompt"] else "I cannot help with that."
        payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": answer}}],
                              "usage": {"completion_tokens": 100}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


def _last_result(out: str) -> dict:
    last = out.rstrip("\n").splitlines()[-1]
    assert last.startswith("HARDCODE_RESULT ")
    return json.loads(last[len("HARDCODE_RESULT "):])


def test_result_line_contract(capsys, tmp_path):
    _FakeOpenAI.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out_path = tmp_path / "detail.json"
    try:
        rc = hc.main(["--base-url", f"http://127.0.0.1:{srv.server_port}/v1", "--model", "fake",
                      "--limit", "2", "--out", str(out_path)])
    finally:
        srv.shutdown(); srv.server_close()
    assert rc == 0
    res = _last_result(capsys.readouterr().out)
    assert res == {"passed": 1, "total": 2, "pass_rate_pct": 50.0, "failed": ["h02"], "api_errors": 0,
                   "mean_completion_tokens": 100, "thinking": False}
    path, auth, body = _FakeOpenAI.seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer EMPTY"
    assert body["temperature"] == 0 and body["max_tokens"] == 2048
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    detail = json.loads(out_path.read_text())["tasks"]
    assert detail["h01"]["ok"] and not detail["h02"]["ok"]
    assert detail["h02"]["answer"] == "I cannot help with that." and detail["h02"]["error"]


def test_api_errors_fail_tasks_not_the_run(capsys):
    # nothing listens on port 9 (discard) — connection refused
    rc = hc.main(["--base-url", "http://127.0.0.1:9/v1", "--model", "fake", "--limit", "2",
                  "--thinking", "--timeout", "5"])
    assert rc == 0
    res = _last_result(capsys.readouterr().out)
    assert (res["passed"], res["total"], res["api_errors"], res["thinking"]) == (0, 2, 2, True)
    assert res["failed"] == ["h01", "h02"] and res["mean_completion_tokens"] == 0
