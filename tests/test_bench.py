"""`johnny bench` pure-logic tests — no docker, no network.

Covers target resolution (model id / placement id / substring), the placement→point
inversion both GPU and CPU, arc output parsing, and the registry score writeback.
"""

from __future__ import annotations

from johnny import bench as B


def _placement(pid, tp=2, cpu=False):
    if cpu:
        return {"id": pid, "backend": "vllm",
                "knobs": {"max_model_len": 8192, "max_num_seqs": 8, "max_num_batched_tokens": 4096},
                "extra": {"device": "cpu", "cpuset": "0-15", "runner": "pooling"}}
    return {"id": pid, "backend": "vllm",
            "knobs": {"gpu_count": tp, "tensor_parallel_size": tp, "quant": "awq",
                      "max_model_len": 262144, "gpu_memory_util": 0.92, "max_num_seqs": 32,
                      "max_num_batched_tokens": 16384, "kv_cache_dtype": "auto",
                      "mtp": {"enabled": False}},
            "extra": {"tool_call_parser": "qwen3_xml", "reasoning_parser": "qwen3"},
            "perf": {"peak_tok_s": 1.0, "single_stream_tok_s": 2.0}}


def _reg():
    return {"models": {
        "Ornith-1.0-9B-AWQ-FP8": {"identity": {"repo_id": "cyankiwi/Ornith-1.0-9B-AWQ-FP8"},
                                  "placements": [_placement("induct-tp2-gmu0.92"), _placement("induct-tp4-gmu0.92", tp=4)]},
        "gemma-embed": {"identity": {}, "placements": [_placement("induct-cpu-0-15", cpu=True)]},
    }}


def test_resolve_model_id_returns_all_placements():
    cands = B.resolve_target(_reg(), "Ornith-1.0-9B-AWQ-FP8")
    assert [p["id"] for _, p in cands] == ["induct-tp2-gmu0.92", "induct-tp4-gmu0.92"]


def test_resolve_placement_exact_and_substring():
    assert [p["id"] for _, p in B.resolve_target(_reg(), "induct-tp4-gmu0.92")] == ["induct-tp4-gmu0.92"]
    assert [p["id"] for _, p in B.resolve_target(_reg(), "tp4-gmu")] == ["induct-tp4-gmu0.92"]
    # model-substring fallback
    assert {mid for mid, _ in B.resolve_target(_reg(), "Ornith")} == {"Ornith-1.0-9B-AWQ-FP8"}
    assert B.resolve_target(_reg(), "nope-nothing") == []


def test_point_from_placement_gpu_and_cpu():
    p = B.point_from_placement(_placement("x", tp=4))
    assert p["tp"] == 4 and p["gpu_memory_util"] == 0.92 and p["kv_cache_dtype"] == "auto"
    assert p["embeddings"] is False
    c = B.point_from_placement(_placement("y", cpu=True))
    assert c["device"] == "cpu" and c["cpuset"] == "0-15" and c["embeddings"] is True


def test_parse_arc_output():
    out = ("...\nAccuracy:      1114/1172 = 95.05%\nNo extraction: 12  (1.0%)\n"
           "API errors:    3\nElapsed:       1234s\n")
    r = B.parse_arc_output(out)
    assert r == {"accuracy_pct": 95.05, "correct": 1114, "total": 1172,
                 "no_extraction": 12, "api_errors": 3}
    assert B.parse_arc_output("no summary here") is None


def test_write_scores_updates_placement(monkeypatch):
    reg = _reg()
    saved = {}
    monkeypatch.setattr(B.store, "load", lambda: reg)
    monkeypatch.setattr(B.store, "save", lambda r: saved.update(r))
    ok = B.write_scores("Ornith-1.0-9B-AWQ-FP8", "induct-tp2-gmu0.92",
                        perf={"peak_tok_s": 1826.6, "single_tok_s": 73.1},
                        quality={"arc": {"accuracy_pct": 94.2, "date": "2026-07-18"}})
    assert ok and saved
    p = reg["models"]["Ornith-1.0-9B-AWQ-FP8"]["placements"][0]
    assert p["perf"] == {"peak_tok_s": 1826.6, "single_stream_tok_s": 73.1}
    assert p["quality"]["arc"]["accuracy_pct"] == 94.2
    # other placement untouched; unknown id → False, no save
    assert "quality" not in reg["models"]["Ornith-1.0-9B-AWQ-FP8"]["placements"][1]
    assert B.write_scores("Ornith-1.0-9B-AWQ-FP8", "no-such-id", perf={"peak_tok_s": 1}) is False


def test_write_report(tmp_path):
    results = {"perf": {"ok": True, "peak_tok_s": 1826.6, "single_tok_s": 73.1,
                        "kv_cache_tokens": 1.19e6, "max_concurrency": 4.55},
               "arc": {"ok": False, "error": "missing eval deps: openai"}}
    path = B.write_report(tmp_path, "m", "induct-tp2", results)
    text = path.read_text()
    assert "peak 1826.6 tok/s" in text and "KV 1.19M tok" in text
    assert "failed: missing eval deps" in text
