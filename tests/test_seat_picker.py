"""End-of-sweep seat-picker tests — pure-logic, no docker, no tty.

Covers the multi-select key decoding and toggle loop, the numbered fallback's pick
parser, the sweep-table index space (ok rows only, row order), and the pipeline-side
select contract (winner keeps the use-case tag, manual extras don't).
"""

from __future__ import annotations

from johnny.cli import _pick_seats, _render_sweep_results
from johnny.external import picker


def _r(tp, peak, ok=True):
    return {"point": {"tp": tp, "gpu_memory_util": 0.9, "max_num_seqs": 32,
                      "max_num_batched_tokens": 16384, "max_model_len": 4096},
            "ok": ok, "peak_tok_s": peak, "single_tok_s": peak / 20}


def test_parse_picks_numbers_commas_spaces_dedup():
    assert picker.parse_picks("3", 12) == [3]
    assert picker.parse_picks("1,7,12", 12) == [1, 7, 12]
    assert picker.parse_picks("7 1  12", 12) == [1, 7, 12]
    assert picker.parse_picks("2,2,2", 12) == [2]
    assert picker.parse_picks("a", 3) == [1, 2, 3]
    assert picker.parse_picks("ALL", 3) == [1, 2, 3]


def test_parse_picks_rejects_garbage_and_range():
    for bad in ("", "tp4", "0", "6", "1,9"):
        assert picker.parse_picks(bad, 5) is None


def test_decode_multi_select_keys():
    assert picker._decode(b" ") == "toggle"
    assert picker._decode(b"a") == "all"
    assert picker._decode(b"A") == "all"
    assert picker._decode(b"\x1b[A") == "up"
    assert picker._decode(b"\r") == "enter"
    assert picker._decode(b"q") == "cancel"


def test_multi_select_toggle_loop(monkeypatch):
    """Drive the interactive loop with scripted keys: move to row 2 and toggle it,
    toggle row 0 off (it starts preselected), accept → only row 2 chosen."""
    monkeypatch.setattr(picker, "_interactive_capable", lambda: True)
    keys = iter(["down", "down", "toggle", "up", "up", "toggle", "enter"])
    monkeypatch.setattr(picker, "_read_key", lambda: next(keys))
    got = picker.multi_select(["a", "b", "c"], render=str, preselected={0})
    assert got == [2]


def test_multi_select_cancel_and_all(monkeypatch):
    monkeypatch.setattr(picker, "_interactive_capable", lambda: True)
    monkeypatch.setattr(picker, "_read_key", lambda: "cancel")
    assert picker.multi_select(["a", "b"], render=str) is None
    keys = iter(["all", "enter"])
    monkeypatch.setattr(picker, "_read_key", lambda: next(keys))
    assert picker.multi_select(["a", "b", "c"], render=str) == [0, 1, 2]


def test_numbered_fallback_blank_keeps_preselection(monkeypatch):
    monkeypatch.setattr(picker, "_interactive_capable", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    assert picker.multi_select(["a", "b", "c"], render=str, preselected={2}) == [2]
    monkeypatch.setattr("builtins.input", lambda *_: "1,3")
    assert picker.multi_select(["a", "b", "c"], render=str) == [0, 2]


def test_render_returns_ok_rows_in_row_order():
    results = [_r(1, 100.0), _r(2, 0.0, ok=False), _r(4, 400.0)]
    rows = _render_sweep_results(results, results[2], None)
    assert rows == [results[0], results[2]]  # failed row excluded from the index space


def test_pick_seats_maps_indices_to_results(monkeypatch):
    """_pick_seats returns the picked result dicts; winner is preselected; cancel → None."""
    results = [_r(1, 100.0), _r(2, 0.0, ok=False), _r(2, 200.0), _r(4, 400.0)]
    winner = results[3]
    seen = {}

    def fake_multi(items, render, title="", hint="", preselected=None):
        seen["items"], seen["pre"] = items, preselected
        return [0, 2]
    monkeypatch.setattr(picker, "multi_select", fake_multi)
    state = {"rendered": False}
    got = _pick_seats(results, winner, None, state)
    assert state["rendered"]
    assert seen["items"] == [results[0], results[2], results[3]]
    assert seen["pre"] == {2}  # winner's row in ok-space
    assert got == [results[0], results[3]]

    monkeypatch.setattr(picker, "multi_select", lambda *a, **k: None)
    assert _pick_seats(results, winner, None, {"rendered": False}) is None


def test_cached_count_matches_state_sigs(monkeypatch, tmp_path):
    """--resume confirm: plan points already in state.json count as cached replays."""
    from types import SimpleNamespace

    from johnny import config as C
    from johnny.induct import pipeline, report

    monkeypatch.setattr(C, "get_paths", lambda: SimpleNamespace(state_dir=tmp_path))
    benched = _r(1, 100.0)["point"] | {"gpu_memory_util": 0.92}
    pipeline._save_state("m", {"results": {report._point_sig(benched): {"ok": True}}})
    fresh = benched | {"max_num_seqs": 64}
    assert pipeline.cached_count("m", [benched, fresh]) == 1
    assert pipeline.cached_count("no-state-yet", [benched]) == 0


def test_pipeline_select_contract(monkeypatch):
    """The placement loop honors select(): every chosen seat is written, only the
    winner keeps the use-case tag, distinct points get distinct registry ids."""
    from johnny.induct import report

    written = []
    monkeypatch.setattr(report, "write_placement",
                        lambda mid, a, placement, hw, local_path=None: written.append(placement))

    results = [_r(1, 100.0), _r(2, 200.0), _r(4, 400.0)]
    winner = results[2]
    chosen = [results[0], results[2]]
    hw = type("HW", (), {"fingerprint": "test-fp"})()
    for r in chosen:
        p = report.to_placement("m", r, {"arch": "X"}, hw, "v", "throughput" if r is winner else None)
        report.write_placement("m", {"arch": "X"}, p, hw)
    assert [p["knobs"]["tensor_parallel_size"] for p in written] == [1, 4]
    assert [p["use_case"] for p in written] == [None, "throughput"]
    assert len({p["id"] for p in written}) == 2
