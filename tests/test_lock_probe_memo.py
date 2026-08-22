"""The mutation lock must be a probe-memo boundary.

Regression for 2026-08-22: `profile up daily` launched coder (TP2) and chat (TP2)
0.5s apart and both landed on GPUs 0,1 — the second `launch.up` planned against
the 2s-TTL memoized `docker ps` taken before the first container existed.
"""

from __future__ import annotations

from unittest import mock

from johnny.runtime import lock, probe


def _memo_docker_ps(rows):
    probe.invalidate()
    calls = {"n": 0}

    def fake_run(argv, timeout=10):
        calls["n"] += 1
        return 0, "\n".join(__import__("json").dumps(r) for r in rows), ""

    return calls, fake_run


def test_docker_ps_is_memoized_within_ttl(tmp_path):
    calls, fake_run = _memo_docker_ps([{"Names": "a"}])
    with mock.patch.object(probe, "run", fake_run):
        assert probe.docker_ps() == [{"Names": "a"}]
        assert probe.docker_ps() == [{"Names": "a"}]
    assert calls["n"] == 1


def test_mutation_lock_invalidates_memo_on_enter_and_exit(tmp_path):
    calls, fake_run = _memo_docker_ps([{"Names": "before"}])
    paths = mock.Mock(config_dir=tmp_path)
    with mock.patch.object(probe, "run", fake_run), \
         mock.patch.object(lock.C, "get_paths", return_value=paths):
        probe.docker_ps()                      # warm the memo
        assert calls["n"] == 1
        with lock.mutation_lock():
            probe.docker_ps()                  # enter → re-probe (sibling may have mutated)
            assert calls["n"] == 2
            probe.docker_ps()                  # still memoized inside the critical section
            assert calls["n"] == 2
        probe.docker_ps()                      # exit → re-probe (we just mutated)
        assert calls["n"] == 3
