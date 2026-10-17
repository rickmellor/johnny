"""_wait_ready -> "ready" | "exited" | "timeout", and drivers' alive().

`exited` is the point: a container that already died must not be reported as
`loading` until the wall-clock deadline (AGENTS.md: "up --wait keeps reporting
loading for a container that has already exited")."""
import time as _time

import johnny.backends.vllm as vllm_mod
from johnny.backends.base import Driver
from johnny.backends.vllm import VllmDriver
from johnny.engine import launch


class Clock:
    """Wall clock that only advances when the code under test sleeps."""

    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, secs):
        self.now += secs


class FakeDrv:
    """Scripted alive() answers; the last one repeats once the list runs out."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def alive(self, seat):
        self.calls.append(seat)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def _patch(monkeypatch, clock, probe_answers):
    monkeypatch.setattr(_time, "time", clock.time)
    monkeypatch.setattr(_time, "sleep", clock.sleep)
    answers = list(probe_answers)
    monkeypatch.setattr(launch.probe, "probe_models",
                        lambda port, *a, **kw: answers.pop(0) if len(answers) > 1 else answers[0])


def test_ready_on_first_poll_never_checks_alive(monkeypatch):
    clock = Clock()
    _patch(monkeypatch, clock, [True])
    drv = FakeDrv(True)
    assert launch._wait_ready(8001, 30, drv=drv, seat="johnny-x-8001") == "ready"
    assert drv.calls == [] and clock.now == 0.0


def test_exited_when_probe_fails_and_container_is_gone(monkeypatch):
    clock = Clock()
    _patch(monkeypatch, clock, [False])
    drv = FakeDrv(False)
    assert launch._wait_ready(8001, 30, drv=drv, seat="johnny-x-8001") == "exited"
    assert drv.calls[:1] == ["johnny-x-8001"] and clock.now < 30   # early, not at the deadline


def test_timeout_while_the_container_is_still_running(monkeypatch):
    clock = Clock()
    _patch(monkeypatch, clock, [False])
    drv = FakeDrv(True)
    assert launch._wait_ready(8001, 30, drv=drv, seat="johnny-x-8001") == "timeout"
    assert drv.calls and clock.now >= 30


def test_no_driver_never_calls_alive_and_times_out(monkeypatch):
    clock = Clock()
    _patch(monkeypatch, clock, [False])
    assert launch._wait_ready(8001, 12) == "timeout"
    assert clock.now >= 12


def _fake_run(monkeypatch, rc, out):
    seen = []

    def fake_run(cmd, *args, **kwargs):
        seen.append(cmd if isinstance(cmd, str) else " ".join(cmd))
        return rc, out, ""

    monkeypatch.setattr(vllm_mod, "run", fake_run)
    return seen


def test_vllm_alive_true_for_a_running_container(monkeypatch):
    seen = _fake_run(monkeypatch, 0, "true\n")
    assert VllmDriver().alive("johnny-x-8001") is True
    assert "johnny-x-8001" in seen[0] and ".State.Running" in seen[0]


def test_vllm_alive_false_when_stopped_or_missing(monkeypatch):
    _fake_run(monkeypatch, 0, "false")
    assert VllmDriver().alive("johnny-x-8001") is False
    _fake_run(monkeypatch, 1, "")                                  # no such container
    assert VllmDriver().alive("johnny-x-8001") is False


def test_base_driver_alive_defaults_true():
    assert Driver().alive("johnny-x-8001") is True
