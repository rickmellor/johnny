"""systemd backend: a host process johnny owns via a `systemctl --user` unit (no docker)."""
from __future__ import annotations

import subprocess
from unittest import mock

from johnny.backends import get_driver
from johnny.backends.systemd import SystemdDriver

REG = {"models": {"saint-features": {"identity": {}, "placements": [
    {"id": "rtx4080-systemd", "backend": "systemd", "knobs": {"gpu_count": 0},
     "extra": {"unit": "saint-features.service", "port": 8005, "served_model": "nomic-embed", "health": "/health", "device": "RTX 4080"}}]}}}


def _cp(stdout: str, rc: int = 0):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


def test_get_driver_knows_systemd():
    assert isinstance(get_driver("systemd"), SystemdDriver)


def test_runtime_state_reports_active_units_only():
    drv = SystemdDriver()
    with mock.patch("johnny.registry.store.load", return_value=REG), \
         mock.patch("johnny.backends.systemd._systemctl", return_value=_cp("ActiveState=active\nSubState=running\nMainPID=42\n")), \
         mock.patch("johnny.backends.systemd._healthy", return_value=True):
        seats = drv.runtime_state()
    assert len(seats) == 1 and seats[0].name == "saint-features.service"
    assert seats[0].port == 8005 and seats[0].model == "nomic-embed" and seats[0].state == "ready" and seats[0].gpus == []
    assert seats[0].extra["labels"]["johnny.model"] == "saint-features" and seats[0].extra["device"] == "RTX 4080"
    with mock.patch("johnny.registry.store.load", return_value=REG), \
         mock.patch("johnny.backends.systemd._systemctl", return_value=_cp("ActiveState=inactive\nSubState=dead\nMainPID=0\n")):
        assert drv.runtime_state() == []


def test_active_but_unhealthy_is_loading():
    with mock.patch("johnny.registry.store.load", return_value=REG), \
         mock.patch("johnny.backends.systemd._systemctl", return_value=_cp("ActiveState=active\nSubState=running\nMainPID=42\n")), \
         mock.patch("johnny.backends.systemd._healthy", return_value=False):
        assert SystemdDriver().runtime_state()[0].state == "loading"


def test_launch_and_stop_call_systemctl():
    calls = []
    def fake(*args, timeout=20):
        calls.append(args); return _cp("")
    with mock.patch("johnny.backends.systemd._systemctl", side_effect=fake):
        seat = SystemdDriver().launch({"unit": "x.service", "port": 1, "model": "m", "model_id": "m", "placement": "p"})
        SystemdDriver().stop("x.service")
    assert seat.state == "loading" and calls == [("start", "x.service"), ("stop", "x.service")]


def test_one_unit_can_back_several_named_seats():
    reg = {"models": {
        "a": {"placements": [{"id": "p", "backend": "systemd", "extra": {"unit": "u.service", "port": 1, "served_model": "a", "seat_name": "u.service#a"}}]},
        "b": {"placements": [{"id": "p", "backend": "systemd", "extra": {"unit": "u.service", "port": 1, "served_model": "b", "seat_name": "u.service#b"}}]}}}
    calls = []
    def fake(*args, timeout=20):
        calls.append(args); return _cp("ActiveState=active\nSubState=running\nMainPID=1\n")
    with mock.patch("johnny.registry.store.load", return_value=reg), mock.patch("johnny.backends.systemd._systemctl", side_effect=fake), \
         mock.patch("johnny.backends.systemd._healthy", return_value=True):
        names = [s.name for s in SystemdDriver().runtime_state()]
        SystemdDriver().stop("u.service#b")
    assert names == ["u.service#a", "u.service#b"] and calls[-1] == ("stop", "u.service")
