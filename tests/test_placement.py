"""Placement + launch seat-identity tests — the scale-out (12b-quad) fixes.

Pure-logic: no docker. Covers port-aware seat lookup (same model, many seats),
and the --force GPU fill that must never ship an empty visibility mask.
"""

from __future__ import annotations

from types import SimpleNamespace

from johnny.engine import launch
from johnny.engine.placement import assign_gpus, fill_gpus_forced, free_gpus
from johnny.hardware.detect import GPU, GpuGroup, Hardware


def _box(ngpu=4, vram=32.0):
    gpus = [GPU(i, "R9700", vram, "gfx1201", "amd") for i in range(ngpu)]
    groups = [GpuGroup("amd", "gfx1201", vram, ngpu, list(range(ngpu)),
                       ["fp8", "bf16", "fp16"], f"amd-gfx1201-{int(vram)}g")]
    return Hardware("amd", gpus, groups, True, False, vram * ngpu, 251.0,
                    ["fp8", "bf16", "fp16"], "probe", f"{ngpu}xamd-gfx1201-{int(vram)}g")


def _seat(model, port, gpus):
    return SimpleNamespace(name=f"johnny-{model}-{port}", model=model, port=port,
                           gpus=gpus, extra={"labels": {"johnny.model": model}})


def test_find_seat_port_scoped():
    seats = [_seat("gemma-12b", 8000, [0]), _seat("gemma-12b", 8001, [1])]
    assert launch._find_seat(seats, "gemma-12b", port=8001).port == 8001
    assert launch._find_seat(seats, "gemma-12b", port=8002) is None
    # without a port, model-keyed first match (plain `johnny up` semantics)
    assert launch._find_seat(seats, "gemma-12b").port == 8000


def test_sequential_scaleout_assigns_distinct_gpus():
    hw = _box()
    seats = []
    for port in (8000, 8001, 8002, 8003):
        free = free_gpus(hw, seats)
        gpus = assign_gpus(1, hw, free)
        seats.append(_seat("gemma-12b", port, gpus))
    assert [s.gpus for s in seats] == [[0], [1], [2], [3]]


def test_fill_gpus_forced_never_empty_and_prefers_least_loaded():
    hw = _box()
    seats = [_seat("a", 8000, [0]), _seat("b", 8001, [0]), _seat("c", 8002, [1])]
    # GPUs 2,3 free → normal path; forced fill only tops up a short assignment
    assert fill_gpus_forced(1, hw, seats, [2]) == [2]
    # nothing free at all: pick the least-subscribed busy GPUs, deterministic
    busy = [_seat(f"m{i}", 8000 + i, [i]) for i in range(4)]
    assert fill_gpus_forced(1, hw, busy, []) == [0]
    assert fill_gpus_forced(2, hw, busy + [_seat("x", 8010, [0])], []) == [1, 2]


def test_fill_gpus_forced_tops_up_partial_tp_assignment():
    hw = _box()
    seats = [_seat("a", 8000, [0]), _seat("b", 8001, [1]), _seat("c", 8002, [2])]
    # TP=2 with one free GPU: keep the free one, add the least-loaded busy one
    assert fill_gpus_forced(2, hw, seats, [3]) == [0, 3]
