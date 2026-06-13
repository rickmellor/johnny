"""Mutation lock — placement *decisions* must be serialized.

State reads are lock-free (derived from docker), but two concurrent `up`s could
both decide the same GPUs/ports are free and collide. A single flock on the config
dir serializes mutations. POSIX-only (fcntl); on non-POSIX the lock is a no-op
(the vLLM backend is Linux-only anyway, and LM Studio/Ollama manage their own).
"""

from __future__ import annotations

import contextlib
import time

from .. import config as C

try:
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    _HAVE_FCNTL = False


@contextlib.contextmanager
def mutation_lock(timeout: float = 30.0):
    paths = C.get_paths()
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    lockfile = paths.config_dir / ".johnny.lock"
    if not _HAVE_FCNTL:
        yield
        return
    f = open(lockfile, "w")
    deadline = time.time() + timeout
    try:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() > deadline:
                    raise TimeoutError("another johnny mutation holds the lock")
                time.sleep(0.2)
        yield
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()
