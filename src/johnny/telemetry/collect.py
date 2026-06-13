"""Telemetry store + ingest (§3.7).

A small SQLite the reaper and pollers share: last-activity (idle detection), pins
(reaper exemption — ephemeral pins live here because docker labels are immutable
post-create), metric samples, and load timings (eta_s). The ingest spool is the
daemon-optional proxy-push path: providers append JSONL records to
$XDG_STATE_HOME/johnny/ingest/; johnny rotates-before-read so a mid-append record
is never split and nothing is double-ingested.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from .. import config as C
from . import schema


def now() -> int:
    return int(time.time())


def connect() -> sqlite3.Connection:
    paths = C.get_paths()
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(paths.db_file)
    schema.apply(conn)
    return conn


# --- activity (reaper idle detection) ---
def record_activity(seat: str, ts: int | None = None, requests: int | None = None) -> None:
    ts = ts if ts is not None else now()
    with connect() as conn:
        conn.execute(
            "INSERT INTO activity(seat, last_ts, requests) VALUES(?,?,?) "
            "ON CONFLICT(seat) DO UPDATE SET last_ts=excluded.last_ts, "
            "requests=COALESCE(excluded.requests, activity.requests)",
            (seat, ts, requests),
        )


def last_activity(seat: str) -> int | None:
    with connect() as conn:
        row = conn.execute("SELECT last_ts FROM activity WHERE seat=?", (seat,)).fetchone()
    return row[0] if row else None


# --- pins (reaper exemption) ---
def add_pin(seat: str, ttl_s: int | None = None) -> None:
    expires = (now() + ttl_s) if ttl_s else None
    with connect() as conn:
        conn.execute(
            "INSERT INTO pins(seat, expires_ts) VALUES(?,?) "
            "ON CONFLICT(seat) DO UPDATE SET expires_ts=excluded.expires_ts",
            (seat, expires),
        )


def remove_pin(seat: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM pins WHERE seat=?", (seat,))


def is_pinned(seat: str, at: int | None = None) -> bool:
    at = at if at is not None else now()
    with connect() as conn:
        row = conn.execute("SELECT expires_ts FROM pins WHERE seat=?", (seat,)).fetchone()
    if not row:
        return False
    expires = row[0]
    return expires is None or expires > at


def list_pins() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT seat, expires_ts FROM pins").fetchall()
    return [{"seat": s, "expires_ts": e} for s, e in rows]


# --- samples + load timings ---
def record_sample(seat: str, sample: dict, ts: int | None = None) -> None:
    ts = ts if ts is not None else now()
    with connect() as conn:
        conn.execute(
            "INSERT INTO samples(ts, seat, source, gen_tok_s, prompt_tok_s, ttft_ms, running, waiting, ctx_used, kv_util)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                ts, seat, sample.get("source", "engine"),
                sample.get("gen_tok_s"), sample.get("prompt_tok_s"), sample.get("ttft_ms"),
                sample.get("running"), sample.get("waiting"), sample.get("ctx_used"), sample.get("kv_util"),
            ),
        )


def record_load_event(seat: str, model: str, placement: str, started_ts: int, ready_ts: int | None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO load_events(seat, model, placement, started_ts, ready_ts) VALUES(?,?,?,?,?)",
            (seat, model, placement, started_ts, ready_ts),
        )


def rollup(seat: str | None = None, since_s: int | None = None) -> list[dict]:
    """Aggregate metric samples per seat (monitoring trends, §3.8)."""
    since = (now() - since_s) if since_s else 0
    q = ("SELECT seat, COUNT(*), AVG(gen_tok_s), MAX(gen_tok_s), AVG(ttft_ms), MAX(running) "
         "FROM samples WHERE ts>=?")
    args: list = [since]
    if seat:
        q += " AND seat=?"
        args.append(seat)
    q += " GROUP BY seat"
    with connect() as conn:
        rows = conn.execute(q, args).fetchall()
    return [
        {"seat": r[0], "samples": r[1], "avg_gen_tok_s": r[2], "max_gen_tok_s": r[3],
         "avg_ttft_ms": r[4], "peak_running": r[5]}
        for r in rows
    ]


def cold_start_estimate(model: str, placement: str | None = None) -> float | None:
    """Median historical cold-start seconds for a model/placement, or None."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT ready_ts - started_ts FROM load_events "
            "WHERE model=? AND ready_ts IS NOT NULL AND (? IS NULL OR placement=?)",
            (model, placement, placement),
        ).fetchall()
    durs = sorted(d[0] for d in rows if d[0] and d[0] > 0)
    if not durs:
        return None
    return float(durs[len(durs) // 2])


# --- ingest spool (proxy-push, rotate-before-read) ---
def ingest_spool() -> int:
    """Consume pending JSONL records from the ingest dir. Returns count ingested.

    Rotate-before-read: each spool file is renamed before being parsed, so a
    concurrent appender's in-flight record is never split and nothing is read twice.
    """
    paths = C.get_paths()
    ingest_dir = paths.ingest_dir
    if not ingest_dir.exists():
        return 0
    count = 0
    for f in sorted(ingest_dir.glob("*.jsonl")):
        if f.name.endswith(".consuming"):
            continue
        rotated = f.with_suffix(f.suffix + f".consuming.{os.getpid()}")
        try:
            f.rename(rotated)
        except OSError:
            continue
        try:
            for line in rotated.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seat = rec.get("seat") or rec.get("johnny_seat")
                if not seat:
                    continue
                rec["source"] = "proxy"
                record_sample(seat, rec, ts=rec.get("ts"))
                record_activity(seat, ts=rec.get("ts"))
                count += 1
        finally:
            rotated.unlink(missing_ok=True)
    return count
