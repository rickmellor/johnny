"""Telemetry SQLite schema (§3.7).

Small, single-DB store the poller/reaper share. Persisted from P3 because idle
detection *requires* last-activity history. `PRAGMA user_version` carries the
schema version for forward migration (separate from the YAML files' schema_version).
"""

from __future__ import annotations

SCHEMA_VERSION = 1

DDL = [
    # last-activity per seat (drives the reaper).
    "CREATE TABLE IF NOT EXISTS activity ("
    " seat TEXT PRIMARY KEY, last_ts INTEGER NOT NULL, requests INTEGER DEFAULT 0)",
    # ephemeral pins (reaper exemption). NULL expires_ts = pinned until unpinned.
    "CREATE TABLE IF NOT EXISTS pins (seat TEXT PRIMARY KEY, expires_ts INTEGER)",
    # normalized metric samples, source-tagged (engine | proxy | derived).
    "CREATE TABLE IF NOT EXISTS samples ("
    " ts INTEGER NOT NULL, seat TEXT NOT NULL, source TEXT NOT NULL,"
    " gen_tok_s REAL, prompt_tok_s REAL, ttft_ms REAL,"
    " running INTEGER, waiting INTEGER, ctx_used INTEGER, kv_util REAL)",
    "CREATE INDEX IF NOT EXISTS idx_samples_seat_ts ON samples(seat, ts)",
    # load timings → eta_s estimates for resolve.
    "CREATE TABLE IF NOT EXISTS load_events ("
    " seat TEXT, model TEXT, placement TEXT, started_ts INTEGER, ready_ts INTEGER)",
]


def apply(conn) -> None:
    cur = conn.cursor()
    for stmt in DDL:
        cur.execute(stmt)
    cur.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
