"""Synthetic parent event-ledger shape for derived-schema admission tests."""

import sqlite3
from pathlib import Path


def make_ops_event_schema_stale(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("DROP INDEX idx_daemon_events_idempotency")
        conn.execute("ALTER TABLE daemon_events DROP COLUMN idempotency_key")
        conn.execute("UPDATE schema_identity SET identity = 'synthetic-parent-runtime' WHERE tier = 'ops'")
