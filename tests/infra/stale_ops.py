"""Synthetic parent event-ledger shape for derived-schema admission tests."""

import sqlite3
from pathlib import Path


def make_ops_event_schema_stale(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("DROP INDEX idx_daemon_events_idempotency")
        conn.execute("ALTER TABLE daemon_events DROP COLUMN idempotency_key")
        conn.execute("UPDATE schema_identity SET identity = 'synthetic-parent-runtime' WHERE tier = 'ops'")


def durable_sql_inventory(root: Path) -> dict[str, tuple[str, ...]]:
    """Retain durable rows, schema, and version independent of journal headers."""
    inventory = {}
    for tier in ("source", "user", "audit"):
        path = root / f"{tier}.db"
        with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as conn:
            inventory[tier] = (str(conn.execute("PRAGMA user_version").fetchone()[0]), *conn.iterdump())
    return inventory


def seed_custody_files(root: Path) -> dict[str, bytes]:
    """Neutral file custody outside disposable ops and its SQLite sidecars."""
    files = {
        "blob/synthetic-custody": b"synthetic blob evidence",
        ".index-generations/synthetic-custody": b"synthetic generation evidence",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return files


def custody_file_inventory(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for directory in (root / "blob", root / ".index-generations")
        for path in directory.rglob("*")
        if path.is_file()
    }
