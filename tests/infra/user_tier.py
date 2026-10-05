"""Focused helpers for tests that exercise user-tier persistence semantics."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def connect_user_db(path: Path) -> sqlite3.Connection:
    """Open a user database with the row shape used by semantic tests."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def connect_user_tier(path: Path) -> sqlite3.Connection:
    """Open and initialize a user-tier database for a semantic test.

    This deliberately covers only the repeated user-tier setup used by tests;
    tests whose subject is connection configuration or schema initialization
    should continue to open SQLite connections and call the bootstrap directly.
    """
    conn = connect_user_db(path)
    initialize_archive_tier(conn, ArchiveTier.USER)
    return conn


def connect_measured_user_tier(path: Path) -> sqlite3.Connection:
    """Open an initialized user tier through the measured connection creator.

    User-tier writers settle their statements through the creator that
    opened the connection, so a law that drives those writers needs this one.
    """
    conn = connect_measured(path)
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.USER)
    return conn


__all__ = ["connect_measured_user_tier", "connect_user_db", "connect_user_tier"]
