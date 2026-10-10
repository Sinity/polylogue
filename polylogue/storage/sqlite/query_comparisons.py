"""Connection-local SQL adapters for the canonical query comparisons."""

from __future__ import annotations

import sqlite3

from polylogue.core.query_comparisons import path_matches_prefix, prose_contains


def register_query_comparisons(connection: sqlite3.Connection) -> None:
    connection.create_function("pl_path_prefix", 2, path_matches_prefix, deterministic=True)
    connection.create_function("pl_prose_contains", 2, prose_contains, deterministic=True)
