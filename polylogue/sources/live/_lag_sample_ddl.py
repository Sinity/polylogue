"""Shared DDL for ``live_cursor_lag_sample`` (#1753).

Both :mod:`polylogue.sources.live.cursor` (creates the table on daemon startup)
and :mod:`polylogue.daemon.cursor_lag_baseline` (owns the substrate reads/writes)
need this DDL.  A direct cross-import between those modules creates a circular
dependency through ``daemon/__init__``, so the schema fragment lives here in a
leaf module with no intra-package imports.
"""

from __future__ import annotations

_LAG_SAMPLE_DDL = """
CREATE TABLE IF NOT EXISTS live_cursor_lag_sample (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    family TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    max_lag_s REAL NOT NULL,
    stuck_file_count INTEGER NOT NULL,
    p50_lag_s REAL NOT NULL,
    p95_lag_s REAL NOT NULL
)
"""

_LAG_SAMPLE_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_lag_sample_family_time
ON live_cursor_lag_sample(family, observed_at DESC)
"""

__all__ = ["_LAG_SAMPLE_DDL", "_LAG_SAMPLE_INDEX_DDL"]
