"""Insights-dedicated SQLite storage layer.

Insight tables that historically lived in the archive DB are extracted
to a separate ``insights.db`` file in the same archive directory. This
keeps the archive write path lean and allows independent read-model
rebuilds without affecting archive queries.

The insight schema mirrors the archive insight tables (session_profiles,
session_work_events, session_phases, etc.) but lives in its own database
file so that maintenance operations (rebuild, reset, reindex) do not
contend with archive reads.

Schema
------
The insight tables mirror the archive insight DDL defined in
``polylogue.storage.backends.schema_ddl_insight_profiles``,
``schema_ddl_insight_timelines``, and ``schema_ddl_insight_aggregates``.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.schema_ddl_insight_aggregates import (
    SESSION_INSIGHT_AGGREGATE_DDL,
)
from polylogue.storage.sqlite.schema_ddl_insight_profiles import (
    SESSION_INSIGHT_PROFILE_DDL,
)
from polylogue.storage.sqlite.schema_ddl_insight_timelines import (
    SESSION_INSIGHT_TIMELINE_DDL,
)

logger = logging.getLogger(__name__)

INSIGHT_SCHEMA_VERSION = 1

# Full schema for insights.db
INSIGHT_DDL = (
    SESSION_INSIGHT_PROFILE_DDL + "\n\n" + SESSION_INSIGHT_TIMELINE_DDL + "\n\n" + SESSION_INSIGHT_AGGREGATE_DDL
)


class InsightsDB:
    """Dedicated SQLite connection to ``insights.db``.

    Usage
    -----
    .. code-block:: python

        db = InsightsDB(archive_dir / "insights.db")
        db.ensure_schema()
        with db.connect() as conn:
            conn.execute("SELECT 1 FROM session_profiles LIMIT 1")
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    @property
    def db_path(self) -> Path:
        return self._db_path

    @classmethod
    def from_archive_dir(cls, archive_dir: str | Path) -> InsightsDB:
        """Create an InsightsDB rooted next to the archive database.

        The insights database lives at ``<archive_dir>/insights.db``,
        alongside the main ``<archive_dir>/archive.db``.
        """
        return cls(Path(archive_dir) / "insights.db")

    def connect(self) -> sqlite3.Connection:
        """Open a sync connection to the insights database."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        """Create or verify the insights schema."""
        conn = self.connect()
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                conn.executescript(INSIGHT_DDL)
                conn.execute(f"PRAGMA user_version = {INSIGHT_SCHEMA_VERSION}")
                conn.commit()
                logger.debug("Created fresh insights schema v%s", INSIGHT_SCHEMA_VERSION)
            elif version != INSIGHT_SCHEMA_VERSION:
                logger.warning(
                    "Insights DB schema v%s does not match expected v%s. Rebuilding insights schema.",
                    version,
                    INSIGHT_SCHEMA_VERSION,
                )
                _rebuild_schema(conn)
            else:
                logger.debug("Insights DB at schema v%s, no action needed", version)
        finally:
            conn.close()

    def clear(self) -> None:
        """Drop all insight tables and re-create them."""
        conn = self.connect()
        try:
            _rebuild_schema(conn)
        finally:
            conn.close()

    def close(self) -> None:
        """No-op for compatibility; connections are short-lived."""


def _rebuild_schema(conn: sqlite3.Connection) -> None:
    """Drop all insight tables and re-create from DDL."""
    tables = [
        "session_profiles_fts",
        "session_profile_evidence_fts",
        "session_profile_inference_fts",
        "session_profile_enrichment_fts",
        "session_profiles",
        "session_work_events_fts",
        "session_work_events",
        "session_phases",
        "session_threads_fts",
        "session_threads",
        "day_session_summaries",
        "week_session_summaries",
        "session_tag_rollups",
    ]
    for table in tables:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.executescript(INSIGHT_DDL)
    conn.execute(f"PRAGMA user_version = {INSIGHT_SCHEMA_VERSION}")
    conn.commit()
    logger.info("Rebuilt insights schema to v%s", INSIGHT_SCHEMA_VERSION)


__all__ = [
    "INSIGHT_DDL",
    "INSIGHT_SCHEMA_VERSION",
    "InsightsDB",
]
