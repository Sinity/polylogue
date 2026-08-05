"""Optional steady-state ingest SLO samples in the disposable ops tier."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.enums import SloSampleLabel
from polylogue.storage.sqlite.archive_tiers.ops import SLO_SAMPLES_DDL
from polylogue.storage.sqlite.connection_profile import open_daemon_connection, open_readonly_connection

SLO_SAMPLE_RETENTION_MS = 30 * 24 * 60 * 60 * 1000
SLO_SAMPLE_ROW_CAP = 20_000


@dataclass(frozen=True, slots=True)
class ArchiveSloSample:
    """One bounded optional steady-state telemetry sample."""

    sample_id: str
    label: str
    scope: str
    value: float
    observed_at_ms: int
    window_start_ms: int | None
    window_end_ms: int | None
    metadata: dict[str, object]


def _ensure_slo_samples_table(conn: sqlite3.Connection) -> None:
    """Self-heal the optional SLO table without changing OPS_SCHEMA_VERSION."""
    conn.executescript(SLO_SAMPLES_DDL)


def _prune_slo_samples(conn: sqlite3.Connection, *, now_ms: int) -> None:
    conn.execute("DELETE FROM slo_samples WHERE observed_at_ms < ?", (now_ms - SLO_SAMPLE_RETENTION_MS,))
    row_count = int(conn.execute("SELECT COUNT(*) FROM slo_samples").fetchone()[0])
    if row_count > SLO_SAMPLE_ROW_CAP:
        conn.execute(
            """
            DELETE FROM slo_samples WHERE sample_id IN (
                SELECT sample_id FROM slo_samples
                ORDER BY observed_at_ms ASC LIMIT ?
            )
            """,
            (row_count - SLO_SAMPLE_ROW_CAP,),
        )


def record_slo_sample(
    ops_db: Path,
    *,
    label: SloSampleLabel | str,
    value: float,
    observed_at_ms: int,
    scope: str = "archive",
    window_start_ms: int | None = None,
    window_end_ms: int | None = None,
    metadata: dict[str, object] | None = None,
    sample_id: str | None = None,
) -> str:
    """Append one SLO sample, pruning by age and a hard row cap."""
    resolved_label = label if isinstance(label, SloSampleLabel) else SloSampleLabel(label)
    sample_id = sample_id or str(uuid.uuid4())
    with open_daemon_connection(ops_db) as conn:
        _ensure_slo_samples_table(conn)
        with conn:
            conn.execute(
                """
                INSERT INTO slo_samples(
                    sample_id, label, scope, value, observed_at_ms,
                    window_start_ms, window_end_ms, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    resolved_label.value,
                    scope,
                    float(value),
                    int(observed_at_ms),
                    window_start_ms,
                    window_end_ms,
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            _prune_slo_samples(conn, now_ms=observed_at_ms)
    return sample_id


def list_slo_samples(
    ops_db: Path,
    *,
    label: SloSampleLabel | str | None = None,
    scope: str | None = None,
    since_ms: int | None = None,
    limit: int = 1000,
) -> tuple[ArchiveSloSample, ...]:
    """Read optional SLO samples newest-first, returning an empty cold start."""
    with open_readonly_connection(ops_db) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'slo_samples' LIMIT 1"
        ).fetchone()
        if exists is None:
            return ()
        clauses: list[str] = []
        params: list[object] = []
        if label is not None:
            resolved_label = label if isinstance(label, SloSampleLabel) else SloSampleLabel(label)
            clauses.append("label = ?")
            params.append(resolved_label.value)
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if since_ms is not None:
            clauses.append("observed_at_ms >= ?")
            params.append(int(since_ms))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"""
            SELECT sample_id, label, scope, value, observed_at_ms,
                   window_start_ms, window_end_ms, metadata_json
            FROM slo_samples {where}
            ORDER BY observed_at_ms DESC, sample_id DESC LIMIT ?
            """,
            (*params, max(0, int(limit))),
        ).fetchall()
    return tuple(
        ArchiveSloSample(
            sample_id=str(row[0]),
            label=str(row[1]),
            scope=str(row[2]),
            value=float(row[3]),
            observed_at_ms=int(row[4]),
            window_start_ms=int(row[5]) if row[5] is not None else None,
            window_end_ms=int(row[6]) if row[6] is not None else None,
            metadata=json.loads(str(row[7])) if isinstance(row[7], str) else {},
        )
        for row in rows
    )


def gc_slo_samples(ops_db: Path, *, now_ms: int) -> int:
    """Apply retention and return the number of removed samples."""
    with open_daemon_connection(ops_db) as conn:
        _ensure_slo_samples_table(conn)
        with conn:
            before = int(conn.execute("SELECT COUNT(*) FROM slo_samples").fetchone()[0])
            _prune_slo_samples(conn, now_ms=now_ms)
            remaining = int(conn.execute("SELECT COUNT(*) FROM slo_samples").fetchone()[0])
    return before - remaining


__all__ = [
    "ArchiveSloSample",
    "SLO_SAMPLE_RETENTION_MS",
    "SLO_SAMPLE_ROW_CAP",
    "gc_slo_samples",
    "list_slo_samples",
    "record_slo_sample",
]
