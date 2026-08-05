"""Read-only derived-model snapshots for real archive rebuild differentials.

The current index DDL is the census authority.  A new ordinary index table
must either be compared here or receive an explicit non-comparison reason;
otherwise the differential fails before it can quietly lose coverage.
"""

from __future__ import annotations

import dataclasses
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from polylogue.storage.fts.sql import FTS_INDEXABLE_MESSAGE_COUNT_SQL
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.index import FTS_FRESHNESS_STATE_DDL, INDEX_DDL

SqlValue = str | int | float | bytes | None
FactRow = tuple[SqlValue, ...]

_CREATE_TABLE = re.compile(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_CREATE_VIRTUAL_TABLE = re.compile(
    r"CREATE\s+VIRTUAL\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE
)

# These tables record maintenance attempts rather than the logical index
# model. Their stable semantic consequences are compared through the current
# revision heads, FTS readiness, materialization markers, and debt state.
_NON_COMPARABLE_TABLES: dict[str, str] = {
    "fts_freshness_state": "compared through FtsReadiness without its wall-clock check timestamp",
    "messages_fts_identity": "FTS support relation compared through public search and FtsReadiness",
    "query_unit_frame_state": "cursor invalidation epoch depends on write-route history",
    "raw_revision_applications": "attempt receipts contain generated decision ids and wall-clock timestamps",
}

# One explicit entry per comparable DDL table. Empty sets are declarations:
# they make an added table fail this test until its volatility is considered.
_VOLATILE_COLUMNS: dict[str, frozenset[str]] = {
    "action_pairs": frozenset(),
    "agent_meta_sidecar_purge_receipts": frozenset(),
    "attachment_native_ids": frozenset(),
    "attachment_refs": frozenset(),
    "attachments": frozenset(),
    "blocks": frozenset(),
    "delegation_facts": frozenset(),
    "delegation_refresh_scope": frozenset(),
    "derived_refresh_guard": frozenset(),
    "file_edits": frozenset(),
    "insight_materialization": frozenset({"materialized_at_ms"}),
    "messages": frozenset(),
    "paste_spans": frozenset(),
    "raw_revision_heads": frozenset({"decided_at_ms"}),
    "repo_checkouts": frozenset(),
    "repos": frozenset(),
    "session_agent_policies": frozenset(),
    "session_commits": frozenset(),
    "session_events": frozenset(),
    "session_latency_profiles": frozenset({"materialized_at"}),
    "session_links": frozenset({"observed_at_ms", "resolved_at_ms"}),
    "session_model_usage": frozenset(),
    "session_phases": frozenset(),
    "session_profiles": frozenset({"materialized_at", "priced_at_ms"}),
    "session_provider_usage_events": frozenset(),
    "session_refs": frozenset(),
    "session_repos": frozenset(),
    "session_tag_rollups": frozenset({"materialized_at"}),
    "session_tags": frozenset(),
    "session_work_events": frozenset(),
    "session_working_dirs": frozenset(),
    "sessions": frozenset(),
    "thread_sessions": frozenset(),
    "threads": frozenset({"materialized_at"}),
    "web_content_constructs": frozenset(),
    "work_evidence_edges": frozenset(),
    "work_evidence_graphs": frozenset(),
    "work_evidence_nodes": frozenset(),
}

_PUBLIC_VOLATILE_FIELDS = frozenset({"materialized_at", "checked_at", "priced_at"})


@dataclass(frozen=True, slots=True)
class TableProjection:
    columns: tuple[str, ...]
    rows: tuple[FactRow, ...]


@dataclass(frozen=True, slots=True)
class FtsReadiness:
    ledger: tuple[FactRow, ...]
    source_rows: int
    indexed_rows: int
    public_index_count: int
    public_searches: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class DerivedModelSnapshot:
    tables: tuple[tuple[str, TableProjection], ...]
    public_reads: tuple[tuple[str, object], ...]
    fts: FtsReadiness
    materialization_markers: tuple[FactRow, ...]
    open_debt: tuple[FactRow, ...]


def compared_table_census() -> tuple[str, ...]:
    """Return all ordinary current-DDL index tables with a declared policy."""
    ddl = f"{FTS_FRESHNESS_STATE_DDL}\n{INDEX_DDL}"
    tables = frozenset(_CREATE_TABLE.findall(ddl))
    virtual_tables = frozenset(_CREATE_VIRTUAL_TABLE.findall(ddl))
    if virtual_tables != {"messages_fts", "blocks_command_trigram", "session_work_events_fts"}:
        raise AssertionError(f"unclassified virtual index tables: {sorted(virtual_tables)}")
    classified = set(_VOLATILE_COLUMNS) | set(_NON_COMPARABLE_TABLES)
    if missing := tables - classified:
        raise AssertionError(f"current index DDL tables lack differential classification: {sorted(missing)}")
    if stale := classified - tables:
        raise AssertionError(f"differential table declarations no longer exist in index DDL: {sorted(stale)}")
    return tuple(sorted(tables - set(_NON_COMPARABLE_TABLES)))


def snapshot_derived_model(
    archive_root: Path,
    index_path: Path,
    *,
    session_ids: tuple[str, ...],
    search_queries: tuple[str, ...],
) -> DerivedModelSnapshot:
    """Read one archive generation without mutating any archive tier."""
    census = compared_table_census()
    with _connect(index_path) as conn:
        tables = tuple((table, _project_table(conn, table)) for table in census)
        markers = _marker_rows(conn)
        fts_ledger = _fts_ledger_rows(conn)
        source_rows = int(conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone()[0])
        indexed_rows = int(conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        public_reads = _public_reads(archive, session_ids)
        searches = tuple((query, tuple(archive.search_blocks(query))) for query in search_queries)
        public_index_count = int(archive.index_status()["count"])
    return DerivedModelSnapshot(
        tables=tables,
        public_reads=public_reads,
        fts=FtsReadiness(
            ledger=fts_ledger,
            source_rows=source_rows,
            indexed_rows=indexed_rows,
            public_index_count=public_index_count,
            public_searches=searches,
        ),
        materialization_markers=markers,
        open_debt=_open_debt_rows(archive_root / "ops.db"),
    )


def assert_derived_models_equivalent(expected: DerivedModelSnapshot, actual: DerivedModelSnapshot) -> None:
    """Fail with the first durable/public differential, named for diagnosis."""
    expected_tables = dict(expected.tables)
    actual_tables = dict(actual.tables)
    if expected_tables.keys() != actual_tables.keys():
        raise AssertionError(
            f"derived-table census differs: expected={sorted(expected_tables)}, actual={sorted(actual_tables)}"
        )
    for table in expected_tables:
        if expected_tables[table] != actual_tables[table]:
            raise AssertionError(
                f"derived table {table} differs: {_table_difference(expected_tables[table], actual_tables[table])}"
            )
    if expected.public_reads != actual.public_reads:
        raise AssertionError(
            f"public insight reads differ: {_value_difference(expected.public_reads, actual.public_reads)}"
        )
    if expected.fts != actual.fts:
        raise AssertionError("FTS readiness or public FTS reads differ")
    if expected.materialization_markers != actual.materialization_markers:
        raise AssertionError("insight materialization markers differ")
    if expected.open_debt != actual.open_debt:
        raise AssertionError(f"open convergence debt differs: expected={expected.open_debt}, actual={actual.open_debt}")


def assert_derived_model_ready(snapshot: DerivedModelSnapshot) -> None:
    """Keep a matching but jointly stale generation from passing the lane."""
    if snapshot.fts.source_rows != snapshot.fts.indexed_rows:
        raise AssertionError(
            f"FTS is not ready: source_rows={snapshot.fts.source_rows}, indexed_rows={snapshot.fts.indexed_rows}"
        )
    if snapshot.open_debt:
        raise AssertionError(f"convergence debt remains: {snapshot.open_debt}")


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _project_table(conn: sqlite3.Connection, table: str) -> TableProjection:
    volatile = _VOLATILE_COLUMNS[table]
    actual_columns = {str(row["name"]) for row in conn.execute(f'PRAGMA table_xinfo("{table}")')}
    if unknown := volatile - actual_columns:
        raise AssertionError(f"volatile declaration for {table} names missing columns: {sorted(unknown)}")
    columns = tuple(
        str(row["name"]) for row in conn.execute(f'PRAGMA table_xinfo("{table}")') if str(row["name"]) not in volatile
    )
    quoted = ", ".join(f'"{column}"' for column in columns)
    rows = tuple(sorted((_fact_row(row) for row in conn.execute(f'SELECT {quoted} FROM "{table}"')), key=repr))
    return TableProjection(columns=columns, rows=rows)


def _marker_rows(conn: sqlite3.Connection) -> tuple[FactRow, ...]:
    rows = conn.execute(
        """
        SELECT insight_type, session_id, materializer_version, source_updated_at_ms,
               source_sort_key_ms, input_high_water_mark_ms,
               input_high_water_mark_source, input_row_count
        FROM insight_materialization
        ORDER BY insight_type, session_id
        """
    )
    return tuple(_fact_row(row) for row in rows)


def _fts_ledger_rows(conn: sqlite3.Connection) -> tuple[FactRow, ...]:
    rows = conn.execute(
        """
        SELECT surface, state, source_rows, indexed_rows, missing_rows,
               excess_rows, duplicate_rows, detail
        FROM fts_freshness_state
        ORDER BY surface
        """
    )
    return tuple(_fact_row(row) for row in rows)


def _open_debt_rows(ops_path: Path) -> tuple[FactRow, ...]:
    if not ops_path.exists():
        return ()
    with sqlite3.connect(ops_path) as conn:
        row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'convergence_debt'").fetchone()
        if row is None:
            return ()
        rows = conn.execute(
            """
            SELECT stage, target_type, target_id, status, attempts,
                   last_error, next_retry_at, materializer_version
            FROM convergence_debt
            WHERE status != 'resolved'
            ORDER BY stage, target_type, target_id
            """
        )
        return tuple(tuple(_normalize(value) for value in row) for row in rows)


def _public_reads(archive: ArchiveStore, session_ids: tuple[str, ...]) -> tuple[tuple[str, object], ...]:
    values: list[tuple[str, object]] = []
    for session_id in session_ids:
        values.extend(
            (
                (f"profile:{session_id}", _freeze_public(archive.get_session_profile_insight(session_id))),
                (f"work-events:{session_id}", _freeze_public(archive.get_session_work_event_insights(session_id))),
                (f"phases:{session_id}", _freeze_public(archive.get_session_phase_insights(session_id))),
            )
        )
    values.append(("threads", _freeze_public(archive.list_thread_insights(limit=None))))
    return tuple(values)


def _freeze_public(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _freeze_public(dataclasses.asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _freeze_public(model_dump())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_public(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _PUBLIC_VOLATILE_FIELDS
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(_freeze_public(item) for item in value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _fact_row(row: sqlite3.Row) -> FactRow:
    return tuple(_normalize(value) for value in row)


def _normalize(value: Any) -> SqlValue:
    if isinstance(value, bytes):
        return value.hex()
    if value is None or isinstance(value, (str, int, float)):
        return value
    raise TypeError(f"unsupported SQLite fact value: {type(value)!r}")


def _table_difference(expected: TableProjection, actual: TableProjection) -> str:
    if expected.columns != actual.columns:
        return f"columns expected={expected.columns}, actual={actual.columns}"
    expected_rows = set(expected.rows)
    actual_rows = set(actual.rows)
    return f"only_expected={sorted(expected_rows - actual_rows, key=repr)[:2]!r}, only_actual={sorted(actual_rows - expected_rows, key=repr)[:2]!r}"


def _value_difference(expected: object, actual: object) -> str:
    expected_items = dict(expected) if isinstance(expected, tuple) else {"value": expected}
    actual_items = dict(actual) if isinstance(actual, tuple) else {"value": actual}
    keys = sorted(set(expected_items) | set(actual_items))
    for key in keys:
        if expected_items.get(key) != actual_items.get(key):
            return f"{key}: expected={expected_items.get(key)!r}, actual={actual_items.get(key)!r}"
    return f"expected={expected!r}, actual={actual!r}"
