"""Real-SQLite fixtures and independent facts for convergence survivor tests.

This module adapts the production archive writers, daemon stages, and ops
ledger. It deliberately owns no alternate convergence state machine.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polylogue.daemon.convergence_stages as convergence_stages
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.scenarios import WorkloadEnvelopeSpec, partial_convergence_canary_spec
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection import open_connection

SqlValue = str | int | float | bytes | None
FactRow = tuple[SqlValue, ...]


@dataclass(frozen=True, slots=True)
class PartialConvergenceArchive:
    root: Path
    index_db: Path
    source_db: Path
    ops_db: Path
    target_source: Path
    unrelated_source: Path
    target_session_id: str
    unrelated_session_id: str
    workload_spec: WorkloadEnvelopeSpec

    def make_target_quiet(self) -> None:
        truncate_sparse(self.target_source, 1_024)


@dataclass(frozen=True, slots=True)
class DebtLedgerRow:
    debt_id: str
    stage: str
    subject_type: str
    subject_id: str
    status: str
    attempts: int
    last_error: str | None
    next_retry_at: str | None
    materializer_version: str | None
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class SessionMaterializationFacts:
    """Stable terminal facts, excluding attempt-time materialization stamps."""

    profile: FactRow | None
    materializations: tuple[FactRow, ...]
    work_events: tuple[FactRow, ...]
    phases: tuple[FactRow, ...]
    threads: tuple[FactRow, ...]
    thread_sessions: tuple[FactRow, ...]
    table_counts: tuple[tuple[str, int], ...]


def seed_partial_convergence_archive(root: Path, *, target_hot: bool) -> PartialConvergenceArchive:
    """Seed the current partial-convergence workload through typed archive writes."""
    root.mkdir(parents=True, exist_ok=True)
    index_db = root / "index.db"
    source_db = root / "source.db"
    ops_db = root / "ops.db"
    target_source = root / "profile-growing-codex.jsonl"
    unrelated_source = root / "unrelated-codex.jsonl"
    target_size = convergence_stages._HOT_INSIGHT_SOURCE_BYTES + 1 if target_hot else 1_024
    truncate_sparse(target_source, target_size)
    truncate_sparse(unrelated_source, 1_024)

    with open_connection(index_db) as conn:
        target_session_id = _seed_raw_source_session(
            conn,
            session_id="convergence-survivor",
            source_path=target_source,
        )
        unrelated_session_id = _seed_raw_source_session(
            conn,
            session_id="convergence-unrelated",
            source_path=unrelated_source,
        )
        conn.commit()

    return PartialConvergenceArchive(
        root=root,
        index_db=index_db,
        source_db=source_db,
        ops_db=ops_db,
        target_source=target_source,
        unrelated_source=unrelated_source,
        target_session_id=target_session_id,
        unrelated_session_id=unrelated_session_id,
        workload_spec=partial_convergence_canary_spec(
            profile_id="workload-profile:testdiet-02-partial-convergence",
            archive_id="archive:testdiet-02-partial-convergence",
        ),
    )


def truncate_sparse(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)


def debt_ledger_row(
    ops_db: Path,
    *,
    stage: str,
    subject_type: str,
    subject_id: str,
) -> DebtLedgerRow | None:
    with sqlite3.connect(ops_db) as conn:
        row = conn.execute(
            """
            SELECT debt_id, stage, target_type, target_id, status, attempts,
                   last_error, next_retry_at, materializer_version,
                   created_at_ms, updated_at_ms
            FROM convergence_debt
            WHERE stage = ? AND target_type = ? AND target_id = ?
            """,
            (stage, subject_type, subject_id),
        ).fetchone()
    if row is None:
        return None
    return DebtLedgerRow(
        debt_id=str(row[0]),
        stage=str(row[1]),
        subject_type=str(row[2]),
        subject_id=str(row[3]),
        status=str(row[4]),
        attempts=int(row[5]),
        last_error=None if row[6] is None else str(row[6]),
        next_retry_at=None if row[7] is None else str(row[7]),
        materializer_version=None if row[8] is None else str(row[8]),
        created_at_ms=int(row[9]),
        updated_at_ms=int(row[10]),
    )


def set_debt_retry_at(
    ops_db: Path,
    *,
    stage: str,
    subject_type: str,
    subject_id: str,
    retry_at: str,
) -> None:
    with sqlite3.connect(ops_db) as conn:
        cursor = conn.execute(
            """
            UPDATE convergence_debt
            SET next_retry_at = ?
            WHERE stage = ? AND target_type = ? AND target_id = ?
            """,
            (retry_at, stage, subject_type, subject_id),
        )
        conn.commit()
    if cursor.rowcount != 1:
        raise AssertionError(f"expected one convergence debt row, updated {cursor.rowcount}")


def make_messages_fts_stale(index_db: Path, *, session_id: str) -> int:
    """Delete only this session's real FTS rows to create unrelated stage debt."""
    with sqlite3.connect(index_db) as conn:
        row_ids = [
            int(row[0])
            for row in conn.execute(
                "SELECT rowid FROM blocks WHERE session_id = ? ORDER BY rowid",
                (session_id,),
            ).fetchall()
        ]
        conn.executemany("DELETE FROM messages_fts WHERE rowid = ?", ((row_id,) for row_id in row_ids))
        conn.commit()
    if not row_ids:
        raise AssertionError(f"session {session_id!r} has no indexed blocks")
    return len(row_ids)


def messages_fts_match_count(index_db: Path, query: str) -> int:
    with sqlite3.connect(index_db) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
            (query,),
        ).fetchone()
    return 0 if row is None else int(row[0])


def raw_authority_facts(source_db: Path) -> tuple[FactRow, ...]:
    with sqlite3.connect(source_db) as conn:
        rows = conn.execute(
            """
            SELECT raw_id, origin, native_id, source_path, hex(blob_hash),
                   blob_size, acquired_at_ms
            FROM raw_sessions
            ORDER BY raw_id
            """
        ).fetchall()
    return _fact_rows(rows)


def session_materialization_facts(index_db: Path, *, session_id: str) -> SessionMaterializationFacts:
    with sqlite3.connect(index_db) as conn:
        profile_row = conn.execute(
            """
            SELECT session_id, logical_session_id, materializer_version,
                   source_updated_at, source_sort_key, input_high_water_mark,
                   input_high_water_mark_source, input_row_count, source_name,
                   title, message_count, work_event_count, phase_count,
                   word_count, tool_use_count, thinking_count, total_cost_usd,
                   total_duration_ms, workflow_shape, terminal_state,
                   total_input_tokens, total_output_tokens,
                   evidence_payload_json, inference_payload_json,
                   enrichment_payload_json
            FROM session_profiles
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        materializations = conn.execute(
            """
            SELECT insight_type, session_id, materializer_version,
                   source_updated_at_ms, source_sort_key_ms,
                   input_high_water_mark_ms, input_high_water_mark_source,
                   input_row_count
            FROM insight_materialization
            WHERE session_id = ?
            ORDER BY insight_type
            """,
            (session_id,),
        ).fetchall()
        work_events = conn.execute(
            """
            SELECT session_id, position, work_event_type, summary, confidence,
                   start_index, end_index, started_at_ms, ended_at_ms,
                   duration_ms, file_paths_json, tools_used_json,
                   input_high_water_mark, input_high_water_mark_source,
                   evidence_json, inference_json, search_text
            FROM session_work_events
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        phases = conn.execute(
            """
            SELECT session_id, position, start_index, end_index,
                   started_at_ms, ended_at_ms, duration_ms,
                   tool_counts_json, word_count,
                   input_high_water_mark, input_high_water_mark_source,
                   evidence_json, inference_json, search_text
            FROM session_phases
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        threads = conn.execute(
            """
            SELECT t.thread_id, t.dominant_repo_id, t.materializer_version,
                   t.source_updated_at, t.input_high_water_mark,
                   t.input_high_water_mark_source, t.input_row_count,
                   t.start_time, t.end_time, t.dominant_repo,
                   t.session_ids_json, t.session_count, t.depth, t.branch_count,
                   t.total_messages, t.total_cost_usd, t.wall_duration_ms,
                   t.work_event_breakdown_json, t.payload_json, t.search_text
            FROM threads AS t
            JOIN thread_sessions AS ts ON ts.thread_id = t.thread_id
            WHERE ts.session_id = ?
            ORDER BY t.thread_id
            """,
            (session_id,),
        ).fetchall()
        thread_sessions = conn.execute(
            """
            SELECT thread_id, session_id, position
            FROM thread_sessions
            WHERE session_id = ?
            ORDER BY thread_id, position
            """,
            (session_id,),
        ).fetchall()
        count_tables = (
            "sessions",
            "messages",
            "blocks",
            "session_profiles",
            "insight_materialization",
            "session_work_events",
            "session_phases",
            "threads",
            "thread_sessions",
        )
        table_counts = tuple(
            (table, int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])) for table in count_tables
        )
    return SessionMaterializationFacts(
        profile=None if profile_row is None else cast(FactRow, tuple(profile_row)),
        materializations=_fact_rows(materializations),
        work_events=_fact_rows(work_events),
        phases=_fact_rows(phases),
        threads=_fact_rows(threads),
        thread_sessions=_fact_rows(thread_sessions),
        table_counts=table_counts,
    )


def _seed_raw_source_session(conn: sqlite3.Connection, *, session_id: str, source_path: Path) -> str:
    raw_id = f"raw-{session_id}"
    source_db = _main_db_path(conn).with_name("source.db")
    with sqlite3.connect(source_db) as source_conn:
        initialize_archive_tier(source_conn, ArchiveTier.SOURCE)
        source_conn.execute(
            """
            INSERT OR REPLACE INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "codex-session",
                session_id,
                str(source_path),
                hashlib.sha256(f"raw:{session_id}".encode()).digest(),
                source_path.stat().st_size,
                1_769_000_000_000,
            ),
        )
        source_conn.commit()
    return write_parsed_session_to_archive(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=session_id,
            title=session_id,
            created_at="2026-05-24T01:00:00+00:00",
            updated_at="2026-05-24T01:00:00+00:00",
            messages=[
                ParsedMessage(
                    provider_message_id="msg-1",
                    role=Role.normalize("user"),
                    text=f"Message for {session_id}",
                    position=0,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TEXT,
                            text=f"Message for {session_id}",
                        )
                    ],
                )
            ],
        ),
        raw_id=raw_id,
        content_hash=hashlib.sha256(f"session:{session_id}".encode()).hexdigest(),
    )


def _main_db_path(conn: sqlite3.Connection) -> Path:
    row = conn.execute("PRAGMA database_list").fetchone()
    if row is None or not row[2]:
        raise AssertionError("archive index connection has no main database path")
    return Path(str(row[2]))


def _fact_rows(rows: list[tuple[object, ...]]) -> tuple[FactRow, ...]:
    return tuple(cast(FactRow, tuple(row)) for row in rows)


__all__ = [
    "DebtLedgerRow",
    "PartialConvergenceArchive",
    "SessionMaterializationFacts",
    "debt_ledger_row",
    "make_messages_fts_stale",
    "messages_fts_match_count",
    "raw_authority_facts",
    "seed_partial_convergence_archive",
    "session_materialization_facts",
    "set_debt_retry_at",
    "truncate_sparse",
]
