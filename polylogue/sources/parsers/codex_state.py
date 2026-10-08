"""Codex live SQLite state — ``~/.codex/*.sqlite``.

Codex keeps seven SQLite databases outside the JSONL rollout files that
``parsers/codex.py`` parses:

    state_5.sqlite            threads, thread_spawn_edges, thread_artifacts / thread_attachments,
                              thread_dynamic_tools, thread_sections, projects,
                              project_roots, and operational bookkeeping
    goals_1.sqlite            thread_goals, thread_goal_continuation_deferrals
    memories_1.sqlite         stage1_outputs, jobs
    logs_2.sqlite             logs (runtime tracing: level/target/module_path/file/line)
    codex-dev.db              inbox_items, automations, automation_runs
    thread_history_1.sqlite   thread_turns, thread_items, thread_realtime_items,
                              thread_history_projection_state
    queue_1.sqlite            queued_items, queued_thread_revisions

``CODEX_STATE_FIDELITY`` states the disposition and the reason for each
database. ``CODEX_STATE_TABLE_FIDELITY`` does the same for every observed
``state_5.sqlite`` table. Every one has a disposition: an undeclared database
or table beside a declared one is a silent acquisition decision.

``threads.title`` and ``thread_spawn_edges`` are evidence the JSONL rollout
files never carry at all (verified empirically: no rollout ``session_meta``
record embeds a curated title or a parent/child spawn relationship — spawned
subagents are only linked in-session via ``forked_from_id``/
``source.subagent.thread_spawn``, which ``parsers/codex.py`` already reads;
``thread_spawn_edges`` is Codex's own orchestration-level record of the same
relationship plus edges that in-session evidence doesn't capture, such as
edges from a still-running or crashed child).

This module owns detecting and parsing that state. It is deliberately
independent of ``parsers/codex.py`` (which owns JSONL rollout parsing) and of
``sources/assembly_codex.py`` (which owns live, ambient title enrichment
during ingest). It reads through ``sources/sqlite_export.logical_source_context``,
so the same functions serve a retained canonical export and the operator's
live file.
"""

from __future__ import annotations

import math
import sqlite3
from codecs import getincrementaldecoder
from collections.abc import Generator, Iterable, Iterator
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from polylogue.core.json import JSONDocument
from polylogue.sources.sqlite_export import LogicalExportError, logical_source_context, logical_source_shape

CODEX_STATE_DB_MARKER = "codex_state_db"

CodexSqliteKind: TypeAlias = Literal[
    "thread_state",  # state_5.sqlite -- threads + thread_spawn_edges
    "goals",  # goals_1.sqlite -- thread_goals
    "memories",  # memories_1.sqlite -- stage1_outputs
    "logs",  # logs_2.sqlite -- runtime tracing
    "automation",  # codex-dev.db -- inbox/automation scheduling
    "thread_history",  # thread_history_1.sqlite -- UI projection of the rollout
    "queue",  # queue_1.sqlite -- input queued for submission
    "unknown",
]

CodexAcquisitionDisposition: TypeAlias = Literal["acquire", "acquire-partial", "out-of-scope"]
CodexTableDisposition: TypeAlias = Literal[
    "retained-and-consumed",
    "retained-for-later-consumption",
    "deliberately-excluded",
]

# Required-table fingerprints used for classification. Checked in this order;
# the first match wins. A file must have ALL tables in the tuple to match.
_KIND_REQUIRED_TABLES: tuple[tuple[CodexSqliteKind, tuple[str, ...]], ...] = (
    ("thread_state", ("threads", "thread_spawn_edges")),
    ("goals", ("thread_goals",)),
    ("memories", ("stage1_outputs",)),
    ("logs", ("logs",)),
    ("automation", ("automations", "automation_runs")),
    ("thread_history", ("thread_items", "thread_history_projection_state")),
    ("queue", ("queued_items", "queued_thread_revisions")),
)


@dataclass(frozen=True, slots=True)
class CodexStateDbClassification:
    """One database's evidence classification and the reason for it."""

    kind: CodexSqliteKind
    disposition: CodexAcquisitionDisposition
    filenames: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class CodexStateTableClassification:
    """One ``state_5.sqlite`` table's retained-evidence disposition."""

    table: str
    disposition: CodexTableDisposition
    reason: str


#: Every SQLite database Codex keeps, with its disposition and the reason.
#: ``sources/origin_specs.py:_codex_spec()`` declares the same dispositions as
#: ``DatabaseMemberRule`` members; the two are checked against each other.
CODEX_STATE_FIDELITY: tuple[CodexStateDbClassification, ...] = (
    CodexStateDbClassification(
        kind="thread_state",
        disposition="acquire",
        filenames=("state_5.sqlite",),
        reason=(
            "threads.title and thread_spawn_edges have no other evidence source: "
            "no Codex rollout JSONL session_meta record carries a curated title or "
            "a parent/child spawn relationship at the orchestration level."
        ),
    ),
    CodexStateDbClassification(
        kind="goals",
        disposition="acquire-partial",
        filenames=("goals_1.sqlite",),
        reason=(
            "thread_goals.objective is stated task intent unavailable anywhere else, "
            "but the table is small (tens of rows) and low-churn; acquire the raw "
            "snapshot for durability, then derive current values through the shared "
            "material reader with source and thread provenance."
        ),
    ),
    CodexStateDbClassification(
        kind="memories",
        disposition="acquire-partial",
        filenames=("memories_1.sqlite",),
        reason=(
            "stage1_outputs.raw_memory is Codex-side memory with no archive "
            "representation, but its content is a derived summary over content "
            "Polylogue already ingests from the JSONL rollout; acquire the raw "
            "snapshot for durability, then retain it as explicitly provider-generated "
            "material rather than a user assertion."
        ),
    ),
    CodexStateDbClassification(
        kind="logs",
        disposition="out-of-scope",
        filenames=("logs_2.sqlite",),
        reason=(
            "627 MB of runtime tracing (level/target/module_path/file/line), not "
            "session evidence. Acquiring it by default would roughly double this "
            "archive's Codex footprint for no session-reconstruction value; treat "
            "any future runtime-observability use case as a separate, deliberate "
            "decision, not a default acquisition."
        ),
    ),
    CodexStateDbClassification(
        kind="automation",
        disposition="out-of-scope",
        filenames=("codex-dev.db",),
        reason=(
            "Local CLI automation scheduling config (inbox items, cron-like "
            "automations, automation run records), not AI session content. Empty "
            "on every install observed. automation_runs.thread_id references a "
            "session, but the row itself is scheduling state, not conversation "
            "evidence."
        ),
    ),
    CodexStateDbClassification(
        kind="thread_history",
        disposition="out-of-scope",
        filenames=("thread_history_1.sqlite",),
        reason=(
            "A UI projection of the rollout JSONL the archive already acquires, "
            "measured against it (2026-09-07, 3.6 GB on this install, 120-thread "
            "sample): every sampled thread had its rollout file; 119 of 120 "
            "thread_history_projection_state cursors sat exactly at rollout EOF "
            "and the remaining one was mid-write; all 14,215 thread_items item_ids "
            "appeared verbatim in the rollout bytes; every non-commandExecution "
            "text probe (1,446 of 1,446) appeared among the rollout's decoded "
            "strings; and 1,950 of 1,950 commandExecution rows reproduced a "
            "rollout function_call command array exactly. Acquiring it would "
            "roughly quadruple this archive's Codex footprint for content it "
            "already holds."
        ),
    ),
    CodexStateDbClassification(
        kind="queue",
        disposition="out-of-scope",
        filenames=("queue_1.sqlite",),
        reason=(
            "Input queued for submission, plus a per-thread revision counter. A "
            "queued item leaves the table when it is submitted, at which point it "
            "is a rollout userMessage the archive acquires; what remains is intent "
            "that has not happened yet, not evidence of a session. Both tables "
            "were empty on the install measured (2026-09-07)."
        ),
    ),
)


#: Every table observed in ``state_5.sqlite`` on 2026-09-12. Retained tables
#: become part of the member's canonical logical export, even when the only
#: current consumer is durable raw evidence. The parallel OriginSpec table
#: rules feed schema observation; tests keep both declarations aligned.
CODEX_STATE_TABLE_FIDELITY: tuple[CodexStateTableClassification, ...] = (
    CodexStateTableClassification(
        "threads",
        "retained-and-consumed",
        "Curated titles and orchestration metadata feed the thread-state projection.",
    ),
    CodexStateTableClassification(
        "thread_spawn_edges",
        "retained-and-consumed",
        "Codex orchestration parent/child edges feed the thread-state projection.",
    ),
    CodexStateTableClassification(
        "thread_artifacts",
        "retained-for-later-consumption",
        "Artifact identity and payload are thread evidence with no typed projection yet.",
    ),
    CodexStateTableClassification(
        "thread_attachments",
        "retained-for-later-consumption",
        "Attachment identity and payload are thread evidence with no typed projection yet.",
    ),
    CodexStateTableClassification(
        "thread_dynamic_tools",
        "retained-for-later-consumption",
        "Dynamic tool descriptions and schemas are thread evidence with no typed projection yet.",
    ),
    CodexStateTableClassification(
        "thread_sections",
        "retained-for-later-consumption",
        "Thread section definitions contextualize the retained thread section references.",
    ),
    CodexStateTableClassification(
        "projects",
        "retained-for-later-consumption",
        "Project identity and metadata contextualize retained thread project references.",
    ),
    CodexStateTableClassification(
        "project_roots",
        "retained-for-later-consumption",
        "Project roots contextualize retained project references without a typed projection yet.",
    ),
    CodexStateTableClassification(
        "_sqlx_migrations",
        "deliberately-excluded",
        "Database migration bookkeeping is not session evidence.",
    ),
    CodexStateTableClassification(
        "backfill_state",
        "deliberately-excluded",
        "Resumable backfill cursor state is operational bookkeeping, not session evidence.",
    ),
    CodexStateTableClassification(
        "external_agent_config_imports",
        "deliberately-excluded",
        "External-agent configuration import status is operational configuration, not session evidence.",
    ),
    CodexStateTableClassification(
        "project_idempotency_keys",
        "deliberately-excluded",
        "Project request deduplication keys are operational state, not session evidence.",
    ),
    CodexStateTableClassification(
        "remote_control_enrollments",
        "deliberately-excluded",
        "Remote-control enrollment configuration can carry connection details and is not session evidence.",
    ),
    CodexStateTableClassification(
        "rollout_migration_skipped_rollouts",
        "deliberately-excluded",
        "Rollout migration skip bookkeeping duplicates rollout discovery operational state.",
    ),
    CodexStateTableClassification(
        "rollout_migration_state",
        "deliberately-excluded",
        "Rollout migration cursors are operational bookkeeping, not session evidence.",
    ),
)

IN_SCOPE_KINDS: frozenset[CodexSqliteKind] = frozenset(
    classification.kind for classification in CODEX_STATE_FIDELITY if classification.disposition != "out-of-scope"
)


def classify_codex_sqlite_path(path: Path, *, immutable: bool = False) -> CodexSqliteKind:
    """Classify a retained export or a live Codex SQLite file by its table shape.

    Returns ``"unknown"`` for anything unreadable or unrecognized rather than
    raising -- classification runs against a live, possibly-locked file.
    """
    try:
        tables = frozenset(logical_source_shape(path, immutable=immutable))
    except (sqlite3.Error, OSError, LogicalExportError, ValueError):
        return "unknown"
    return classify_codex_sqlite_tables(tables)


def classify_codex_sqlite_tables(tables: frozenset[str]) -> CodexSqliteKind:
    """Use the same declared table signatures for live and bound acquisition."""
    for kind, required in _KIND_REQUIRED_TABLES:
        if required and set(required).issubset(tables):
            return kind
    return "unknown"


def is_in_scope_codex_sqlite_path(path: Path, *, immutable: bool = False) -> bool:
    """Return whether *path* is one of the databases this module acquires."""
    return classify_codex_sqlite_path(path, immutable=immutable) in IN_SCOPE_KINDS


def declared_codex_sqlite_classification(path: Path) -> CodexStateDbClassification | None:
    """Return the declared acquisition mode for a known database name."""
    for classification in CODEX_STATE_FIDELITY:
        if path.name in classification.filenames:
            return classification
    return None


def marker_payload(path: Path, *, kind: CodexSqliteKind, immutable: bool = False) -> JSONDocument:
    """Return the JSON marker that routes a raw SQLite blob to this parser."""
    payload: JSONDocument = {
        "polylogue_artifact": CODEX_STATE_DB_MARKER,
        "state_db_path": str(path),
        "state_db_kind": kind,
    }
    if immutable:
        payload["sqlite_immutable"] = True
    return payload


def looks_like_state_db_payload(payload: JSONDocument) -> bool:
    return (
        payload.get("polylogue_artifact") == CODEX_STATE_DB_MARKER
        and isinstance(payload.get("state_db_path"), str)
        and payload.get("state_db_kind") in IN_SCOPE_KINDS
    )


@dataclass(frozen=True, slots=True)
class CodexThreadRecord:
    """One ``threads`` row from ``state_5.sqlite``."""

    thread_id: str
    title: str
    cwd: str
    created_at_ms: int
    updated_at_ms: int
    source: str
    model: str | None
    agent_nickname: str | None
    agent_role: str | None
    archived: bool


@dataclass(frozen=True, slots=True)
class CodexSpawnEdge:
    """One ``thread_spawn_edges`` row from ``state_5.sqlite``.

    ``status`` is Codex's own orchestration lifecycle label for the child
    (e.g. ``"closed"``, ``"running"``) -- distinct from and not to be
    confused with ``polylogue``'s own ``TopologyEdgeStatus``.
    """

    parent_thread_id: str
    child_thread_id: str
    status: str


@dataclass(frozen=True, slots=True)
class CodexStateSnapshot:
    """Repeatable thread and spawn record views of one sealed state export."""

    threads: Iterable[CodexThreadRecord]
    spawn_edges: Iterable[CodexSpawnEdge]


#: Default keyset page for complete state materialization.
CODEX_STATE_PAGE_ROWS = 256

#: Text chunk size in characters. Materialization fetches every chunk of a
#: field by SQL ``substr``; the chunk size paces reads and never clips text.
CODEX_STATE_MAX_TEXT_CHARS = 64_000

#: Byte window for a materialization commit, not a total export limit.
CODEX_STATE_MAX_AGGREGATE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CodexStatePart:
    """One bounded, addressable projection of a state row or text field."""

    thread_id: str
    item_id: str
    part_kind: str
    payload: dict[str, object]


def _iter_text_chunks(chunks: Iterable[bytes], chunk_chars: int) -> Generator[str, None, None]:
    """Decode a SQLite TEXT value incrementally, including embedded NULs."""
    decoder = getincrementaldecoder("utf-8")()
    pending = ""
    emitted = False
    for raw in chunks:
        pending += decoder.decode(raw)
        while len(pending) >= chunk_chars:
            yield pending[:chunk_chars]
            emitted = True
            pending = pending[chunk_chars:]
    pending += decoder.decode(b"", final=True)
    if pending or not emitted:
        yield pending


def _text_length_chars(chunks: Iterable[bytes]) -> int:
    """Count characters with bounded reads; SQLite length(TEXT) stops at NUL."""
    decoder = getincrementaldecoder("utf-8")()
    count = 0
    for raw in chunks:
        count += len(decoder.decode(raw))
    count += len(decoder.decode(b"", final=True))
    return count


def iter_codex_state_parts(
    path: Path,
    *,
    state_kind: Literal["goals", "memories"],
    page_size: int = CODEX_STATE_PAGE_ROWS,
    text_chars: int = CODEX_STATE_MAX_TEXT_CHARS,
    immutable: bool = False,
) -> Generator[CodexStatePart, None, None]:
    """Read every valid state row in keyset pages and every text field in chunks.

    The retained logical export is immutable. ``rowid`` is only an internal
    traversal key; thread and item ids remain the public material coordinates.
    """
    if page_size < 1 or text_chars < 1:
        raise ValueError("page_size and text_chars must be positive")
    table = "thread_goals" if state_kind == "goals" else "stage1_outputs"
    fields = ("objective",) if state_kind == "goals" else ("raw_memory", "rollout_summary")
    # Imports stay local to the actual Native read route. Acquisition's
    # metadata-only source probe need not initialize this storage stack.
    from polylogue.core.compute_cancel import check_compute_cancelled
    from polylogue.sources.sqlite_export import readable_table_info
    from polylogue.storage.io_phase_metrics import close_connection_cursor, connection_cursor
    from polylogue.storage.sqlite.connection_profile import (
        NativeConnectionSettlementError,
        retained_native_sql_owners_on_current_thread,
    )
    from polylogue.storage.sqlite.literal_cells import SQLiteLiteralCell, owned_literal_stream, stream_literal_cell

    with logical_source_context(path, immutable=immutable) as conn:
        conn.row_factory = sqlite3.Row
        owners = tuple(owner for owner in retained_native_sql_owners_on_current_thread() if owner.connection is conn)
        if len(owners) != 1:
            raise LogicalExportError("Codex state reading requires its exact existing Native owner")
        owner = owners[0]
        owner.require_connection()
        # This factory owns a fresh connection. Keep shape, metadata, character
        # counting and every literal chunk on one native read view, including
        # direct live SQLite inputs as well as immutable reconstructions.
        with connection_cursor(conn, "BEGIN"):
            pass
        shape = readable_table_info(conn, table)
        columns = {str(row[1]) for row in shape}
        incremental = not any(row[6] in (2, 3) for row in shape)

        def close_literal_cursor(cursor: sqlite3.Cursor) -> None:
            try:
                close_connection_cursor(conn, cursor)
            except BaseException as failure:
                owner.close_required = True
                raise NativeConnectionSettlementError(owner, failure) from failure

        def literal_chunks(field: str, source_rowid: int, row: sqlite3.Row) -> Generator[bytes, None, None]:
            kind = str(row[f"kind_{field}"])
            if kind not in {"text", "blob"}:
                raise LogicalExportError("Codex state text field lacks its TEXT or BLOB storage class")
            yield from stream_literal_cell(
                conn,
                SQLiteLiteralCell("text" if kind == "text" else "blob", int(row[f"bytes_{field}"])),
                expression=field,
                source_sql=f"FROM {table} WHERE rowid=?",
                parameters=(source_rowid,),
                incremental=(lambda: owner.readonly_blob(table, field, source_rowid)) if incremental else None,
                close_cursor=close_literal_cursor,
                check_cancel=check_compute_cancelled,
            )

        after_rowid: int | None = None
        while True:
            # A page carries metadata only. Even one provider-generated
            # multi-gigabyte field never enters Python at once.
            metadata = (
                (
                    "thread_id",
                    "goal_id",
                    "status",
                    "token_budget",
                    "tokens_used",
                    "time_used_seconds",
                    "created_at_ms",
                    "updated_at_ms",
                )
                if state_kind == "goals"
                else ("thread_id", "source_updated_at", "generated_at", "usage_count", "selected_for_phase2")
            )
            select = ["rowid AS source_rowid", *(name for name in metadata if name in columns)]
            if state_kind == "memories" and "rollout_slug" in columns:
                select.append(
                    "(rollout_slug IS NOT NULL AND length(CAST(rollout_slug AS BLOB)) > 0) AS has_rollout_slug"
                )
            for field in fields:
                if field in columns:
                    select.extend(
                        (
                            f"({field} IS NULL) AS null_{field}",
                            f"typeof({field}) AS kind_{field}",
                            f"length(CAST({field} AS BLOB)) AS bytes_{field}",
                        )
                    )
                else:
                    select.append(f"1 AS null_{field}")
            where = "" if after_rowid is None else " WHERE rowid > ?"
            params = (*((after_rowid,) if after_rowid is not None else ()), page_size)
            check_compute_cancelled()
            with connection_cursor(
                conn, f"SELECT {', '.join(select)} FROM {table}{where} ORDER BY rowid LIMIT ?", params
            ) as cursor:
                rows = cursor.fetchall()
            if not rows:
                break
            for row in rows:
                source_rowid = int(row["source_rowid"])
                thread_id = _row_str(row, "thread_id")
                item_id = _row_str(row, "goal_id") if state_kind == "goals" and "goal_id" in columns else thread_id
                if not thread_id or not item_id:
                    yield CodexStatePart(thread_id, item_id, "invalid", {"source_rowid": source_rowid})
                    continue
                with ExitStack() as stack:
                    chunks: dict[str, Iterator[str]] = {}
                    first: dict[str, str] = {}
                    following: dict[str, str | None] = {}
                    char_lengths: dict[str, int] = {}
                    for field in fields:
                        if field in columns and not _row_int(row, f"null_{field}"):
                            if int(row[f"bytes_{field}"]) > text_chars:
                                with owned_literal_stream(literal_chunks(field, source_rowid, row)) as counted:
                                    char_lengths[field] = _text_length_chars(counted)
                            raw_chunks = stack.enter_context(
                                owned_literal_stream(literal_chunks(field, source_rowid, row))
                            )
                            chunks[field] = stack.enter_context(closing(_iter_text_chunks(raw_chunks, text_chars)))
                        else:
                            chunks[field] = iter(("",))
                        first[field] = next(chunks[field])
                        following[field] = next(chunks[field], None)
                    if state_kind == "goals":
                        payload: dict[str, object] = {
                            "thread_id": thread_id,
                            "goal_id": item_id,
                            "objective": first["objective"],
                            "status": _row_str(row, "status") if "status" in columns else "",
                            "token_budget": _row_opt_int(row, "token_budget") if "token_budget" in columns else None,
                            "tokens_used": _row_int(row, "tokens_used") if "tokens_used" in columns else 0,
                            "time_used_seconds": _row_int(row, "time_used_seconds")
                            if "time_used_seconds" in columns
                            else 0,
                            "created_at_ms": _row_int(row, "created_at_ms") if "created_at_ms" in columns else 0,
                            "updated_at_ms": _row_int(row, "updated_at_ms") if "updated_at_ms" in columns else 0,
                            "provider": "codex",
                            "generated": False,
                        }
                    else:
                        payload = {
                            "thread_id": thread_id,
                            "raw_memory": first["raw_memory"],
                            "rollout_summary": first["rollout_summary"],
                            "source_updated_at_ms": _row_int(row, "source_updated_at"),
                            "generated_at_ms": _row_int(row, "generated_at"),
                            "usage_count": _row_opt_int(row, "usage_count"),
                            "has_rollout_slug": bool(_row_int(row, "has_rollout_slug"))
                            if "rollout_slug" in columns
                            else False,
                            "selected_for_phase2": bool(_row_int(row, "selected_for_phase2")),
                            "provider": "codex",
                            "generated": True,
                        }
                    if any(chunk is not None for chunk in following.values()):
                        payload["text_continuation"] = {
                            field: {"length_chars": char_lengths[field], "chunk_chars": text_chars}
                            for field, chunk in following.items()
                            if chunk is not None
                        }
                    yield CodexStatePart(thread_id, item_id, "record", payload)
                    for field, chunk in following.items():
                        offset = len(first[field])
                        while chunk is not None:
                            successor = next(chunks[field], None)
                            yield CodexStatePart(
                                thread_id,
                                f"{item_id}:{field}:{offset}",
                                "text_chunk",
                                {
                                    "record_type": state_kind,
                                    "thread_id": thread_id,
                                    "item_id": item_id,
                                    "field": field,
                                    "offset_chars": offset,
                                    "text": chunk,
                                    "final": successor is None,
                                    "provider": "codex",
                                },
                            )
                            offset += len(chunk)
                            chunk = successor
            after_rowid = int(rows[-1]["source_rowid"])


def _row_str(row: sqlite3.Row, key: str, default: str = "") -> str:
    value = row[key]
    return value if isinstance(value, str) else default


def _row_int(row: sqlite3.Row, key: str, default: int = 0) -> int:
    value = row[key]
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return int(value) if isinstance(value, float) and math.isfinite(value) and value.is_integer() else default


def _row_opt_str(row: sqlite3.Row, key: str) -> str | None:
    value = row[key]
    return value if isinstance(value, str) and value else None


def _row_opt_int(row: sqlite3.Row, key: str) -> int | None:
    value = row[key]
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return int(value) if isinstance(value, float) and math.isfinite(value) and value.is_integer() else None


def iter_codex_state_records(
    path: Path,
    *,
    immutable: bool = False,
) -> Iterator[CodexThreadRecord | CodexSpawnEdge]:
    """Decode complete state records in pages through one creator-owned export."""
    from polylogue.core.compute_cancel import check_compute_cancelled

    with logical_source_context(path, immutable=immutable) as conn:
        conn.row_factory = sqlite3.Row
        for table in ("threads", "thread_spawn_edges"):
            query = (
                "SELECT id, title, cwd, created_at_ms, updated_at_ms, source, model, "
                "agent_nickname, agent_role, archived FROM threads ORDER BY id"
                if table == "threads"
                else "SELECT parent_thread_id, child_thread_id, status FROM thread_spawn_edges "
                "ORDER BY parent_thread_id, child_thread_id"
            )
            cursor = conn.execute(query)
            try:
                while True:
                    check_compute_cancelled()
                    rows = cursor.fetchmany(CODEX_STATE_PAGE_ROWS)
                    if not rows:
                        break
                    for row in rows:
                        check_compute_cancelled()
                        if table == "threads":
                            if _row_str(row, "id"):
                                yield CodexThreadRecord(
                                    thread_id=_row_str(row, "id"),
                                    title=_row_str(row, "title"),
                                    cwd=_row_str(row, "cwd"),
                                    created_at_ms=_row_int(row, "created_at_ms"),
                                    updated_at_ms=_row_int(row, "updated_at_ms"),
                                    source=_row_str(row, "source"),
                                    model=_row_opt_str(row, "model"),
                                    agent_nickname=_row_opt_str(row, "agent_nickname"),
                                    agent_role=_row_opt_str(row, "agent_role"),
                                    archived=bool(_row_int(row, "archived")),
                                )
                        elif _row_str(row, "parent_thread_id") and _row_str(row, "child_thread_id"):
                            yield CodexSpawnEdge(
                                parent_thread_id=_row_str(row, "parent_thread_id"),
                                child_thread_id=_row_str(row, "child_thread_id"),
                                status=_row_str(row, "status"),
                            )
            finally:
                cursor.close()


__all__ = [
    "CODEX_STATE_DB_MARKER",
    "CODEX_STATE_MAX_AGGREGATE_BYTES",
    "CODEX_STATE_PAGE_ROWS",
    "CODEX_STATE_MAX_TEXT_CHARS",
    "CODEX_STATE_FIDELITY",
    "CODEX_STATE_TABLE_FIDELITY",
    "IN_SCOPE_KINDS",
    "CodexAcquisitionDisposition",
    "CodexSpawnEdge",
    "CodexSqliteKind",
    "CodexStateDbClassification",
    "CodexStateTableClassification",
    "CodexStatePart",
    "CodexStateSnapshot",
    "CodexTableDisposition",
    "CodexThreadRecord",
    "classify_codex_sqlite_path",
    "declared_codex_sqlite_classification",
    "is_in_scope_codex_sqlite_path",
    "looks_like_state_db_payload",
    "marker_payload",
    "iter_codex_state_records",
    "iter_codex_state_parts",
]
