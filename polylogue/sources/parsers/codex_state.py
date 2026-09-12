"""Codex live SQLite state — ``~/.codex/*.sqlite``.

Codex keeps seven SQLite databases outside the JSONL rollout files that
``parsers/codex.py`` parses:

    state_5.sqlite            threads, thread_spawn_edges, thread_artifacts,
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
during ingest). It reads through ``sources/sqlite_export.open_logical_source``,
so the same functions serve a retained canonical export and the operator's
live file.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from polylogue.core.json import JSONDocument
from polylogue.sources.sqlite_export import LogicalExportError, logical_source_shape, open_logical_source

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


def _connect_readonly(path: Path, *, timeout: float = 1.0, immutable: bool = False) -> sqlite3.Connection:
    """Open *path* for reading, whether it is a retained export or a live file.

    The retained material for a declared member is its canonical logical
    export; the operator's live ``~/.codex`` databases are still read in
    place for detection and ambient title enrichment. Never takes a write
    lock against a live Codex.
    """
    return open_logical_source(path, immutable=immutable, timeout=timeout)


def classify_codex_sqlite_path(path: Path, *, immutable: bool = False) -> CodexSqliteKind:
    """Classify a retained export or a live Codex SQLite file by its table shape.

    Returns ``"unknown"`` for anything unreadable or unrecognized rather than
    raising -- classification runs against a live, possibly-locked file.
    """
    try:
        tables = frozenset(logical_source_shape(path, immutable=immutable))
    except (sqlite3.Error, OSError, LogicalExportError, ValueError):
        return "unknown"
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
    """Everything ``parse_codex_state_db`` extracts from ``state_5.sqlite``."""

    threads: tuple[CodexThreadRecord, ...]
    spawn_edges: tuple[CodexSpawnEdge, ...]


@dataclass(frozen=True, slots=True)
class CodexThreadGoal:
    """One ``thread_goals`` row from ``goals_1.sqlite``."""

    thread_id: str
    goal_id: str
    objective: str
    status: str
    token_budget: int | None
    tokens_used: int
    time_used_seconds: int
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class CodexMemoryRecord:
    """One ``stage1_outputs`` row from ``memories_1.sqlite``.

    The text is provider-generated material, not a user assertion.  It is
    exposed so the production material read route can retain it with explicit
    Codex provenance; it is never promoted into a message or session row.
    """

    thread_id: str
    source_updated_at_ms: int
    generated_at_ms: int
    raw_memory: str
    rollout_summary: str
    usage_count: int | None
    has_rollout_slug: bool
    selected_for_phase2: bool


def _row_str(row: sqlite3.Row, key: str, default: str = "") -> str:
    value = row[key]
    return value if isinstance(value, str) else default


def _row_int(row: sqlite3.Row, key: str, default: int = 0) -> int:
    value = row[key]
    return int(value) if isinstance(value, (int, float)) else default


def _row_opt_str(row: sqlite3.Row, key: str) -> str | None:
    value = row[key]
    return value if isinstance(value, str) and value else None


def _row_opt_int(row: sqlite3.Row, key: str) -> int | None:
    value = row[key]
    return int(value) if isinstance(value, (int, float)) else None


def parse_codex_state_db(path: Path, *, immutable: bool = False) -> CodexStateSnapshot:
    """Parse ``threads`` and ``thread_spawn_edges`` from a Codex ``state_5.sqlite`` snapshot.

    *path* must already be a consistent, non-live snapshot (see
    ``sources.sqlite_snapshot.snapshot_sqlite_database`` / ``stage_sqlite_snapshot``)
    -- this function itself only ever opens read-only, but callers acquiring
    from the live file are responsible for snapshotting first (polylogue-0jf4
    acceptance criterion 4).
    """
    with closing(_connect_readonly(path, immutable=immutable)) as conn:
        conn.row_factory = sqlite3.Row
        thread_rows = conn.execute(
            "SELECT id, title, cwd, created_at_ms, updated_at_ms, source, model, "
            "agent_nickname, agent_role, archived FROM threads ORDER BY id"
        ).fetchall()
        edge_rows = conn.execute(
            "SELECT parent_thread_id, child_thread_id, status FROM thread_spawn_edges "
            "ORDER BY parent_thread_id, child_thread_id"
        ).fetchall()
    threads = tuple(
        CodexThreadRecord(
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
        for row in thread_rows
        if _row_str(row, "id")
    )
    edges = tuple(
        CodexSpawnEdge(
            parent_thread_id=_row_str(row, "parent_thread_id"),
            child_thread_id=_row_str(row, "child_thread_id"),
            status=_row_str(row, "status"),
        )
        for row in edge_rows
        if _row_str(row, "parent_thread_id") and _row_str(row, "child_thread_id")
    )
    return CodexStateSnapshot(threads=threads, spawn_edges=edges)


def parse_codex_goals_db(path: Path, *, immutable: bool = False) -> tuple[CodexThreadGoal, ...]:
    """Parse ``thread_goals`` from a Codex ``goals_1.sqlite`` snapshot."""
    with closing(_connect_readonly(path, immutable=immutable)) as conn:
        conn.row_factory = sqlite3.Row
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(thread_goals)")}

        def col(name: str, fallback: str) -> str:
            return name if name in columns else f"{fallback} AS {name}"

        rows = conn.execute(
            "SELECT "
            + ", ".join(
                (
                    col(name, fallback)
                    for name, fallback in (
                        ("thread_id", "''"),
                        ("goal_id", "thread_id"),
                        ("objective", "''"),
                        ("status", "''"),
                        ("token_budget", "NULL"),
                        ("tokens_used", "0"),
                        ("time_used_seconds", "0"),
                        ("created_at_ms", "0"),
                        ("updated_at_ms", "0"),
                    )
                )
            )
            + " FROM thread_goals ORDER BY thread_id"
        ).fetchall()
    return tuple(
        CodexThreadGoal(
            thread_id=_row_str(row, "thread_id"),
            goal_id=_row_str(row, "goal_id"),
            objective=_row_str(row, "objective"),
            status=_row_str(row, "status"),
            token_budget=_row_opt_int(row, "token_budget"),
            tokens_used=_row_int(row, "tokens_used"),
            time_used_seconds=_row_int(row, "time_used_seconds"),
            created_at_ms=_row_int(row, "created_at_ms"),
            updated_at_ms=_row_int(row, "updated_at_ms"),
        )
        for row in rows
        if _row_str(row, "thread_id") and _row_str(row, "goal_id")
    )


def parse_codex_memories_db(path: Path, *, immutable: bool = False) -> tuple[CodexMemoryRecord, ...]:
    """Parse generated memory content and accounting from a retained snapshot."""
    with closing(_connect_readonly(path, immutable=immutable)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT thread_id, source_updated_at, generated_at, raw_memory, rollout_summary, usage_count, "
            "rollout_slug, selected_for_phase2 FROM stage1_outputs ORDER BY thread_id"
        ).fetchall()
    return tuple(
        CodexMemoryRecord(
            thread_id=_row_str(row, "thread_id"),
            source_updated_at_ms=_row_int(row, "source_updated_at"),
            generated_at_ms=_row_int(row, "generated_at"),
            raw_memory=_row_str(row, "raw_memory"),
            rollout_summary=_row_str(row, "rollout_summary"),
            usage_count=_row_opt_int(row, "usage_count"),
            has_rollout_slug=_row_opt_str(row, "rollout_slug") is not None,
            selected_for_phase2=bool(_row_int(row, "selected_for_phase2")),
        )
        for row in rows
        if _row_str(row, "thread_id")
    )


__all__ = [
    "CODEX_STATE_DB_MARKER",
    "CODEX_STATE_FIDELITY",
    "CODEX_STATE_TABLE_FIDELITY",
    "IN_SCOPE_KINDS",
    "CodexAcquisitionDisposition",
    "CodexMemoryRecord",
    "CodexSpawnEdge",
    "CodexSqliteKind",
    "CodexStateDbClassification",
    "CodexStateTableClassification",
    "CodexStateSnapshot",
    "CodexTableDisposition",
    "CodexThreadGoal",
    "CodexThreadRecord",
    "classify_codex_sqlite_path",
    "declared_codex_sqlite_classification",
    "is_in_scope_codex_sqlite_path",
    "looks_like_state_db_payload",
    "marker_payload",
    "parse_codex_goals_db",
    "parse_codex_memories_db",
    "parse_codex_state_db",
]
