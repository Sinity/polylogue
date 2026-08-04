"""Codex live SQLite state — ``~/.codex/*.sqlite``.

Codex keeps five SQLite databases outside the JSONL rollout files that
``parsers/codex.py`` parses (polylogue-0jf4):

    state_5.sqlite     threads, thread_spawn_edges, thread_dynamic_tools,
                       remote_control_enrollments, external_agent_config_imports
    goals_1.sqlite     thread_goals, thread_goal_continuation_deferrals
    memories_1.sqlite  stage1_outputs, jobs
    logs_2.sqlite      logs (runtime tracing: level/target/module_path/file/line)
    codex-dev.db       inbox_items, automations, automation_runs

``threads.title`` and ``thread_spawn_edges`` are evidence the JSONL rollout
files never carry at all (verified empirically: no rollout ``session_meta``
record embeds a curated title or a parent/child spawn relationship — spawned
subagents are only linked in-session via ``forked_from_id``/
``source.subagent.thread_spawn``, which ``parsers/codex.py`` already reads;
``thread_spawn_edges`` is Codex's own orchestration-level record of the same
relationship plus edges that in-session evidence doesn't capture, such as
edges from a still-running or crashed child).

This module owns detecting, snapshotting, and parsing that state.  It is
deliberately independent of ``parsers/codex.py`` (which owns JSONL rollout
parsing) and of ``sources/assembly_codex.py`` (which owns live, ambient
title enrichment during ingest, polylogue-ih67, still in flight) — it does
not modify either.  What it adds is a durable, content-addressed capture of
the raw databases plus typed accessors for the evidence that has no other
home, following the same consistent-snapshot pattern
``parsers/hermes_state.py`` / ``sources/sqlite_snapshot.py`` already use for
Hermes: back up the live file (``sqlite3.Connection.backup()``) into an
immutable copy before ever hashing or parsing it, so a lock or a concurrent
writer never blocks the acquisition and the parsed content never observes a
half-written page.

Wiring this module's output into live ingestion (routing acquired bytes
through ``sources/dispatch.py`` the way
``hermes_state.looks_like_state_db_payload`` is routed there, widening the
``codex`` ``WatchSource`` root in ``sources/live/watcher.py`` from
``~/.codex/sessions`` to ``~/.codex``, and deciding how ``thread_spawn_edges``
reach ``session_events`` on the *existing* ``codex-session`` rows without a
full-replace data loss) is out of this module's write scope and is not done
here — see the module docstring of the accompanying test file and the
project report for the precise follow-up.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from polylogue.core.json import JSONDocument

from .base import ParsedSessionEvent

CODEX_STATE_DB_MARKER = "codex_state_db"

CodexSqliteKind: TypeAlias = Literal[
    "thread_state",  # state_5.sqlite -- threads + thread_spawn_edges
    "goals",  # goals_1.sqlite -- thread_goals
    "memories",  # memories_1.sqlite -- stage1_outputs
    "logs",  # logs_2.sqlite -- runtime tracing
    "automation",  # codex-dev.db -- inbox/automation scheduling
    "unknown",
]

CodexAcquisitionDisposition: TypeAlias = Literal["acquire", "acquire-partial", "out-of-scope"]

# Required-table fingerprints used for classification. Checked in this order;
# the first match wins. A file must have ALL tables in the tuple to match.
_KIND_REQUIRED_TABLES: tuple[tuple[CodexSqliteKind, tuple[str, ...]], ...] = (
    ("thread_state", ("threads", "thread_spawn_edges")),
    ("goals", ("thread_goals",)),
    ("memories", ("stage1_outputs",)),
    ("logs", ("logs",)),
    ("automation", ("automations", "automation_runs")),
)


@dataclass(frozen=True, slots=True)
class CodexStateDbClassification:
    """One database's evidence classification and the reason for it."""

    kind: CodexSqliteKind
    disposition: CodexAcquisitionDisposition
    reason: str


# polylogue-0jf4 acceptance criterion 1: classify each of the five databases.
# This is the fidelity declaration content; the canonical home for it is
# ``sources/origin_specs.py:_codex_spec()``'s ``fidelity_notes`` tuple, which
# is outside this module's write scope (see the module docstring) -- these
# are the exact classifications and reasons to fold in there.
CODEX_STATE_FIDELITY: tuple[CodexStateDbClassification, ...] = (
    CodexStateDbClassification(
        kind="thread_state",
        disposition="acquire",
        reason=(
            "threads.title and thread_spawn_edges have no other evidence source: "
            "no Codex rollout JSONL session_meta record carries a curated title or "
            "a parent/child spawn relationship at the orchestration level."
        ),
    ),
    CodexStateDbClassification(
        kind="goals",
        disposition="acquire-partial",
        reason=(
            "thread_goals.objective is stated task intent unavailable anywhere else, "
            "but the table is small (tens of rows) and low-churn; acquire the raw "
            "snapshot for durability, no session_events wiring is included in this "
            "change."
        ),
    ),
    CodexStateDbClassification(
        kind="memories",
        disposition="acquire-partial",
        reason=(
            "stage1_outputs.raw_memory is Codex-side memory with no archive "
            "representation, but its content is a derived summary over content "
            "Polylogue already ingests from the JSONL rollout; acquire the raw "
            "snapshot for durability, no parsed/typed consumption is included in "
            "this change."
        ),
    ),
    CodexStateDbClassification(
        kind="logs",
        disposition="out-of-scope",
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
        reason=(
            "Local CLI automation scheduling config (inbox items, cron-like "
            "automations, automation run records), not AI session content. Empty "
            "on every install observed. automation_runs.thread_id references a "
            "session, but the row itself is scheduling state, not conversation "
            "evidence."
        ),
    ),
)

IN_SCOPE_KINDS: frozenset[CodexSqliteKind] = frozenset(
    classification.kind for classification in CODEX_STATE_FIDELITY if classification.disposition != "out-of-scope"
)


def _connect_readonly(path: Path, *, timeout: float = 1.0, immutable: bool = False) -> sqlite3.Connection:
    """Open *path* read-only. Never takes a write lock against a live Codex."""
    uri = path.resolve().as_uri() + "?mode=ro"
    if immutable:
        uri += "&immutable=1"
    return sqlite3.connect(uri, uri=True, timeout=timeout)


def _table_names(conn: sqlite3.Connection) -> frozenset[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return frozenset(str(row[0]) for row in rows)


def classify_codex_sqlite_path(path: Path, *, immutable: bool = False) -> CodexSqliteKind:
    """Classify a live Codex SQLite file by its table shape.

    Returns ``"unknown"`` for anything unreadable or unrecognized rather than
    raising -- classification runs against a live, possibly-locked file.
    """
    try:
        with closing(_connect_readonly(path, immutable=immutable)) as conn:
            tables = _table_names(conn)
    except sqlite3.Error:
        return "unknown"
    for kind, required in _KIND_REQUIRED_TABLES:
        if required and set(required).issubset(tables):
            return kind
    return "unknown"


def is_in_scope_codex_sqlite_path(path: Path, *, immutable: bool = False) -> bool:
    """Return whether *path* is one of the databases this module acquires."""
    return classify_codex_sqlite_path(path, immutable=immutable) in IN_SCOPE_KINDS


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

    ``raw_memory``/``rollout_summary`` text is intentionally not exposed here
    -- see ``CODEX_STATE_FIDELITY``'s ``"memories"`` disposition
    (acquire-partial: raw bytes are captured for durability, structured
    consumption is deferred). Only structural facts are surfaced.
    """

    thread_id: str
    source_updated_at_ms: int
    generated_at_ms: int
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
        rows = conn.execute(
            "SELECT thread_id, goal_id, objective, status, token_budget, tokens_used, "
            "time_used_seconds, created_at_ms, updated_at_ms FROM thread_goals ORDER BY thread_id"
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
    """Parse structural facts from a Codex ``memories_1.sqlite`` snapshot.

    See ``CodexMemoryRecord`` -- raw memory text is deliberately not surfaced.
    """
    with closing(_connect_readonly(path, immutable=immutable)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT thread_id, source_updated_at, generated_at, usage_count, "
            "rollout_slug, selected_for_phase2 FROM stage1_outputs ORDER BY thread_id"
        ).fetchall()
    return tuple(
        CodexMemoryRecord(
            thread_id=_row_str(row, "thread_id"),
            source_updated_at_ms=_row_int(row, "source_updated_at"),
            generated_at_ms=_row_int(row, "generated_at"),
            usage_count=_row_opt_int(row, "usage_count"),
            has_rollout_slug=_row_opt_str(row, "rollout_slug") is not None,
            selected_for_phase2=bool(_row_int(row, "selected_for_phase2")),
        )
        for row in rows
        if _row_str(row, "thread_id")
    )


_SPAWN_EVENT_TYPE = "codex_thread_spawn_edge"


def spawn_edges_as_session_events(
    edges: Iterable[CodexSpawnEdge],
) -> dict[str, list[ParsedSessionEvent]]:
    """Group spawn edges into per-parent-thread ``ParsedSessionEvent`` lists.

    Ready for a future write-path consumer to attach onto the *existing*
    ``codex-session`` row identified by ``origin:parent_thread_id`` -- see the
    module docstring for why that wiring is not done in this change.
    ``session_events`` accepts a new ``event_type`` value with no index schema
    change (polylogue-0jf4 constraint), which is what this shape targets.
    """
    grouped: dict[str, list[ParsedSessionEvent]] = {}
    for edge in edges:
        grouped.setdefault(edge.parent_thread_id, []).append(
            ParsedSessionEvent(
                event_type=_SPAWN_EVENT_TYPE,
                payload={
                    "parent_thread_id": edge.parent_thread_id,
                    "child_thread_id": edge.child_thread_id,
                    "status": edge.status,
                },
            )
        )
    return grouped


__all__ = [
    "CODEX_STATE_DB_MARKER",
    "CODEX_STATE_FIDELITY",
    "IN_SCOPE_KINDS",
    "CodexAcquisitionDisposition",
    "CodexMemoryRecord",
    "CodexSpawnEdge",
    "CodexSqliteKind",
    "CodexStateDbClassification",
    "CodexStateSnapshot",
    "CodexThreadGoal",
    "CodexThreadRecord",
    "classify_codex_sqlite_path",
    "is_in_scope_codex_sqlite_path",
    "looks_like_state_db_payload",
    "marker_payload",
    "parse_codex_goals_db",
    "parse_codex_memories_db",
    "parse_codex_state_db",
    "spawn_edges_as_session_events",
]
