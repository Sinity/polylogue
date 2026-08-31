"""Session-linked read model over admitted ``todo_snapshot`` artifacts (polylogue-t0p).

Every ``~/.claude/todos/*.json`` file Claude Code writes is admitted as raw,
provenance-carrying evidence via the ``todo_snapshot`` ``OriginArtifactRule``
(``polylogue.sources.origin_specs``) exactly like Workflow's own fact
artifacts -- ``source.db`` remains the sole authority and every raw revision
the watcher ever observed is retained (``raw_sessions``); ``raw_artifacts``
records the current classification per source path.

This module is the read side: it loads every currently-classified
``todo_snapshot`` row, parses its payload, and groups the result by session so
a consumer gets one :class:`ClaudeTodoPlanState` per session -- the agent's
CURRENT plan plus, when the watcher captured more than one snapshot for the
same session, a per-item status-transition history. Deliberately
storage-free (mirrors ``insights/run_projection.py``'s own justification):
this is a pure read-time projection over already-admitted raw evidence, not a
new durable table, because there is no measured query pressure yet that would
justify one.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.enums import Origin, Provider
from polylogue.core.refs import ObjectRef
from polylogue.sources.origin_specs import artifact_rule_for_path
from polylogue.sources.parsers.claude.todos import ClaudeTodoSnapshot, parse_claude_todo_artifact
from polylogue.storage.blob_store import BlobStore


@dataclass(frozen=True, slots=True)
class ClaudeTodoSnapshotEvidence:
    """One observed snapshot plus its acquisition time and evidence pointer."""

    snapshot: ClaudeTodoSnapshot
    acquired_at_ms: int
    evidence_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class ClaudeTodoPlanState:
    """One session's plan history: every observed snapshot, oldest first."""

    session_id: str
    agent_id: str | None
    snapshots: tuple[ClaudeTodoSnapshotEvidence, ...]

    @property
    def latest(self) -> ClaudeTodoSnapshotEvidence:
        return self.snapshots[-1]

    @property
    def plan_completion_rate(self) -> float | None:
        """The construct ``metric:plan_completion_rate`` reports (see ``registered_metrics.py``)."""
        return self.latest.snapshot.completion_rate

    def status_transitions(self) -> dict[str, tuple[str, ...]]:
        """Per-item-id status sequence across every observed snapshot, oldest first.

        Only tracks items with a stable ``id`` (Claude Code always assigns
        one; a malformed entry without one is excluded -- there is no
        identity to track a transition against). A single-snapshot session
        still returns one-element sequences: "no transition observed yet" is
        a real, distinct fact from "this item flipped status".
        """
        history: dict[str, list[str]] = {}
        for evidence in self.snapshots:
            for item in evidence.snapshot.items:
                if item.item_id is None:
                    continue
                history.setdefault(item.item_id, []).append(item.status)
        return {item_id: tuple(statuses) for item_id, statuses in history.items()}


def load_claude_todo_plan_states(archive_root: Path) -> tuple[ClaudeTodoPlanState, ...]:
    """Materialize every session's plan state from EVERY retained ``todo_snapshot`` raw revision.

    Reads ``raw_sessions`` directly rather than the current-revision-only
    ``raw_artifacts`` pointer table: Claude Code overwrites the same path on
    every ``TodoWrite`` call, but the archive's mutable-file admission path
    retains every observed revision as its own ``raw_sessions`` row (see
    ``docs/internals.md``'s revision-retention model). Reading only the
    current pointer would collapse a session's whole plan history down to
    its latest snapshot, defeating the point of a status-transition read
    model.
    """
    source_db = archive_root / "source.db"
    if not source_db.exists():
        return ()

    with sqlite3.connect(source_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT raw_id, source_path, lower(hex(blob_hash)) AS blob_hash, acquired_at_ms
            FROM raw_sessions
            WHERE origin = ?
            ORDER BY acquired_at_ms ASC, rowid ASC
            """,
            (Origin.CLAUDE_CODE_SESSION.value,),
        ).fetchall()

    if not rows:
        return ()

    blob_store = BlobStore(archive_root / "blob")
    by_session: dict[tuple[str, str | None], list[ClaudeTodoSnapshotEvidence]] = {}
    for row in rows:
        source_path = str(row["source_path"])
        rule = artifact_rule_for_path(Provider.CLAUDE_CODE, source_path)
        if rule is None or rule.kind != "todo_snapshot":
            continue
        payload = blob_store.read_all(str(row["blob_hash"]))
        snapshot = parse_claude_todo_artifact(source_path, payload)
        if snapshot is None or snapshot.session_id is None:
            continue
        key = (snapshot.session_id, snapshot.agent_id)
        by_session.setdefault(key, []).append(
            ClaudeTodoSnapshotEvidence(
                snapshot=snapshot,
                acquired_at_ms=int(row["acquired_at_ms"]),
                evidence_ref=ObjectRef(kind="artifact", object_id=f"raw:{row['raw_id']}"),
            )
        )

    return tuple(
        ClaudeTodoPlanState(session_id=session_id, agent_id=agent_id, snapshots=tuple(evidences))
        for (session_id, agent_id), evidences in sorted(by_session.items())
    )


__all__ = [
    "ClaudeTodoPlanState",
    "ClaudeTodoSnapshotEvidence",
    "load_claude_todo_plan_states",
]
