"""Atomic index receipts for raw-revision replay decisions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from polylogue.archive.revision_replay import ApplicationDecision

_MESSAGE_FTS_TRIGGERS = {"messages_fts_ai", "messages_fts_ad", "messages_fts_au"}


@dataclass(frozen=True, slots=True)
class RevisionApplicationReceipt:
    raw_id: str
    session_id: str
    logical_source_key: str
    source_revision: str
    acquisition_generation: int
    decision: ApplicationDecision
    accepted_raw_id: str | None
    accepted_source_revision: str | None
    accepted_content_hash: bytes | None
    accepted_frontier_kind: str | None = None
    accepted_frontier: int | None = None
    baseline_raw_id: str | None = None
    predecessor_raw_id: str | None = None
    append_end_offset: int | None = None
    detail: str = ""

    @property
    def decision_id(self) -> str:
        payload = {
            "accepted_raw_id": self.accepted_raw_id,
            "accepted_source_revision": self.accepted_source_revision,
            "decision": self.decision.value,
            "logical_source_key": self.logical_source_key,
            "raw_id": self.raw_id,
            "session_id": self.session_id,
            "source_revision": self.source_revision,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def assert_session_fts_exact_sync(conn: sqlite3.Connection, session_id: str) -> None:
    """Fail unless the current session's indexable blocks have exact FTS rows."""
    triggers = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'messages_fts_a%'")
    }
    if not _MESSAGE_FTS_TRIGGERS.issubset(triggers):
        raise RuntimeError("raw revision application requires canonical message FTS triggers")
    expected, indexed = conn.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE b.search_text != ''),
            COUNT(d.id) FILTER (WHERE b.search_text != '')
        FROM blocks AS b
        LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
        WHERE b.session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if int(expected or 0) != int(indexed or 0):
        raise RuntimeError(
            f"raw revision application FTS proof failed for {session_id}: "
            f"expected {int(expected or 0)}, indexed {int(indexed or 0)}"
        )


def record_revision_application_sync(
    conn: sqlite3.Connection,
    receipt: RevisionApplicationReceipt,
    *,
    decided_at_ms: int,
) -> None:
    """Insert one immutable receipt and CAS the accepted logical head."""
    accepted = (
        receipt.accepted_raw_id,
        receipt.accepted_source_revision,
        receipt.accepted_content_hash,
    )
    if any(value is None for value in accepted) and not all(value is None for value in accepted):
        raise ValueError("accepted revision receipt fields must be all present or all absent")
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO raw_revision_applications (
            decision_id, raw_id, session_id, logical_source_key, source_revision,
            acquisition_generation, decision, accepted_raw_id,
            accepted_source_revision, accepted_content_hash, baseline_raw_id,
            predecessor_raw_id, append_end_offset, detail, decided_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            receipt.decision_id,
            receipt.raw_id,
            receipt.session_id,
            receipt.logical_source_key,
            receipt.source_revision,
            receipt.acquisition_generation,
            receipt.decision.value,
            receipt.accepted_raw_id,
            receipt.accepted_source_revision,
            receipt.accepted_content_hash,
            receipt.baseline_raw_id,
            receipt.predecessor_raw_id,
            receipt.append_end_offset,
            receipt.detail,
            decided_at_ms,
        ),
    )
    if cursor.rowcount == 0:
        existing = conn.execute(
            """
            SELECT raw_id, session_id, decision, accepted_raw_id,
                   accepted_source_revision, accepted_content_hash
            FROM raw_revision_applications WHERE decision_id = ?
            """,
            (receipt.decision_id,),
        ).fetchone()
        expected = (
            receipt.raw_id,
            receipt.session_id,
            receipt.decision.value,
            receipt.accepted_raw_id,
            receipt.accepted_source_revision,
            receipt.accepted_content_hash,
        )
        if existing is None or tuple(existing) != expected:
            raise RuntimeError(f"conflicting raw revision application receipt: {receipt.decision_id}")
    if receipt.accepted_raw_id is None or receipt.decision not in {
        ApplicationDecision.SELECTED_BASELINE,
        ApplicationDecision.APPLIED_APPEND,
    }:
        return
    assert receipt.accepted_source_revision is not None
    assert receipt.accepted_content_hash is not None
    existing_head = conn.execute(
        """
        SELECT session_id, accepted_raw_id, accepted_source_revision,
               accepted_content_hash, accepted_frontier_kind, accepted_frontier,
               acquisition_generation, append_end_offset
        FROM raw_revision_heads WHERE logical_source_key = ?
        """,
        (receipt.logical_source_key,),
    ).fetchone()
    if existing_head is not None:
        if receipt.accepted_frontier_kind not in {"byte", "semantic"} or receipt.accepted_frontier is None:
            raise ValueError("accepted revision receipt requires a typed frontier")
        if str(existing_head[4]) != receipt.accepted_frontier_kind:
            raise RuntimeError("raw revision CAS rejected an incomparable accepted frontier")
        existing_frontier = int(existing_head[5])
        if receipt.accepted_frontier < existing_frontier:
            raise RuntimeError("raw revision CAS rejected an older accepted frontier")
        if receipt.accepted_frontier == existing_frontier:
            existing_semantics = (existing_head[0], existing_head[3], existing_head[4], existing_head[5])
            accepted_semantics = (
                receipt.session_id,
                receipt.accepted_content_hash,
                receipt.accepted_frontier_kind,
                receipt.accepted_frontier,
            )
            if existing_semantics != accepted_semantics:
                raise RuntimeError("raw revision CAS rejected a conflicting accepted head")
            if tuple(existing_head) == (
                receipt.session_id,
                receipt.accepted_raw_id,
                receipt.accepted_source_revision,
                receipt.accepted_content_hash,
                receipt.accepted_frontier_kind,
                receipt.accepted_frontier,
                receipt.acquisition_generation,
                receipt.append_end_offset,
            ):
                return
    conn.execute(
        """
        INSERT INTO raw_revision_heads (
            logical_source_key, session_id, accepted_raw_id,
            accepted_source_revision, accepted_content_hash,
            accepted_frontier_kind, accepted_frontier,
            acquisition_generation, append_end_offset, decided_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(logical_source_key) DO UPDATE SET
            session_id = excluded.session_id,
            accepted_raw_id = excluded.accepted_raw_id,
            accepted_source_revision = excluded.accepted_source_revision,
            accepted_content_hash = excluded.accepted_content_hash,
            accepted_frontier_kind = excluded.accepted_frontier_kind,
            accepted_frontier = excluded.accepted_frontier,
            acquisition_generation = excluded.acquisition_generation,
            append_end_offset = excluded.append_end_offset,
            decided_at_ms = excluded.decided_at_ms
        """,
        (
            receipt.logical_source_key,
            receipt.session_id,
            receipt.accepted_raw_id,
            receipt.accepted_source_revision,
            receipt.accepted_content_hash,
            receipt.accepted_frontier_kind,
            receipt.accepted_frontier,
            receipt.acquisition_generation,
            receipt.append_end_offset,
            decided_at_ms,
        ),
    )


__all__ = [
    "RevisionApplicationReceipt",
    "assert_session_fts_exact_sync",
    "record_revision_application_sync",
]
