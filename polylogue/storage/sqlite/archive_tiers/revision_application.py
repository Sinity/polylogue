"""Atomic index receipts for raw-revision replay decisions.

Writer module: index.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from polylogue.archive.revision_replay import ApplicationDecision

_MESSAGE_FTS_TRIGGERS = {"messages_fts_ai", "messages_fts_ad", "messages_fts_au"}


def _hash_prefix(value: bytes | None) -> str:
    """Short hex prefix for a content hash, safe for error messages."""
    if value is None:
        return "None"
    return value.hex()[:12]


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
    fold_authorization: FullSnapshotFoldAuthorization | None = None

    @property
    def decision_id(self) -> str:
        payload = {
            "accepted_raw_id": self.accepted_raw_id,
            "accepted_source_revision": self.accepted_source_revision,
            "accepted_content_hash": self.accepted_content_hash.hex()
            if self.accepted_content_hash is not None
            else None,
            "accepted_frontier_kind": self.accepted_frontier_kind,
            "accepted_frontier": self.accepted_frontier,
            "acquisition_generation": self.acquisition_generation,
            "append_end_offset": self.append_end_offset,
            "baseline_raw_id": self.baseline_raw_id,
            "decision": self.decision.value,
            "logical_source_key": self.logical_source_key,
            "predecessor_raw_id": self.predecessor_raw_id,
            "raw_id": self.raw_id,
            "session_id": self.session_id,
            "source_revision": self.source_revision,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FullSnapshotFoldAuthorization:
    """One-shot authority for a cryptographically proven byte-chain fold.

    This is deliberately an in-memory transaction capability, not a new
    precedence rule.  It binds one incoming full receipt to the exact accepted
    append head it proved, so it cannot authorize a later or different CAS.
    """

    logical_source_key: str
    session_id: str
    accepted_append_raw_id: str
    accepted_append_source_revision: str
    accepted_append_content_hash: bytes
    frontier: int
    full_raw_id: str
    full_source_revision: str

    def permits(self, existing_head: tuple[object, ...], receipt: RevisionApplicationReceipt) -> bool:
        return (
            tuple(existing_head[:6])
            == (
                self.session_id,
                self.accepted_append_raw_id,
                self.accepted_append_source_revision,
                self.accepted_append_content_hash,
                "byte",
                self.frontier,
            )
            and receipt.logical_source_key == self.logical_source_key
            and receipt.decision is ApplicationDecision.SELECTED_BASELINE
            and receipt.session_id == self.session_id
            and receipt.raw_id == self.full_raw_id
            and receipt.accepted_raw_id == self.full_raw_id
            and receipt.source_revision == self.full_source_revision
            and receipt.accepted_source_revision == self.full_source_revision
            and receipt.accepted_frontier_kind == "byte"
            and receipt.accepted_frontier == self.frontier
            and receipt.append_end_offset is None
        )


def assert_session_fts_exact_sync(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    bulk_build: bool = False,
    allow_pending: bool = False,
) -> None:
    """Fail unless the current session's indexable blocks have exact FTS rows.

    ``bulk_build`` (polylogue-v6i3, default ``False``): a bulk-generation-
    build replay deliberately leaves ``messages_fts`` empty for every session
    throughout the build (see ``write_parsed_session_to_archive``'s
    ``bulk_build`` mode) and repopulates it archive-wide exactly once at
    readiness -- so the per-session row-count parity check below does not
    apply mid-build; it is *expected* to be out of sync, not a bug. The
    trigger-presence check still runs unconditionally: the guard row only
    gates trigger BODIES (see ``_bulk_fts_session_guard``), the triggers
    themselves remain structurally present in ``sqlite_master`` throughout,
    so this half of the proof is unaffected by bulk-build mode and stays a
    real check, not a weakened default.
    """
    triggers = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'messages_fts_a%'")
    }
    if not _MESSAGE_FTS_TRIGGERS.issubset(triggers):
        raise RuntimeError("raw revision application requires canonical message FTS triggers")
    if bulk_build or allow_pending:
        return
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


def _record_revision_application_sync(
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
    frontier = (receipt.accepted_frontier_kind, receipt.accepted_frontier)
    if any(value is None for value in frontier) and not all(value is None for value in frontier):
        raise ValueError("accepted frontier receipt fields must be all present or all absent")
    if all(value is None for value in accepted) != all(value is None for value in frontier):
        raise ValueError("accepted identity and frontier receipt fields must be present or absent together")
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO raw_revision_applications (
            decision_id, raw_id, session_id, logical_source_key, source_revision,
            acquisition_generation, decision, accepted_raw_id,
            accepted_source_revision, accepted_content_hash, accepted_frontier_kind,
            accepted_frontier, baseline_raw_id,
            predecessor_raw_id, append_end_offset, detail, decided_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            receipt.accepted_frontier_kind,
            receipt.accepted_frontier,
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
            SELECT raw_id, session_id, logical_source_key, source_revision,
                   acquisition_generation, decision, accepted_raw_id,
                   accepted_source_revision, accepted_content_hash,
                   accepted_frontier_kind, accepted_frontier, baseline_raw_id,
                   predecessor_raw_id, append_end_offset
            FROM raw_revision_applications WHERE decision_id = ?
            """,
            (receipt.decision_id,),
        ).fetchone()
        if existing is not None:
            expected = (
                receipt.raw_id,
                receipt.session_id,
                receipt.logical_source_key,
                receipt.source_revision,
                receipt.acquisition_generation,
                receipt.decision.value,
                receipt.accepted_raw_id,
                receipt.accepted_source_revision,
                receipt.accepted_content_hash,
                receipt.accepted_frontier_kind,
                receipt.accepted_frontier,
                receipt.baseline_raw_id,
                receipt.predecessor_raw_id,
                receipt.append_end_offset,
            )
            if tuple(existing) != expected:
                raise RuntimeError(
                    "conflicting raw revision application receipt: immutable evidence "
                    f"for decision_id={receipt.decision_id} "
                    f"incoming(accepted_content_hash={_hash_prefix(receipt.accepted_content_hash)}, "
                    f"acquisition_generation={receipt.acquisition_generation}, "
                    f"append_end_offset={receipt.append_end_offset}) "
                    f"existing(accepted_content_hash={_hash_prefix(existing[8])}, "
                    f"acquisition_generation={existing[4]!r}, append_end_offset={existing[13]!r})"
                )
        else:
            if receipt.decision is not ApplicationDecision.SUPERSEDED:
                raise RuntimeError(
                    "conflicting raw revision application receipt: no existing row and decision is not SUPERSEDED: "
                    f"decision_id={receipt.decision_id} logical_source_key={receipt.logical_source_key!r} "
                    f"raw_id={receipt.raw_id!r} session_id={receipt.session_id!r} "
                    f"decision={receipt.decision.value!r} accepted_raw_id={receipt.accepted_raw_id!r}"
                )
            semantic_identity = conn.execute(
                """
                SELECT raw_id, session_id, logical_source_key, source_revision,
                       acquisition_generation, decision, accepted_source_revision,
                       accepted_content_hash, accepted_frontier_kind,
                       accepted_frontier, baseline_raw_id, predecessor_raw_id,
                       append_end_offset
                FROM raw_revision_applications
                WHERE raw_id = ? AND session_id = ? AND logical_source_key = ? AND decision = ?
                  AND source_revision = ?
                  AND COALESCE(accepted_source_revision, '') = COALESCE(?, '')
                  AND acquisition_generation = ?
                  AND accepted_content_hash IS ?
                  AND accepted_frontier_kind IS ?
                  AND accepted_frontier IS ?
                  AND baseline_raw_id IS ?
                  AND predecessor_raw_id IS ?
                  AND append_end_offset IS ?
                """,
                (
                    receipt.raw_id,
                    receipt.session_id,
                    receipt.logical_source_key,
                    receipt.decision.value,
                    receipt.source_revision,
                    receipt.accepted_source_revision,
                    receipt.acquisition_generation,
                    receipt.accepted_content_hash,
                    receipt.accepted_frontier_kind,
                    receipt.accepted_frontier,
                    receipt.baseline_raw_id,
                    receipt.predecessor_raw_id,
                    receipt.append_end_offset,
                ),
            ).fetchone()
            expected_identity = (
                receipt.raw_id,
                receipt.session_id,
                receipt.logical_source_key,
                receipt.source_revision,
                receipt.acquisition_generation,
                receipt.decision.value,
                receipt.accepted_source_revision,
                receipt.accepted_content_hash,
                receipt.accepted_frontier_kind,
                receipt.accepted_frontier,
                receipt.baseline_raw_id,
                receipt.predecessor_raw_id,
                receipt.append_end_offset,
            )
            if semantic_identity is None or tuple(semantic_identity) != expected_identity:
                raise RuntimeError(
                    "conflicting raw revision application receipt: no matching semantic-identity row for "
                    f"SUPERSEDED replay: decision_id={receipt.decision_id} "
                    f"logical_source_key={receipt.logical_source_key!r} raw_id={receipt.raw_id!r} "
                    f"session_id={receipt.session_id!r} source_revision={receipt.source_revision!r} "
                    f"accepted_source_revision={receipt.accepted_source_revision!r} "
                    f"accepted_content_hash={_hash_prefix(receipt.accepted_content_hash)} "
                    f"found={semantic_identity!r}"
                )
    if receipt.accepted_raw_id is None or receipt.decision not in {
        ApplicationDecision.SELECTED_BASELINE,
        ApplicationDecision.APPLIED_APPEND,
        ApplicationDecision.REPARSE_REAFFIRMATION,
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
            raise RuntimeError(
                "raw revision CAS rejected an incomparable accepted frontier: "
                f"logical_source_key={receipt.logical_source_key!r} "
                f"existing(session_id={existing_head[0]!r}, accepted_raw_id={existing_head[1]!r}, "
                f"frontier_kind={existing_head[4]!r}, frontier={existing_head[5]!r}) "
                f"incoming(session_id={receipt.session_id!r}, accepted_raw_id={receipt.accepted_raw_id!r}, "
                f"frontier_kind={receipt.accepted_frontier_kind!r}, frontier={receipt.accepted_frontier!r})"
            )
        existing_frontier = int(existing_head[5])
        if receipt.accepted_frontier < existing_frontier:
            raise RuntimeError(
                "raw revision CAS rejected an older accepted frontier: "
                f"logical_source_key={receipt.logical_source_key!r} "
                f"existing(session_id={existing_head[0]!r}, accepted_raw_id={existing_head[1]!r}, "
                f"frontier_kind={existing_head[4]!r}, frontier={existing_frontier}) "
                f"incoming(session_id={receipt.session_id!r}, accepted_raw_id={receipt.accepted_raw_id!r}, "
                f"frontier_kind={receipt.accepted_frontier_kind!r}, frontier={receipt.accepted_frontier})"
            )
        if receipt.accepted_frontier == existing_frontier:
            existing_semantics = (existing_head[0], existing_head[3], existing_head[4], existing_head[5])
            accepted_semantics = (
                receipt.session_id,
                receipt.accepted_content_hash,
                receipt.accepted_frontier_kind,
                receipt.accepted_frontier,
            )
            if existing_semantics != accepted_semantics:
                authorized_fold = receipt.fold_authorization is not None and receipt.fold_authorization.permits(
                    existing_head, receipt
                )
                # Same-raw re-application: the incoming receipt derives from the
                # exact same accepted raw evidence as the existing head (equal
                # `accepted_raw_id`) -- typically a parser/identity fix reparsing
                # the same content-addressed blob into a different content hash
                # and/or session_id. That is re-derivation from identical
                # evidence, not a genuine conflict, so it supersedes the existing
                # head. Cross-raw conflicts (a different `accepted_raw_id`
                # claiming the same frontier with differing semantics) are real
                # divergence and still require fold authorization to pass.
                same_raw_supersede = receipt.accepted_raw_id == existing_head[1]
                if not authorized_fold and not same_raw_supersede:
                    raise RuntimeError(
                        "raw revision CAS rejected a conflicting accepted head: "
                        f"logical_source_key={receipt.logical_source_key!r} "
                        f"existing(session_id={existing_head[0]!r}, accepted_raw_id={existing_head[1]!r}, "
                        f"accepted_content_hash={_hash_prefix(existing_head[3])}, "
                        f"frontier_kind={existing_head[4]!r}, frontier={existing_head[5]!r}) "
                        f"incoming(session_id={receipt.session_id!r}, accepted_raw_id={receipt.accepted_raw_id!r}, "
                        f"accepted_content_hash={_hash_prefix(receipt.accepted_content_hash)}, "
                        f"frontier_kind={receipt.accepted_frontier_kind!r}, frontier={receipt.accepted_frontier!r})"
                    )
            elif tuple(existing_head) == (
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


def record_revision_application_sync(
    conn: sqlite3.Connection,
    receipt: RevisionApplicationReceipt,
    *,
    decided_at_ms: int,
) -> None:
    """Record one receipt and its head CAS as one atomic savepoint operation."""
    savepoint = "raw_revision_application_receipt"
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        _record_revision_application_sync(conn, receipt, decided_at_ms=decided_at_ms)
    except BaseException:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise
    else:
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")


__all__ = [
    "RevisionApplicationReceipt",
    "FullSnapshotFoldAuthorization",
    "assert_session_fts_exact_sync",
    "record_revision_application_sync",
]
