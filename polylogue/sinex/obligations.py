"""Durable CRUD over ``source.db.sinex_publication_obligations``.

Every function here operates on an already-open ``sqlite3.Connection`` and
never calls ``commit()``/``rollback()`` itself -- the caller controls the
transaction boundary. This is deliberate: design polylogue-303r.2 requires
the obligation row to be created "in the same durable source-tier transaction
that records the acquired/normalized revision", so obligation creation must
be composable into a caller's existing transaction, not own its own.
"""

from __future__ import annotations

import sqlite3

from polylogue.sinex.models import (
    ObligationStatus,
    PublicationMode,
    PublicationObligation,
    ReceiptState,
)

_COLUMNS = (
    "object_id",
    "protocol_version",
    "revision_id",
    "manifest_digest",
    "mode",
    "status",
    "attempt_count",
    "last_attempt_at_ms",
    "last_receipt_state",
    "last_error",
    "created_at_ms",
    "updated_at_ms",
    "retired_at_ms",
)


def _row_to_obligation(row: sqlite3.Row) -> PublicationObligation:
    return PublicationObligation(
        object_id=str(row["object_id"]),
        protocol_version=str(row["protocol_version"]),
        revision_id=str(row["revision_id"]),
        manifest_digest=str(row["manifest_digest"]),
        mode=PublicationMode.from_string(str(row["mode"])),
        status=ObligationStatus.from_string(str(row["status"])),
        attempt_count=int(row["attempt_count"]),
        last_attempt_at_ms=(int(row["last_attempt_at_ms"]) if row["last_attempt_at_ms"] is not None else None),
        last_receipt_state=(
            ReceiptState(str(row["last_receipt_state"])) if row["last_receipt_state"] is not None else None
        ),
        last_error=(str(row["last_error"]) if row["last_error"] is not None else None),
        created_at_ms=int(row["created_at_ms"]),
        updated_at_ms=int(row["updated_at_ms"]),
        retired_at_ms=(int(row["retired_at_ms"]) if row["retired_at_ms"] is not None else None),
    )


def record_obligation(
    conn: sqlite3.Connection,
    *,
    object_id: str,
    protocol_version: str,
    revision_id: str,
    manifest_digest: str,
    mode: PublicationMode,
    now_ms: int,
) -> PublicationObligation:
    """Idempotently create (or return the existing) obligation for a revision.

    ``mode`` must not be :attr:`PublicationMode.OFF` -- callers gate obligation
    creation on mode before calling this (off mode performs zero durable
    writes and zero transport work, per design). INSERT OR IGNORE makes a
    retried call for the *same* revision a true no-op: the idempotency key is
    the primary key, so a duplicate obligation is structurally impossible.
    """
    if mode is PublicationMode.OFF:
        raise ValueError("record_obligation must not be called in off mode")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        INSERT OR IGNORE INTO sinex_publication_obligations (
            object_id, protocol_version, revision_id, manifest_digest, mode,
            status, attempt_count, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?)
        """,
        (object_id, protocol_version, revision_id, manifest_digest, mode.value, now_ms, now_ms),
    )
    existing = get_obligation(
        conn,
        object_id=object_id,
        protocol_version=protocol_version,
        revision_id=revision_id,
        manifest_digest=manifest_digest,
    )
    assert existing is not None
    return existing


def get_obligation(
    conn: sqlite3.Connection,
    *,
    object_id: str,
    protocol_version: str,
    revision_id: str,
    manifest_digest: str,
) -> PublicationObligation | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        f"""
        SELECT {", ".join(_COLUMNS)} FROM sinex_publication_obligations
        WHERE object_id = ? AND protocol_version = ? AND revision_id = ? AND manifest_digest = ?
        """,
        (object_id, protocol_version, revision_id, manifest_digest),
    ).fetchone()
    return _row_to_obligation(row) if row is not None else None


def list_obligations(
    conn: sqlite3.Connection,
    *,
    statuses: tuple[ObligationStatus, ...] | None = None,
    object_id: str | None = None,
) -> tuple[PublicationObligation, ...]:
    """List obligations, optionally filtered by status set and/or object."""
    conn.row_factory = sqlite3.Row
    clauses: list[str] = []
    params: list[object] = []
    if statuses is not None:
        placeholders = ",".join("?" for _ in statuses)
        clauses.append(f"status IN ({placeholders})")
        params.extend(status.value for status in statuses)
    if object_id is not None:
        clauses.append("object_id = ?")
        params.append(object_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT {", ".join(_COLUMNS)} FROM sinex_publication_obligations
        {where}
        ORDER BY created_at_ms, object_id
        """,
        params,
    ).fetchall()
    return tuple(_row_to_obligation(row) for row in rows)


def mark_attempt(
    conn: sqlite3.Connection,
    obligation: PublicationObligation,
    *,
    status: ObligationStatus,
    receipt_state: ReceiptState | None,
    error: str | None,
    now_ms: int,
) -> PublicationObligation:
    """Record one transport attempt outcome and advance obligation status.

    Always increments ``attempt_count`` -- this is the durable evidence that
    a retry happened, independent of the disposable ``ops.db`` diagnostics a
    caller may also choose to record.
    """
    retired_at_ms = now_ms if status in (ObligationStatus.CONFIRMED, ObligationStatus.REJECTED) else None
    conn.execute(
        """
        UPDATE sinex_publication_obligations
        SET status = ?, attempt_count = attempt_count + 1, last_attempt_at_ms = ?,
            last_receipt_state = ?, last_error = ?, updated_at_ms = ?, retired_at_ms = ?
        WHERE object_id = ? AND protocol_version = ? AND revision_id = ? AND manifest_digest = ?
        """,
        (
            status.value,
            now_ms,
            receipt_state.value if receipt_state is not None else None,
            error,
            now_ms,
            retired_at_ms,
            obligation.object_id,
            obligation.protocol_version,
            obligation.revision_id,
            obligation.manifest_digest,
        ),
    )
    updated = get_obligation(
        conn,
        object_id=obligation.object_id,
        protocol_version=obligation.protocol_version,
        revision_id=obligation.revision_id,
        manifest_digest=obligation.manifest_digest,
    )
    assert updated is not None
    return updated


__all__ = ["get_obligation", "list_obligations", "mark_attempt", "record_obligation"]
