"""Standalone/off-mode local excision (polylogue-27m).

"The archive can forget on purpose": in standalone/off mode (no Sinex-backed
replica to reconcile), local excision is *authoritative*. Applying an
excision:

1. Deletes the session's vectors from ``embeddings.db`` (if embedded).
2. Deletes the session from ``index.db`` -- ``sessions`` cascades to
   ``messages``/``blocks``/``session_links`` via ``ON DELETE CASCADE``, and
   the FTS triggers clean the contentless search index.
3. Deletes the session's ``blob_refs`` and ``raw_sessions`` rows from
   ``source.db`` (cascading to ``raw_session_memberships``/
   ``raw_membership_census``), then records a durable removed-hash marker in
   ``excised_content`` per blob. That marker is what makes re-ingest
   non-resurrecting: ``write_source_raw_session`` (the acquire-time write
   choke point shared by the CLI import path and the daemon watch path)
   refuses to re-store a payload whose blob hash is recorded there, even
   after an unrelated ``index.db`` rebuild.
4. Deletes ``user.db`` assertions targeting the excised session/messages/
   blocks (including any prior ``SECRET_CANDIDATE`` finding about that exact
   content -- its whole purpose was pointing at now-gone bytes) and writes
   one durable ``EXCISION_RECORD`` audit receipt.

Blob *bytes* are never force-unlinked out from under a lease here. Removing
the ``blob_refs``/``raw_sessions`` rows un-references the blob; the existing
reference-counted blob GC (``polylogue/storage/blob_gc.py``, polylogue-83u)
reclaims the physical bytes on its next run using its own lease discipline.

Mirror/primary-mode lifecycle mechanics (durable request/outbox, fault
injection against a versioned contract fake) live in
:mod:`polylogue.security.lifecycle`. This module is the off/standalone path
only -- see ``docs/security.md`` for the full mode matrix and non-goals.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import (
    is_blob_hash_excised,
    record_excised_blob_hash,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec


@dataclass(frozen=True, slots=True)
class ExcisionRawTarget:
    """One raw acquisition backing the excised session."""

    raw_id: str
    blob_hash: bytes
    source_path: str


@dataclass(frozen=True, slots=True)
class ExcisionTarget:
    """Rows resolved as in-scope for excising one session.

    Resolved once, up front, so a dry-run preview and the real mutation act
    on the identical row set (mirrors the ``reset --session`` fix, jnj.5).
    """

    session_id: str
    raw_targets: tuple[ExcisionRawTarget, ...] = ()
    message_ids: tuple[str, ...] = ()
    block_ids: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        return bool(self.raw_targets or self.message_ids or self.block_ids)


def resolve_session_excision_target(archive_root: Path, session_id: str) -> ExcisionTarget:
    """Resolve the exact rows an excision of ``session_id`` would touch."""

    index_db = archive_root / "index.db"
    source_db = archive_root / "source.db"

    raw_ids: list[str] = []
    message_ids: tuple[str, ...] = ()
    block_ids: tuple[str, ...] = ()

    if index_db.exists():
        conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT raw_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is not None and row[0]:
                raw_ids.append(str(row[0]))
            message_ids = tuple(
                str(r[0])
                for r in conn.execute(
                    "SELECT message_id FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchall()
            )
            block_ids = tuple(
                str(r[0])
                for r in conn.execute(
                    "SELECT block_id FROM blocks WHERE session_id = ?",
                    (session_id,),
                ).fetchall()
            )
        finally:
            conn.close()

    raw_targets: tuple[ExcisionRawTarget, ...] = ()
    if source_db.exists() and raw_ids:
        conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        try:
            placeholders = ",".join("?" for _ in raw_ids)
            rows = conn.execute(
                f"SELECT raw_id, blob_hash, source_path FROM raw_sessions WHERE raw_id IN ({placeholders})",
                raw_ids,
            ).fetchall()
            raw_targets = tuple(
                ExcisionRawTarget(raw_id=str(r[0]), blob_hash=bytes(r[1]), source_path=str(r[2])) for r in rows
            )
        finally:
            conn.close()

    return ExcisionTarget(
        session_id=session_id,
        raw_targets=raw_targets,
        message_ids=message_ids,
        block_ids=block_ids,
    )


def _target_refs(target: ExcisionTarget) -> list[str]:
    refs = [f"session:{target.session_id}"]
    refs.extend(f"message:{message_id}" for message_id in target.message_ids)
    refs.extend(f"block:{block_id}" for block_id in target.block_ids)
    return refs


@dataclass(frozen=True, slots=True)
class ExcisionPlan:
    """Dry-run preview: exact per-tier counts an apply would touch."""

    session_id: str
    found: bool
    source_raw_rows: int = 0
    source_blob_refs: int = 0
    index_sessions: int = 0
    index_messages: int = 0
    index_blocks: int = 0
    embeddings_vectors: int = 0
    user_assertions: int = 0
    already_excised_blob_hashes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "found": self.found,
            "source_raw_rows": self.source_raw_rows,
            "source_blob_refs": self.source_blob_refs,
            "index_sessions": self.index_sessions,
            "index_messages": self.index_messages,
            "index_blocks": self.index_blocks,
            "embeddings_vectors": self.embeddings_vectors,
            "user_assertions": self.user_assertions,
            "already_excised_blob_hashes": list(self.already_excised_blob_hashes),
        }


def plan_session_excision(archive_root: Path, session_id: str) -> ExcisionPlan:
    """Enumerate exactly what an apply would remove, without mutating anything."""

    target = resolve_session_excision_target(archive_root, session_id)
    if not target.found:
        return ExcisionPlan(session_id=session_id, found=False)

    source_db = archive_root / "source.db"
    embeddings_db = archive_root / "embeddings.db"
    user_db = archive_root / "user.db"

    source_blob_refs = 0
    already_excised: list[str] = []
    if source_db.exists() and target.raw_targets:
        conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        try:
            for raw_target in target.raw_targets:
                row = conn.execute(
                    "SELECT COUNT(*) FROM blob_refs WHERE ref_id = ?",
                    (raw_target.raw_id,),
                ).fetchone()
                source_blob_refs += int(row[0]) if row else 0
                if is_blob_hash_excised(conn, raw_target.blob_hash):
                    already_excised.append(raw_target.blob_hash.hex())
        finally:
            conn.close()

    embeddings_vectors = 0
    if embeddings_db.exists() and target.message_ids:
        conn = sqlite3.connect(f"file:{embeddings_db}?mode=ro", uri=True)
        try:
            try_load_sqlite_vec(conn)
            placeholders = ",".join("?" for _ in target.message_ids)
            row = conn.execute(
                f"SELECT COUNT(*) FROM message_embeddings WHERE message_id IN ({placeholders})",
                target.message_ids,
            ).fetchone()
            embeddings_vectors = int(row[0]) if row else 0
        finally:
            conn.close()

    user_assertions = 0
    if user_db.exists():
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        try:
            refs = _target_refs(target)
            placeholders = ",".join("?" for _ in refs)
            row = conn.execute(
                f"SELECT COUNT(*) FROM assertions WHERE target_ref IN ({placeholders})",
                refs,
            ).fetchone()
            user_assertions = int(row[0]) if row else 0
        finally:
            conn.close()

    return ExcisionPlan(
        session_id=session_id,
        found=True,
        source_raw_rows=len(target.raw_targets),
        source_blob_refs=source_blob_refs,
        index_sessions=1,
        index_messages=len(target.message_ids),
        index_blocks=len(target.block_ids),
        embeddings_vectors=embeddings_vectors,
        user_assertions=user_assertions,
        already_excised_blob_hashes=tuple(already_excised),
    )


@dataclass(frozen=True, slots=True)
class ExcisionReceipt:
    """Result of an apply: exact per-tier counts actually removed."""

    session_id: str
    found: bool
    reason: str | None = None
    actor: str | None = None
    excised_at_ms: int | None = None
    receipt_assertion_id: str | None = None
    removed_blob_hashes: tuple[str, ...] = ()
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "found": self.found,
            "reason": self.reason,
            "actor": self.actor,
            "excised_at_ms": self.excised_at_ms,
            "receipt_assertion_id": self.receipt_assertion_id,
            "removed_blob_hashes": list(self.removed_blob_hashes),
            "counts": dict(self.counts),
        }


def _receipt_assertion_id(session_id: str, excised_at_ms: int) -> str:
    digest = hashlib.sha256()
    for part in ("excision-record", session_id, str(excised_at_ms)):
        digest.update(part.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return f"assertion-{AssertionKind.EXCISION_RECORD}:{digest.hexdigest()}"


def apply_session_excision(
    archive_root: Path,
    session_id: str,
    *,
    reason: str,
    actor: str = "user:local",
    now_ms: int | None = None,
) -> ExcisionReceipt:
    """Apply excision: mutate all in-scope tiers and write a durable receipt.

    Idempotent: re-applying to an already-excised (or never-existing)
    session id resolves an empty target and returns ``found=False`` without
    touching anything. Re-recording the same blob hash's removed-content
    marker is itself idempotent (``ON CONFLICT DO NOTHING`` in
    ``record_excised_blob_hash``), so a retried apply after a partial
    failure cannot overwrite the original reason/actor/timestamp of record.
    """

    timestamp = now_ms if now_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
    target = resolve_session_excision_target(archive_root, session_id)
    if not target.found:
        return ExcisionReceipt(session_id=session_id, found=False)

    counts: dict[str, int] = {
        "embeddings_vectors": 0,
        "index_sessions": 0,
        "index_messages": len(target.message_ids),
        "index_blocks": len(target.block_ids),
        "source_blob_refs": 0,
        "source_raw_rows": 0,
        "user_assertions_removed": 0,
    }

    embeddings_db = archive_root / "embeddings.db"
    if embeddings_db.exists() and target.message_ids:
        conn = sqlite3.connect(embeddings_db)
        try:
            try_load_sqlite_vec(conn)
            with conn:
                placeholders = ",".join("?" for _ in target.message_ids)
                cursor = conn.execute(
                    f"DELETE FROM message_embeddings WHERE message_id IN ({placeholders})",
                    target.message_ids,
                )
                counts["embeddings_vectors"] = max(cursor.rowcount, 0)
                conn.execute(
                    f"DELETE FROM message_embeddings_meta WHERE message_id IN ({placeholders})",
                    target.message_ids,
                )
                conn.execute("DELETE FROM embedding_status WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM embedding_failures WHERE session_id = ?", (session_id,))
        finally:
            conn.close()

    index_db = archive_root / "index.db"
    if index_db.exists():
        conn = sqlite3.connect(index_db)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                counts["index_sessions"] = max(cursor.rowcount, 0)
        finally:
            conn.close()

    source_db = archive_root / "source.db"
    removed_hashes: list[str] = []
    if source_db.exists() and target.raw_targets:
        conn = sqlite3.connect(source_db)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                for raw_target in target.raw_targets:
                    cursor = conn.execute("DELETE FROM blob_refs WHERE ref_id = ?", (raw_target.raw_id,))
                    counts["source_blob_refs"] += max(cursor.rowcount, 0)
                    cursor = conn.execute("DELETE FROM raw_sessions WHERE raw_id = ?", (raw_target.raw_id,))
                    counts["source_raw_rows"] += max(cursor.rowcount, 0)
                    record_excised_blob_hash(
                        conn,
                        blob_hash=raw_target.blob_hash,
                        reason=reason,
                        actor=actor,
                        prior_revision=raw_target.raw_id,
                        span=None,
                        excised_at_ms=timestamp,
                    )
                    removed_hashes.append(raw_target.blob_hash.hex())
        finally:
            conn.close()

    user_db = archive_root / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    conn = sqlite3.connect(user_db)
    try:
        with conn:
            refs = _target_refs(target)
            removed_assertions = 0
            for ref in refs:
                cursor = conn.execute("DELETE FROM assertions WHERE target_ref = ?", (ref,))
                removed_assertions += max(cursor.rowcount, 0)
            counts["user_assertions_removed"] = removed_assertions

            receipt_id = _receipt_assertion_id(session_id, timestamp)
            from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

            upsert_assertion(
                conn,
                assertion_id=receipt_id,
                target_ref=f"session:{session_id}",
                kind=AssertionKind.EXCISION_RECORD,
                value={
                    "reason": reason,
                    "actor": actor,
                    "mode": "standalone",
                    "removed_blob_hashes": removed_hashes,
                    "counts": counts,
                    "excised_at_ms": timestamp,
                },
                author_ref="user:local",
                author_kind="user",
                status=AssertionStatus.ACTIVE,
                visibility=AssertionVisibility.PRIVATE,
                context_policy={"inject": False},
                now_ms=timestamp,
            )
    finally:
        conn.close()

    return ExcisionReceipt(
        session_id=session_id,
        found=True,
        reason=reason,
        actor=actor,
        excised_at_ms=timestamp,
        receipt_assertion_id=receipt_id,
        removed_blob_hashes=tuple(removed_hashes),
        counts=counts,
    )


__all__ = [
    "ExcisionPlan",
    "ExcisionRawTarget",
    "ExcisionReceipt",
    "ExcisionTarget",
    "apply_session_excision",
    "plan_session_excision",
    "resolve_session_excision_target",
]
