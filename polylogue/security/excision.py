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
   ``excised_content`` for *every distinct blob hash* grouped under that raw
   ingestion's ``ref_id`` -- not just the raw payload's own hash. ``blob_refs``
   shares one ``ref_id`` across ``ref_type IN ('raw_payload', 'attachment',
   'sidecar')``, so a session's inline attachments (whose content hash can
   differ from the raw payload's) get their own non-resurrection marker too.
   That marker is what makes re-ingest non-resurrecting: both acquire-time
   raw-session write functions (``write_source_raw_session`` and
   ``write_source_raw_session_blob_ref`` -- the payload-in-memory and
   blob-ref/streaming routes respectively, shared by the CLI import path and
   the daemon watch path) refuse to re-store a payload whose blob hash is
   recorded there, even after an unrelated ``index.db`` rebuild.
4. Deletes ``user.db`` assertions targeting the excised session/messages/
   blocks (including any prior ``SECRET_CANDIDATE`` finding about that exact
   content -- its whole purpose was pointing at now-gone bytes) and writes
   one durable ``EXCISION_RECORD`` audit receipt.

**Attachments referenced from elsewhere.** ``attachment_refs.session_id``/
``message_id`` carry ``ON DELETE CASCADE`` to ``sessions``/``messages``, so
deleting the excised session's row already removes only *its own*
attachment references. A content-hash-deduplicated ``attachments`` row that
is still referenced by another, non-excised session's ``attachment_refs`` is
untouched -- excision never deletes shared attachment metadata still in
legitimate use elsewhere; it only unlinks the excised session's reference to
it (and, per the point above, marks that raw ingestion's attachment blob
hash as durably excised so an identical copy re-attached under this same
raw ingestion cannot resurrect).

Blob *bytes* are never force-unlinked out from under a lease here. Removing
the ``blob_refs``/``raw_sessions`` rows un-references the blob; the existing
reference-counted blob GC (``polylogue/storage/blob_gc.py``, polylogue-83u)
reclaims the physical bytes on its next run using its own lease discipline.

**Lineage safety.** A session can be a prefix-sharing lineage *parent*
(``session_links``/``branch_point_message_id`` -- see the top-level
architecture notes): a fork/resume/auto-compaction child stores only its own
divergent tail and recomposes its transcript as parent-up-to-branch +
child-tail. Excising such a parent without also handling its dependents
would silently break every dependent child's composed read.
:func:`apply_session_excision` refuses this by default
(:class:`LineageDependentsError`) and only proceeds with
``cascade_lineage=True``, which excises the whole transitive lineage
together so no dependent composed read is left broken.

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
from polylogue.storage.sqlite.connection_profile import DB_TIMEOUT, READ_DB_TIMEOUT
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

# One-shot excision connections open each tier directly (not via
# open_connection/open_readonly_connection) because they must NOT attach
# sibling tiers -- excision deliberately opens and commits one tier at a
# time so a mid-apply failure leaves at most one tier mutated, never a
# half-written cross-tier transaction. They still need the same
# busy_timeout every other write chokepoint in this codebase sets (see
# storage/blob_publication.py's identical one-shot pattern), so every
# connect() below is immediately followed by an explicit PRAGMA using these
# shared timeout constants.
_READ_BUSY_TIMEOUT_MS = READ_DB_TIMEOUT * 1000
_WRITE_BUSY_TIMEOUT_MS = DB_TIMEOUT * 1000


def _connect_ro(path: Path) -> sqlite3.Connection:
    """Open a one-shot read-only tier connection with the shared busy_timeout."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.execute(f"PRAGMA busy_timeout = {_READ_BUSY_TIMEOUT_MS}")
    return conn


def _connect_rw(path: Path) -> sqlite3.Connection:
    """Open a one-shot read-write tier connection with the shared busy_timeout."""
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA busy_timeout = {_WRITE_BUSY_TIMEOUT_MS}")
    return conn


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
        conn = _connect_ro(index_db)
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
        conn = _connect_ro(source_db)
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


def find_lineage_dependents(archive_root: Path, session_id: str) -> tuple[str, ...]:
    """Return every session whose composed transcript depends on ``session_id``.

    Per the lineage-normalization design (`session_links`,
    `branch_point_message_id`): a prefix-sharing fork/resume/auto-compaction
    child stores only its own divergent tail and recomposes its full
    transcript as parent-up-to-branch + child-tail. If ``session_id`` is
    such a parent, deleting its messages/blocks (as excision does) silently
    breaks every dependent child's composed read -- the branch point would
    dangle with no bytes behind it.

    This walks the full *transitive* closure: a grandchild whose immediate
    parent is itself a dependent of ``session_id`` is included too, because
    excising that intermediate parent would break the grandchild the same
    way. Only ``inheritance = 'prefix-sharing'`` edges matter here --
    ``spawned-fresh`` children do not share bytes with their parent, so
    excising the parent does not touch their content.
    """

    index_db = archive_root / "index.db"
    if not index_db.exists():
        return ()

    conn = _connect_ro(index_db)
    try:
        dependents: list[str] = []
        seen = {session_id}
        frontier = [session_id]
        while frontier:
            parent_id = frontier.pop()
            rows = conn.execute(
                "SELECT src_session_id FROM session_links "
                "WHERE resolved_dst_session_id = ? AND inheritance = 'prefix-sharing'",
                (parent_id,),
            ).fetchall()
            for (child_id,) in rows:
                child_id = str(child_id)
                if child_id in seen:
                    continue
                seen.add(child_id)
                dependents.append(child_id)
                frontier.append(child_id)
        return tuple(dependents)
    finally:
        conn.close()


class LineageDependentsError(RuntimeError):
    """Raised when excising a session would break composed reads of its lineage.

    ``session_id`` is a prefix-sharing lineage parent for the listed
    dependent sessions (see :func:`find_lineage_dependents`); excising it
    without also excising them would delete bytes their composed transcripts
    depend on, leaving a dangling ``branch_point_message_id`` with no
    warning. Pass ``cascade_lineage=True`` to :func:`apply_session_excision`
    (CLI: ``--cascade-lineage``) to excise the whole lineage together
    instead.
    """

    def __init__(self, *, session_id: str, dependent_session_ids: tuple[str, ...]) -> None:
        self.session_id = session_id
        self.dependent_session_ids = dependent_session_ids
        joined = ", ".join(dependent_session_ids)
        super().__init__(
            f"session {session_id!r} is a lineage parent for {len(dependent_session_ids)} "
            f"prefix-sharing session(s) that would lose composed content: {joined}. "
            "Pass cascade_lineage=True (CLI: --cascade-lineage) to excise the entire "
            "lineage together, or exclude this session from this run."
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
    lineage_dependent_session_ids: tuple[str, ...] = ()

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
            "lineage_dependent_session_ids": list(self.lineage_dependent_session_ids),
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
        conn = _connect_ro(source_db)
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
        conn = _connect_ro(embeddings_db)
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
        conn = _connect_ro(user_db)
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
        lineage_dependent_session_ids=find_lineage_dependents(archive_root, session_id),
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
    # Populated only when apply_session_excision cascaded across a
    # prefix-sharing lineage (cascade_lineage=True): the other session ids
    # excised alongside session_id, whose per-tier counts are already
    # folded into `counts`/`removed_blob_hashes` above.
    cascaded_session_ids: tuple[str, ...] = ()

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
            "cascaded_session_ids": list(self.cascaded_session_ids),
        }


def _receipt_assertion_id(session_id: str, excised_at_ms: int) -> str:
    digest = hashlib.sha256()
    for part in ("excision-record", session_id, str(excised_at_ms)):
        digest.update(part.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return f"assertion-{AssertionKind.EXCISION_RECORD}:{digest.hexdigest()}"


def _apply_single_session_excision(
    archive_root: Path,
    session_id: str,
    *,
    reason: str,
    actor: str = "user:local",
    now_ms: int | None = None,
) -> ExcisionReceipt:
    """Apply excision to exactly one session: mutate its tiers, write a receipt.

    Low-level primitive -- does NOT check for lineage dependents. Callers
    should use :func:`apply_session_excision`, which adds the lineage-safety
    guard/cascade on top of this. Idempotent: re-applying to an
    already-excised (or never-existing) session id resolves an empty target
    and returns ``found=False`` without touching anything. Re-recording the
    same blob hash's removed-content marker is itself idempotent (``ON
    CONFLICT DO NOTHING`` in ``record_excised_blob_hash``), so a retried
    apply after a partial failure cannot overwrite the original
    reason/actor/timestamp of record.
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
        conn = _connect_rw(embeddings_db)
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
        conn = _connect_rw(index_db)
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
        conn = _connect_rw(source_db)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                for raw_target in target.raw_targets:
                    # blob_refs groups every blob published under this raw
                    # ingestion by shared ref_id -- ref_type IN
                    # ('raw_payload', 'attachment', 'sidecar'). An
                    # attachment's own content hash can differ from the raw
                    # payload's, so read every distinct hash under this
                    # ref_id BEFORE deleting: each one needs its own durable
                    # excised_content marker, or a re-attached copy of that
                    # exact attachment content (elsewhere it happens to be
                    # re-acquired under the same content hash) would not be
                    # recognized as already-excised.
                    sibling_hashes = {
                        bytes(row[0])
                        for row in conn.execute(
                            "SELECT DISTINCT blob_hash FROM blob_refs WHERE ref_id = ?",
                            (raw_target.raw_id,),
                        ).fetchall()
                    }
                    sibling_hashes.add(raw_target.blob_hash)

                    cursor = conn.execute("DELETE FROM blob_refs WHERE ref_id = ?", (raw_target.raw_id,))
                    counts["source_blob_refs"] += max(cursor.rowcount, 0)
                    cursor = conn.execute("DELETE FROM raw_sessions WHERE raw_id = ?", (raw_target.raw_id,))
                    counts["source_raw_rows"] += max(cursor.rowcount, 0)
                    for blob_hash in sibling_hashes:
                        record_excised_blob_hash(
                            conn,
                            blob_hash=blob_hash,
                            reason=reason,
                            actor=actor,
                            prior_revision=raw_target.raw_id,
                            span=None,
                            excised_at_ms=timestamp,
                        )
                        removed_hashes.append(blob_hash.hex())
        finally:
            conn.close()

    user_db = archive_root / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    conn = _connect_rw(user_db)
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


def apply_session_excision(
    archive_root: Path,
    session_id: str,
    *,
    reason: str,
    actor: str = "user:local",
    now_ms: int | None = None,
    cascade_lineage: bool = False,
) -> ExcisionReceipt:
    """Apply excision: mutate all in-scope tiers and write a durable receipt.

    Lineage safety (polylogue-27m fix round): if ``session_id`` is a
    prefix-sharing lineage parent (see :func:`find_lineage_dependents`),
    removing its messages/blocks would silently break the composed
    transcript of every dependent child -- the child's
    ``branch_point_message_id`` would dangle with no bytes behind it. This
    refuses that by default, raising :class:`LineageDependentsError`, and
    only proceeds when ``cascade_lineage=True``, in which case it excises
    ``session_id`` *and* its full transitive prefix-sharing lineage together
    so no dependent composed read is left broken. When there are no
    dependents (the common case), behavior is identical either way.

    Idempotent per the same rules as :func:`_apply_single_session_excision`.
    The returned receipt's ``counts``/``removed_blob_hashes`` are the sum
    across every session actually removed; ``cascaded_session_ids`` lists
    the dependents removed alongside ``session_id`` (empty when there were
    none, or when ``session_id`` itself was already excised/not found).
    """

    target = resolve_session_excision_target(archive_root, session_id)
    if not target.found:
        return ExcisionReceipt(session_id=session_id, found=False)

    dependent_ids = find_lineage_dependents(archive_root, session_id)
    if dependent_ids and not cascade_lineage:
        raise LineageDependentsError(session_id=session_id, dependent_session_ids=dependent_ids)

    timestamp = now_ms if now_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
    cascaded_receipts = tuple(
        _apply_single_session_excision(archive_root, dependent_id, reason=reason, actor=actor, now_ms=timestamp)
        for dependent_id in dependent_ids
    )
    primary = _apply_single_session_excision(archive_root, session_id, reason=reason, actor=actor, now_ms=timestamp)

    actually_cascaded = tuple(receipt.session_id for receipt in cascaded_receipts if receipt.found)
    if not actually_cascaded:
        return primary

    merged_counts = dict(primary.counts)
    merged_removed_hashes = list(primary.removed_blob_hashes)
    for receipt in cascaded_receipts:
        for key, value in receipt.counts.items():
            merged_counts[key] = merged_counts.get(key, 0) + value
        merged_removed_hashes.extend(receipt.removed_blob_hashes)

    return ExcisionReceipt(
        session_id=primary.session_id,
        found=primary.found,
        reason=primary.reason,
        actor=primary.actor,
        excised_at_ms=primary.excised_at_ms,
        receipt_assertion_id=primary.receipt_assertion_id,
        removed_blob_hashes=tuple(merged_removed_hashes),
        counts=merged_counts,
        cascaded_session_ids=actually_cascaded,
    )


__all__ = [
    "ExcisionPlan",
    "ExcisionRawTarget",
    "ExcisionReceipt",
    "ExcisionTarget",
    "LineageDependentsError",
    "apply_session_excision",
    "find_lineage_dependents",
    "plan_session_excision",
    "resolve_session_excision_target",
]
