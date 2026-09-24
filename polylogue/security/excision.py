"""Standalone/off-mode local excision (polylogue-27m).

"The archive can forget on purpose": in standalone/off mode (no Sinex-backed
replica to reconcile), local excision is *authoritative*. Applying an
excision:

1. Deletes the session's vectors from ``embeddings.db`` (if embedded).
2. Deletes the session's ``blob_refs`` and ``raw_sessions`` rows from
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
3. Deletes the session's durable hook events (``raw_hook_events`` +
   ``hook_event_carriers`` + their ``hook_payload`` blob refs, through
   ``delete_source_hook_event``) and records an ``excised_content`` marker
   for every blob hash they owned. Hook payloads are session-addressable by
   ``(origin, session_native_id)`` and carry no ``raw_sessions`` row, so no
   raw target above reaches them; without this step a completed excision
   left every PreToolUse/PostToolUse payload readable (polylogue-bhhsa).
   Whatever is still readable after the commit is named on the receipt as
   ``retained_hook_events`` and makes ``ExcisionReceipt.complete`` false.
4. Disposes the manifest containers holding those raw acquisitions per
   member (``source_item_raw_members`` + ``source_items``), deletes the
   session's telemetry spans (``otlp_spans``, addressed by the same
   ``(origin, session_native_id)`` key as hook events) and drops any
   ``blob_publication_reservations`` still reserving a now-excised hash.
   See :class:`ContainerDisposition` and
   :mod:`polylogue.security.excision_carriers`.
5. Deletes ``user.db`` assertions targeting the excised session/messages/
   blocks (including any prior ``SECRET_CANDIDATE`` finding about that exact
   content -- its whole purpose was pointing at now-gone bytes) and writes
   one durable ``EXCISION_RECORD`` audit receipt.
6. Deletes the session from ``index.db`` -- ``sessions`` cascades to
   ``messages``/``blocks``/``session_links`` via ``ON DELETE CASCADE``, and
   the FTS triggers clean the contentless search index.

**Declared reach.** Which source-tier relations an excision must reach is
derived from the live schema, not from a hand-kept list:
:func:`polylogue.security.excision_carriers.audit_session_carriers` finds
every table carrying a session key (a ``raw_id``, or an ``(origin,
session_native_id)`` pair) and refuses the excision when one of them has no
declared reach, instead of skipping it into a success receipt
(polylogue-9lrqs, polylogue-14ucm).

**Fact-tier evidence.** Artifacts admitted with ``parse_policy='fact'``
get their own ``raw_sessions`` row but mint no ``sessions`` row, so
``sessions.raw_id`` never names them. Claude Code's
``todos/<session-uuid>[-agent-<uuid>].json`` plan snapshots are linked to
their session only by the identity in their own filename, which is why
excising a session used to leave its plan text readable under the excised
session id (polylogue-si5kj). ``resolve_session_excision_target`` resolves
them from that declared identity and folds them into the seed set *before*
the revision closure runs, so every retained revision of the same plan file
is covered too.

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
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility, Origin, Provider
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.security.excision_carriers import (
    UnclassifiedSessionCarrierError,
    audit_session_carriers,
)
from polylogue.security.excision_policy import (
    ExcisionPolicyError,
    ExcisionPolicySnapshot,
    build_excision_policy_snapshot,
)
from polylogue.sources.origin_specs import artifact_rule_for_path
from polylogue.sources.parsers.claude.todos import session_and_agent_id_from_filename
from polylogue.storage.accepted_marker_inputs import (
    MarkerInputExcisionTarget,
    excise_marker_input_targets_sync,
    marker_input_excision_targets_sync,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import (
    delete_source_hook_event,
    is_blob_hash_excised,
    record_excised_blob_hash,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import (
    READ_PROFILES,
    open_isolated_write_connection,
    open_profiled_connection,
)
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

#: Bound on the revision-closure fixpoint. Each pass can only grow the set,
#: and real revision chains are short; the bound keeps a corrupted
#: predecessor cycle from looping instead of returning what it resolved.
_REVISION_CLOSURE_PASSES = 16

# Excision opens and commits one tier at a time so a mid-apply failure leaves
# at most one tier mutated, never a half-written cross-tier transaction. Both
# factories below are the declared no-sibling-attach routes for exactly that.


def _connect_ro(path: Path) -> sqlite3.Connection:
    """Open a one-shot read-only tier connection with the background profile."""
    return open_profiled_connection(path, profile=READ_PROFILES["background-read"])


def _connect_rw(path: Path) -> sqlite3.Connection:
    """Open a one-shot writable tier connection, attaching no sibling tier."""
    return open_isolated_write_connection(path, purpose=f"excision apply({path})")


@dataclass(frozen=True, slots=True)
class ExcisionRawTarget:
    """One raw acquisition backing the excised session."""

    raw_id: str
    blob_hash: bytes
    source_path: str


@dataclass(frozen=True, slots=True)
class ContainerMember:
    """One record of a manifest container that belongs to the excised session."""

    source_generation_id: str
    source_item_id: str
    record_coordinate: str
    raw_blob_hash: bytes


@dataclass(frozen=True, slots=True)
class ContainerItem:
    """One ``source_items`` row touched by an excision."""

    source_generation_id: str
    source_item_id: str
    blob_hash: bytes | None

    @property
    def label(self) -> str:
        return f"{self.source_generation_id}:{self.source_item_id}"


@dataclass(frozen=True, slots=True)
class ContainerDisposition:
    """Per-member disposition of the manifest containers holding these raws.

    ``source_items`` is a blob-liveness owner
    (``storage/blob_liveness.py``), so leaving its row in place keeps the
    acquired bytes GC-rooted on disk after an excision reported success
    (polylogue-q4f6d). It cannot simply be deleted either: one container item
    (a ChatGPT ``conversations.json``, an archive export) can cover records of
    many sessions, and deleting it would unroot bytes another live session
    still needs.

    The rule is per member: the excised session's ``source_item_raw_members``
    rows go (with an ``excised_content`` marker for each member's own blob
    hash), and the container row itself goes only when no member with a live
    ``raw_id`` remains. A container kept alive by another session's member is
    named on the plan and receipt as a residual instead of being silently
    left behind --- the excised session's bytes are still inside that
    container blob.
    """

    members: tuple[ContainerMember, ...] = ()
    removable_items: tuple[ContainerItem, ...] = ()
    retained_items: tuple[ContainerItem, ...] = ()


@dataclass(frozen=True, slots=True)
class ExcisionTarget:
    """Rows resolved as in-scope for excising one session.

    Resolved once, up front, so a dry-run preview and the real mutation act
    on the identical row set (mirrors the ``reset --session`` fix, jnj.5).
    """

    session_id: str
    session_exists: bool = False
    raw_targets: tuple[ExcisionRawTarget, ...] = ()
    message_ids: tuple[str, ...] = ()
    block_ids: tuple[str, ...] = ()
    #: Durable hook-event ids this excision removes. Hook payloads are
    #: session-addressable by (origin, session_native_id) but carry no
    #: raw_sessions row, so no raw target above reaches them; they are
    #: resolved here and deleted through ``delete_source_hook_event``, with
    #: every blob hash they own marked excised (polylogue-bhhsa).
    hook_event_ids: tuple[str, ...] = ()
    #: Raw ids of fact-tier evidence (parse_policy='fact') whose declared
    #: identity resolves to this session -- today Claude Code's
    #: ``todos/<session-uuid>[-agent-<uuid>].json`` plan snapshots, which
    #: carry no ``sessions`` row and are linked only by their filename
    #: (polylogue-si5kj). Each also appears in ``raw_targets``; this tuple
    #: exists so the plan and receipt can name them separately.
    fact_raw_ids: tuple[str, ...] = ()
    #: Telemetry spans addressed by ``(origin, session_native_id)``. Like
    #: hook events they carry no ``raw_sessions`` row, so no raw target
    #: reaches them; their attributes/events are session-addressable
    #: evidence and the apply deletes them by name.
    otlp_span_ids: tuple[str, ...] = ()
    #: Per-member disposition of the manifest containers that hold these raw
    #: acquisitions (polylogue-q4f6d).
    containers: ContainerDisposition = field(default_factory=lambda: ContainerDisposition())
    #: Materials whose ``material_observations.referrer_ref`` names this
    #: session. Provider-generated evidence retained through the material
    #: route (today Codex ``codex://state/...`` goals and memories) carries
    #: no ``sessions`` row and no ``raw_sessions`` row of its own, so no raw
    #: target reaches it -- its bytes are owned by ``material_observations``
    #: directly (polylogue-xrba4).
    #:
    #: Resolved per referrer rather than through the state export's raw:
    #: one Codex state database holds every thread in the install, so
    #: excising the raw for one thread would destroy unrelated threads'
    #: evidence. The material is the thread-scoped unit.
    material_ids: tuple[str, ...] = ()
    #: Blob hashes those materials own, read with them so the apply can mark
    #: each one excised before the rows that name it are gone.
    material_blob_hashes: tuple[bytes, ...] = ()
    #: Source-owned marker carriers whose sealed payloads are wholly within
    #: this excision. Their terminal evidence remains content-free in
    #: ``excised_marker_inputs`` after the source bytes are erased.
    marker_input_targets: tuple[MarkerInputExcisionTarget, ...] = ()

    @property
    def found(self) -> bool:
        return self.session_exists or bool(
            self.raw_targets
            or self.message_ids
            or self.block_ids
            or self.hook_event_ids
            or self.otlp_span_ids
            or self.material_ids
            or self.marker_input_targets
        )


def resolve_session_excision_target(archive_root: Path, session_id: str) -> ExcisionTarget:
    """Resolve the exact rows an excision of ``session_id`` would touch."""

    return _resolve_session_excision_target(archive_root, session_id, target_session_ids=frozenset({session_id}))


def _resolve_session_excision_target(
    archive_root: Path, session_id: str, *, target_session_ids: frozenset[str]
) -> ExcisionTarget:
    """Resolve one target against the full preflight cascade set."""

    index_db = archive_root / "index.db"
    source_db = archive_root / "source.db"

    raw_ids: list[str] = []
    session_exists = False
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
                session_exists = True
                raw_ids.append(str(row[0]))
            elif row is not None:
                session_exists = True
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

    # sessions.raw_id names only the most recently applied revision. Every
    # superseded baseline and append fragment is retained live by default
    # (see storage/raw_retention.py), each with its own blob_refs rows and a
    # content hash the head's excised_content marker does not cover -- an
    # append revision is a byte-prefix of its successor but hashes
    # differently. Corroborate the head with the index's own revision
    # bookkeeping before crossing into the durable tier.
    if index_db.exists() and session_exists:
        conn = _connect_ro(index_db)
        try:
            raw_ids.extend(_index_revision_raw_ids(conn, session_id))
        finally:
            conn.close()

    raw_targets: tuple[ExcisionRawTarget, ...] = ()
    hook_event_ids: tuple[str, ...] = ()
    fact_raw_ids: tuple[str, ...] = ()
    otlp_span_ids: tuple[str, ...] = ()
    containers = ContainerDisposition()
    material_ids: tuple[str, ...] = ()
    material_blob_hashes: tuple[bytes, ...] = ()
    marker_input_targets: tuple[MarkerInputExcisionTarget, ...] = ()
    if source_db.exists():
        conn = _connect_ro(source_db)
        try:
            # Fail closed before resolving anything: a session-keyed relation
            # with no declared excision reach must refuse, not be skipped
            # into a success receipt (polylogue-9lrqs).
            audit_session_carriers(conn).raise_if_unreachable()
            # Fact-tier evidence carries no sessions row, so the index can
            # never seed it. Resolve it from its own declared identity and
            # add it to the seed set before the revision closure runs, so
            # every retained revision of that same file is covered too.
            fact_raw_ids = _session_fact_raw_ids(conn, session_id)
            raw_ids.extend(fact_raw_ids)
            if raw_ids:
                resolved = _durable_revision_closure(conn, raw_ids)
                placeholders = ",".join("?" for _ in resolved)
                rows = conn.execute(
                    f"SELECT raw_id, blob_hash, source_path FROM raw_sessions WHERE raw_id IN ({placeholders})",
                    resolved,
                ).fetchall()
                raw_targets = tuple(
                    ExcisionRawTarget(raw_id=str(r[0]), blob_hash=bytes(r[1]), source_path=str(r[2])) for r in rows
                )
                containers = _resolve_container_disposition(conn, tuple(t.raw_id for t in raw_targets))
            if _table_exists(conn, "pending_accepted_marker_inputs") and _table_exists(conn, "accepted_marker_inputs"):
                marker_input_targets = marker_input_excision_targets_sync(
                    conn,
                    target_session_ids=target_session_ids,
                    target_raw_ids=frozenset(t.raw_id for t in raw_targets),
                )
            hook_event_ids = _session_hook_event_ids(conn, session_id)
            otlp_span_ids = _session_otlp_span_ids(conn, session_id)
            material_ids, material_blob_hashes = _session_material_targets(conn, session_id)
        finally:
            conn.close()

    return ExcisionTarget(
        session_id=session_id,
        session_exists=session_exists,
        raw_targets=raw_targets,
        message_ids=message_ids,
        block_ids=block_ids,
        hook_event_ids=hook_event_ids,
        fact_raw_ids=fact_raw_ids,
        otlp_span_ids=otlp_span_ids,
        containers=containers,
        material_ids=material_ids,
        material_blob_hashes=material_blob_hashes,
        marker_input_targets=marker_input_targets,
    )


def _session_material_targets(conn: sqlite3.Connection, session_id: str) -> tuple[tuple[str, ...], tuple[bytes, ...]]:
    """Materials retained under this session id, with the blobs they own.

    ``material_observations.referrer_ref`` is the durable, session-scoped
    coordinate the material route writes, and a material's bytes hang off
    that row's own ``blob_hash`` -- there is no ``blob_refs`` row and no
    ``raw_sessions`` row for a material, so nothing the raw-target closure
    resolves reaches it. Missing table is checked, not caught, so a real
    query failure still surfaces.
    """
    if not _table_exists(conn, "material_observations"):
        return (), ()
    rows = conn.execute(
        "SELECT material_id, blob_hash FROM material_observations WHERE referrer_ref = ? ORDER BY material_id",
        (session_id,),
    ).fetchall()
    material_ids = tuple(str(row[0]) for row in rows)
    blob_hashes = tuple({bytes(row[1]) for row in rows if row[1]})
    return material_ids, blob_hashes


def _index_revision_raw_ids(conn: sqlite3.Connection, session_id: str) -> list[str]:
    """Raw ids the index's revision bookkeeping attributes to this session.

    The index is rebuildable, so this is corroborating evidence only: it
    widens the seed set that :func:`_durable_revision_closure` then expands
    against the durable tier, and never narrows it. A tier that predates
    these relations simply contributes nothing -- missing is checked, not
    caught, so a real query failure still surfaces.
    """
    found: list[str] = []
    if _table_exists(conn, "raw_revision_applications"):
        for row in conn.execute(
            "SELECT raw_id, accepted_raw_id, baseline_raw_id, predecessor_raw_id "
            "FROM raw_revision_applications WHERE session_id = ?",
            (session_id,),
        ).fetchall():
            found.extend(str(value) for value in row if value)
    if _table_exists(conn, "raw_revision_heads"):
        for row in conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE session_id = ?",
            (session_id,),
        ).fetchall():
            if row[0]:
                found.append(str(row[0]))
    return found


def _durable_revision_closure(conn: sqlite3.Connection, seeds: Sequence[str]) -> tuple[str, ...]:
    """Expand seed raw ids to every revision of the same logical source.

    ``sessions.raw_id`` names only the most recently applied revision. Every
    superseded baseline and append fragment stays live until an explicit
    retention run compacts it, each with its own ``blob_refs`` rows and a
    content hash the head's ``excised_content`` marker does not cover -- an
    append revision is a byte-prefix of its successor but hashes
    differently, so excising only the head leaves the earlier bytes both
    readable and re-ingestible.

    Two durable relations, unioned to a fixpoint because
    ``predecessor_raw_id``/``baseline_raw_id`` are plain TEXT with no foreign
    key and therefore no cascade:

    1. ``logical_source_key`` on ``raw_session_memberships`` and
       ``raw_sessions`` -- the grouping of revisions of one logical source
    2. the ``predecessor_raw_id`` / ``baseline_raw_id`` links themselves
    """
    resolved: set[str] = {str(seed) for seed in seeds if seed}
    if not resolved:
        return ()
    has_memberships = _table_exists(conn, "raw_session_memberships")
    for _pass in range(_REVISION_CLOSURE_PASSES):
        frontier = tuple(resolved)
        placeholders = ",".join("?" for _ in frontier)
        keys: set[str] = set()
        if has_memberships:
            keys.update(
                str(row[0])
                for row in conn.execute(
                    f"SELECT DISTINCT logical_source_key FROM raw_session_memberships WHERE raw_id IN ({placeholders})",
                    frontier,
                ).fetchall()
                if row[0]
            )
        keys.update(
            str(row[0])
            for row in conn.execute(
                f"SELECT DISTINCT logical_source_key FROM raw_sessions "
                f"WHERE raw_id IN ({placeholders}) AND logical_source_key IS NOT NULL",
                frontier,
            ).fetchall()
            if row[0]
        )
        grown = set(resolved)
        if keys:
            key_placeholders = ",".join("?" for _ in keys)
            key_values = tuple(keys)
            sources = [f"SELECT raw_id FROM raw_sessions WHERE logical_source_key IN ({key_placeholders})"]
            if has_memberships:
                sources.append(
                    f"SELECT raw_id FROM raw_session_memberships WHERE logical_source_key IN ({key_placeholders})"
                )
            for sql in sources:
                grown.update(str(row[0]) for row in conn.execute(sql, key_values).fetchall() if row[0])
        for row in conn.execute(
            f"SELECT predecessor_raw_id, baseline_raw_id FROM raw_sessions WHERE raw_id IN ({placeholders})",
            frontier,
        ).fetchall():
            grown.update(str(value) for value in row if value)
        if grown == resolved:
            break
        resolved = grown
    return tuple(sorted(resolved))


def _session_hook_event_ids(conn: sqlite3.Connection, session_id: str) -> tuple[str, ...]:
    """Hook events addressed to this session, which excision removes.

    Hook payloads are durable and session-addressable by
    ``(origin, session_native_id)`` but deliberately carry no
    ``raw_sessions``/``sessions`` row, so no raw target reaches them. Resolved
    here so the apply can delete each one through the source tier's own paired
    delete route and mark every blob it owns excised.
    """
    origin, _, native_id = session_id.partition(":")
    if not origin or not native_id or not _table_exists(conn, "raw_hook_events"):
        return ()
    return tuple(
        str(row[0])
        for row in conn.execute(
            "SELECT hook_event_id FROM raw_hook_events WHERE origin = ? AND session_native_id = ?",
            (origin, native_id),
        ).fetchall()
    )


def _session_otlp_span_ids(conn: sqlite3.Connection, session_id: str) -> tuple[str, ...]:
    """Telemetry spans addressed to this session, which excision removes.

    ``otlp_spans`` carries the ``(origin, session_native_id)`` session key and
    no ``raw_sessions`` row, so it is reachable only by name --- the same
    shape as hook events. Its ``attributes_json``/``events_json`` are
    session-addressable evidence, so an excision that left them readable
    would be the next instance of the defect the carrier registry exists to
    prevent.
    """
    origin, _, native_id = session_id.partition(":")
    if not origin or not native_id or not _table_exists(conn, "otlp_spans"):
        return ()
    return tuple(
        str(row[0])
        for row in conn.execute(
            "SELECT span_id FROM otlp_spans WHERE origin = ? AND session_native_id = ?",
            (origin, native_id),
        ).fetchall()
    )


def _resolve_container_disposition(conn: sqlite3.Connection, raw_ids: Sequence[str]) -> ContainerDisposition:
    """Decide, per member, what happens to the containers holding these raws.

    See :class:`ContainerDisposition`. Resolved read-only and up front so the
    dry-run preview and the apply act on the identical row set, and so the
    membership is read *before* the ``ON DELETE SET NULL`` foreign key from
    ``raw_sessions`` erases the link it depends on.
    """
    if not raw_ids or not _table_exists(conn, "source_items"):
        return ContainerDisposition()
    has_members = _table_exists(conn, "source_item_raw_members")
    placeholders = ",".join("?" for _ in raw_ids)
    target_ids = tuple(raw_ids)

    members: list[ContainerMember] = []
    affected: set[tuple[str, str]] = set()
    if has_members:
        for row in conn.execute(
            f"SELECT source_generation_id, source_item_id, record_coordinate, raw_blob_hash "
            f"FROM source_item_raw_members WHERE raw_id IN ({placeholders})",
            target_ids,
        ).fetchall():
            members.append(
                ContainerMember(
                    source_generation_id=str(row[0]),
                    source_item_id=str(row[1]),
                    record_coordinate=str(row[2]),
                    raw_blob_hash=bytes(row[3]),
                )
            )
            affected.add((str(row[0]), str(row[1])))

    item_hashes: dict[tuple[str, str], bytes | None] = {}
    for row in conn.execute(
        f"SELECT source_generation_id, source_item_id, blob_hash FROM source_items WHERE raw_id IN ({placeholders})",
        target_ids,
    ).fetchall():
        key = (str(row[0]), str(row[1]))
        affected.add(key)
        item_hashes[key] = bytes(row[2]) if row[2] is not None else None

    removable: list[ContainerItem] = []
    retained: list[ContainerItem] = []
    for generation_id, item_id in sorted(affected):
        if (generation_id, item_id) not in item_hashes:
            row = conn.execute(
                "SELECT blob_hash FROM source_items WHERE source_generation_id = ? AND source_item_id = ?",
                (generation_id, item_id),
            ).fetchone()
            if row is None:
                continue
            item_hashes[(generation_id, item_id)] = bytes(row[0]) if row[0] is not None else None
        surviving = 0
        if has_members:
            surviving = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM source_item_raw_members "
                    f"WHERE source_generation_id = ? AND source_item_id = ? "
                    f"AND raw_id IS NOT NULL AND raw_id NOT IN ({placeholders})",
                    (generation_id, item_id, *target_ids),
                ).fetchone()[0]
            )
        item = ContainerItem(
            source_generation_id=generation_id,
            source_item_id=item_id,
            blob_hash=item_hashes[(generation_id, item_id)],
        )
        (retained if surviving else removable).append(item)

    return ContainerDisposition(
        members=tuple(members),
        removable_items=tuple(removable),
        retained_items=tuple(retained),
    )


def _session_fact_raw_ids(conn: sqlite3.Connection, session_id: str) -> tuple[str, ...]:
    """Raw ids of fact-tier evidence whose declared identity is this session.

    Fact artifacts (``parse_policy='fact'``) are admitted as their own
    ``raw_sessions`` rows but mint no ``sessions`` row, so
    ``sessions.raw_id`` never names them and no index relation reaches them.
    Their only link to a session is the identity their own declaration
    carries. Today exactly one admitted fact kind carries a session-derived
    identity: Claude Code's ``todos/<session-uuid>[-agent-<uuid>].json`` plan
    snapshots, whose filename is the owning session's native id (a delegated
    subagent's snapshot keeps the parent session's uuid as its stem prefix,
    so it belongs to the same excision). The artifact rule, not a local path
    guess, decides what counts as that kind.

    A fact kind whose identity is *not* session-derived is deliberately not
    matched here: excising a session must not take unrelated evidence with
    it.
    """
    origin, _, native_id = session_id.partition(":")
    if not origin or not native_id or origin != Origin.CLAUDE_CODE_SESSION.value:
        return ()
    rows = conn.execute(
        "SELECT raw_id, source_path FROM raw_sessions WHERE origin = ? AND source_path LIKE '%todos/%.json'",
        (origin,),
    ).fetchall()
    resolved: list[str] = []
    for raw_id, source_path in rows:
        rule = artifact_rule_for_path(Provider.CLAUDE_CODE, str(source_path))
        if rule is None or rule.kind != "todo_snapshot":
            continue
        owner_session_id, _agent_id = session_and_agent_id_from_filename(str(source_path))
        if owner_session_id == native_id:
            resolved.append(str(raw_id))
    return tuple(sorted(resolved))


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
    #: Durable hook-event rows for this session that an apply will remove
    #: (polylogue-bhhsa).
    source_hook_events: int = 0
    #: Fact-tier raw rows resolved by declared identity rather than by any
    #: index relation -- Claude Code TODO plan snapshots (polylogue-si5kj).
    #: Already counted in ``source_raw_rows``; named so the preview shows
    #: that this evidence class is in scope.
    source_fact_rows: int = 0
    #: Telemetry spans addressed to this session that an apply will remove.
    source_otlp_spans: int = 0
    #: Container member rows an apply will remove (polylogue-q4f6d).
    source_container_members: int = 0
    #: Container items an apply will remove because no live member remains.
    source_container_items: int = 0
    #: Containers that survive because another session's member is still
    #: live: the excised bytes stay inside that container blob.
    retained_source_containers: tuple[str, ...] = ()
    #: ``material_observations`` rows retained under this session id that an
    #: apply will remove, marking every blob they own excised
    #: (polylogue-xrba4).
    source_materials: int = 0
    #: Sealed source marker carriers whose payloads the apply will erase.
    source_marker_inputs_pending: int = 0
    source_marker_inputs_accepted: int = 0

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
            "source_hook_events": self.source_hook_events,
            "source_fact_rows": self.source_fact_rows,
            "source_otlp_spans": self.source_otlp_spans,
            "source_container_members": self.source_container_members,
            "source_container_items": self.source_container_items,
            "retained_source_containers": list(self.retained_source_containers),
            "source_materials": self.source_materials,
            "source_marker_inputs_pending": self.source_marker_inputs_pending,
            "source_marker_inputs_accepted": self.source_marker_inputs_accepted,
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
    if source_db.exists() and (target.raw_targets or target.material_ids):
        conn = _connect_ro(source_db)
        try:
            for material_hash in target.material_blob_hashes:
                if is_blob_hash_excised(conn, material_hash):
                    already_excised.append(material_hash.hex())
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
            # message_embeddings/message_embeddings_meta are content-addressed
            # (keyed by vector_derivation_hash, polylogue-q88p) and may be
            # shared with messages outside this excision target; the
            # message-scoped count is the per-message ref count, not a raw
            # vector-table count (a shared vector must not be reported as
            # "will be removed" when another message still needs it).
            row = conn.execute(
                f"SELECT COUNT(*) FROM message_embedding_refs WHERE message_id IN ({placeholders})",
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
        source_hook_events=len(target.hook_event_ids),
        source_fact_rows=len(target.fact_raw_ids),
        source_otlp_spans=len(target.otlp_span_ids),
        source_container_members=len(target.containers.members),
        source_container_items=len(target.containers.removable_items),
        retained_source_containers=tuple(item.label for item in target.containers.retained_items),
        source_materials=len(target.material_ids),
        source_marker_inputs_pending=sum(target.state == "pending" for target in target.marker_input_targets),
        source_marker_inputs_accepted=sum(target.state == "accepted" for target in target.marker_input_targets),
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
    #: Hook-event rows addressed to the excised session that were still
    #: readable in ``source.db`` after the apply committed -- a verified
    #: post-condition, not an assumption. A privacy operation must not
    #: report unqualified success while session-addressable payloads -- tool
    #: inputs and outputs, file contents, anything pasted -- survive it
    #: (polylogue-bhhsa).
    retained_hook_events: tuple[str, ...] = ()
    #: Manifest containers that still hold this session's acquired bytes
    #: because another session's member of the same container is still live
    #: (polylogue-q4f6d). The container blob cannot be unrooted without
    #: destroying evidence that is legitimately in use, so this is a named
    #: residual rather than a silent one --- but the bytes are still on disk,
    #: so it is not a complete excision either.
    retained_source_containers: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether every resolved family was actually removed."""
        return not (self.retained_hook_events or self.retained_source_containers)

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
            "retained_hook_events": list(self.retained_hook_events),
            "retained_source_containers": list(self.retained_source_containers),
            "complete": self.complete,
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
    resolved_target: ExcisionTarget | None = None,
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
    target = (
        resolved_target if resolved_target is not None else resolve_session_excision_target(archive_root, session_id)
    )
    if not target.found:
        return ExcisionReceipt(session_id=session_id, found=False)

    counts: dict[str, int] = {
        "embeddings_vectors": 0,
        "embeddings_vectors_gc": 0,
        # The receipt is committed before index cleanup, so record the
        # expected session deletion in it rather than waiting for rowcount.
        "index_sessions": int(target.session_exists),
        "index_messages": len(target.message_ids),
        "index_blocks": len(target.block_ids),
        "source_blob_refs": 0,
        "source_raw_rows": 0,
        "source_fact_rows": len(target.fact_raw_ids),
        "source_hook_events": 0,
        "source_otlp_spans": 0,
        "source_container_members": 0,
        "source_container_items": 0,
        "source_publication_reservations": 0,
        "source_materials": 0,
        "source_marker_inputs_pending": 0,
        "source_marker_inputs_accepted": 0,
        "index_marker_witnesses": 0,
        "user_assertions_removed": 0,
    }

    embeddings_db = archive_root / "embeddings.db"
    if embeddings_db.exists() and target.message_ids:
        conn = _connect_rw(embeddings_db)
        try:
            try_load_sqlite_vec(conn)
            with conn:
                placeholders = ",".join("?" for _ in target.message_ids)
                # message_embeddings/message_embeddings_meta are content-
                # addressed and may be shared with messages outside this
                # excision target (polylogue-q88p) -- deleting a vector by
                # message_id is no longer meaningful (there is no message_id
                # column on those tables) and would be unsafe even if it
                # were, since another live message could still reference the
                # same hash. Delete this excision's refs first, then remove
                # the underlying vector/meta rows only for hashes that no
                # longer have ANY remaining ref -- reference-counted, scoped
                # strictly to the hashes this excision actually touched (not
                # a general background GC sweep).
                affected_hashes = tuple(
                    bytes(row[0])
                    for row in conn.execute(
                        f"SELECT DISTINCT vector_derivation_hash FROM message_embedding_refs "
                        f"WHERE message_id IN ({placeholders})",
                        target.message_ids,
                    ).fetchall()
                )
                cursor = conn.execute(
                    f"DELETE FROM message_embedding_refs WHERE message_id IN ({placeholders})",
                    target.message_ids,
                )
                counts["embeddings_vectors"] = max(cursor.rowcount, 0)
                removed_vector_hashes = 0
                for input_hash in affected_hashes:
                    still_referenced = conn.execute(
                        "SELECT 1 FROM message_embedding_refs WHERE vector_derivation_hash = ? LIMIT 1",
                        (input_hash,),
                    ).fetchone()
                    if still_referenced is not None:
                        continue
                    conn.execute(
                        "DELETE FROM message_embeddings WHERE vector_derivation_hash = ?",
                        (input_hash.hex(),),
                    )
                    removed_vector_hashes += max(
                        conn.execute(
                            "DELETE FROM message_embeddings_meta WHERE vector_derivation_hash = ?",
                            (input_hash,),
                        ).rowcount,
                        0,
                    )
                counts["embeddings_vectors_gc"] = removed_vector_hashes
                conn.execute("DELETE FROM embedding_status WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM embedding_failures WHERE session_id = ?", (session_id,))
        finally:
            conn.close()

    source_db = archive_root / "source.db"
    removed_hashes: list[str] = []
    retained_hook_events: tuple[str, ...] = ()
    retained_source_containers = tuple(item.label for item in target.containers.retained_items)
    if source_db.exists() and (
        target.raw_targets
        or target.hook_event_ids
        or target.otlp_span_ids
        or target.material_ids
        or target.marker_input_targets
    ):
        conn = _connect_rw(source_db)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                marker_counts = excise_marker_input_targets_sync(
                    conn, target.marker_input_targets, excised_at_ms=timestamp
                )
                counts["source_marker_inputs_pending"] = marker_counts["pending"]
                counts["source_marker_inputs_accepted"] = marker_counts["accepted"]
                # Containers first: deleting raw_sessions fires the
                # ON DELETE SET NULL foreign key that erases the raw_id this
                # disposition is keyed on, and source_items is a blob-liveness
                # owner, so a container row left behind keeps the excised
                # bytes rooted against GC (polylogue-q4f6d).
                for member in target.containers.members:
                    conn.execute(
                        "DELETE FROM source_item_raw_members WHERE source_generation_id = ? "
                        "AND source_item_id = ? AND record_coordinate = ?",
                        (member.source_generation_id, member.source_item_id, member.record_coordinate),
                    )
                    counts["source_container_members"] += 1
                    record_excised_blob_hash(
                        conn,
                        blob_hash=member.raw_blob_hash,
                        reason=reason,
                        actor=actor,
                        prior_revision=f"{member.source_item_id}:{member.record_coordinate}",
                        span=None,
                        excised_at_ms=timestamp,
                    )
                    removed_hashes.append(member.raw_blob_hash.hex())
                for item in target.containers.removable_items:
                    conn.execute(
                        "DELETE FROM source_items WHERE source_generation_id = ? AND source_item_id = ?",
                        (item.source_generation_id, item.source_item_id),
                    )
                    counts["source_container_items"] += 1
                    if item.blob_hash is not None:
                        record_excised_blob_hash(
                            conn,
                            blob_hash=item.blob_hash,
                            reason=reason,
                            actor=actor,
                            prior_revision=item.label,
                            span=None,
                            excised_at_ms=timestamp,
                        )
                        removed_hashes.append(item.blob_hash.hex())

                for span_id in target.otlp_span_ids:
                    cursor = conn.execute("DELETE FROM otlp_spans WHERE span_id = ?", (span_id,))
                    counts["source_otlp_spans"] += max(cursor.rowcount, 0)

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

                for hook_event_id in target.hook_event_ids:
                    # A hook event owns its payload bytes through three
                    # durable coordinates: its own blob_hash column, its
                    # carrier rows, and the 'hook_payload' blob ref keyed by
                    # the event id. Read all three BEFORE deleting -- a row
                    # written before the v22 blob_hash backfill has a NULL
                    # there and would otherwise leave an unmarked,
                    # re-ingestible payload behind.
                    owned_hashes: set[bytes] = set()
                    row = conn.execute(
                        "SELECT blob_hash FROM raw_hook_events WHERE hook_event_id = ?",
                        (hook_event_id,),
                    ).fetchone()
                    if row is None:
                        continue
                    if row[0]:
                        owned_hashes.add(bytes(row[0]))
                    owned_hashes.update(
                        bytes(r[0])
                        for r in conn.execute(
                            "SELECT DISTINCT blob_hash FROM hook_event_carriers WHERE hook_event_id = ?",
                            (hook_event_id,),
                        ).fetchall()
                        if r[0]
                    )
                    owned_hashes.update(
                        bytes(r[0])
                        for r in conn.execute(
                            "SELECT DISTINCT blob_hash FROM blob_refs WHERE ref_type = 'hook_payload' AND ref_id = ?",
                            (hook_event_id,),
                        ).fetchall()
                        if r[0]
                    )
                    # The source tier's own paired delete route: event row,
                    # carriers and the owned blob ref together, so no
                    # durable row is left pinning the blob.
                    delete_source_hook_event(conn, hook_event_id, manage_transaction=False)
                    counts["source_hook_events"] += 1
                    for blob_hash in owned_hashes:
                        record_excised_blob_hash(
                            conn,
                            blob_hash=blob_hash,
                            reason=reason,
                            actor=actor,
                            prior_revision=hook_event_id,
                            span=None,
                            excised_at_ms=timestamp,
                        )
                        removed_hashes.append(blob_hash.hex())

                # Materials retained under this session id own their bytes
                # through material_observations.blob_hash alone. Mark every
                # hash excised BEFORE deleting the rows that name it, or the
                # bytes stay both readable in the blob store and
                # re-admissible under the same content hash. The evidence
                # links cascade with the row (foreign_keys is ON above).
                for material_blob_hash in target.material_blob_hashes:
                    record_excised_blob_hash(
                        conn,
                        blob_hash=material_blob_hash,
                        reason=reason,
                        actor=actor,
                        prior_revision=None,
                        span=None,
                        excised_at_ms=timestamp,
                    )
                    removed_hashes.append(material_blob_hash.hex())
                if target.material_ids:
                    # supersedes_material_id is a plain (non-deferred) FK
                    # between materials, and a superseded revision of the
                    # same material shares this referrer, so deleting the
                    # older row first would abort the whole apply. Drop the
                    # chain first; the rows it points at are all going.
                    placeholders = ",".join("?" for _ in target.material_ids)
                    conn.execute(
                        f"UPDATE material_observations SET supersedes_material_id = NULL "
                        f"WHERE supersedes_material_id IN ({placeholders})",
                        target.material_ids,
                    )
                for material_id in target.material_ids:
                    cursor = conn.execute(
                        "DELETE FROM material_observations WHERE material_id = ?",
                        (material_id,),
                    )
                    counts["source_materials"] += max(cursor.rowcount, 0)
                # A publication reservation is a durable claim on a blob by
                # hash. Left standing it keeps an excised blob reserved --- and
                # named --- after the evidence it published is gone
                # (polylogue-aix14). Nothing may publish an excised hash, so the
                # reservation is dropped with the bytes it reserved.
                if removed_hashes and _table_exists(conn, "blob_publication_reservations"):
                    for blob_hash_hex in dict.fromkeys(removed_hashes):
                        cursor = conn.execute(
                            "DELETE FROM blob_publication_reservations WHERE blob_hash = ?",
                            (bytes.fromhex(blob_hash_hex),),
                        )
                        counts["source_publication_reservations"] += max(cursor.rowcount, 0)

            # Verified post-condition, after the transaction committed: any
            # hook event still readable is a residual this operation must
            # name rather than let the per-tier counts read as the whole job.
            retained_hook_events = tuple(
                hook_event_id
                for hook_event_id in target.hook_event_ids
                if conn.execute(
                    "SELECT 1 FROM raw_hook_events WHERE hook_event_id = ?",
                    (hook_event_id,),
                ).fetchone()
                is not None
            )
        finally:
            conn.close()

    # Durable source authority must commit before the rebuildable index loses
    # the key needed to retry an interrupted excision. The receipt is written
    # before index cleanup so a crash after the receipt remains attributable.
    user_db = archive_root / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    conn = _connect_rw(user_db)
    existing_receipt: tuple[str, dict[str, object]] | None = None
    receipt_id = _receipt_assertion_id(session_id, timestamp)
    try:
        with conn:
            row = conn.execute(
                "SELECT assertion_id, value_json FROM assertions "
                "WHERE target_ref = ? AND kind = ? ORDER BY created_at_ms LIMIT 1",
                (f"session:{session_id}", AssertionKind.EXCISION_RECORD.value),
            ).fetchone()
            if row is not None and row[1]:
                value = json.loads(str(row[1]))
                if isinstance(value, dict):
                    existing_receipt = (str(row[0]), value)
            if existing_receipt is None:
                refs = _target_refs(target)
                removed_assertions = 0
                tombstoned_assertions = 0
                for ref in refs:
                    # Marker assertions are replay products keyed by a stable
                    # ``marker-*`` id.  Keep a content-free terminal tombstone
                    # for them: otherwise the next marker convergence pass
                    # sees the deterministic id as absent and lowers the
                    # excised content again.  All other assertions retain the
                    # existing hard-delete policy.
                    marker_cursor = conn.execute(
                        """
                        UPDATE assertions
                        SET target_ref = ?, value_json = '{}', body_text = NULL,
                            evidence_refs_json = '[]', status = ?, updated_at_ms = ?
                        WHERE target_ref = ? AND assertion_id LIKE 'marker-%'
                        """,
                        (
                            f"excision-marker:{session_id}",
                            AssertionStatus.DELETED.value,
                            timestamp,
                            ref,
                        ),
                    )
                    tombstoned_assertions += max(marker_cursor.rowcount, 0)
                    cursor = conn.execute("DELETE FROM assertions WHERE target_ref = ?", (ref,))
                    removed_assertions += max(cursor.rowcount, 0)
                counts["user_assertions_removed"] = removed_assertions
                counts["user_assertions_tombstoned"] = tombstoned_assertions

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

    index_db = archive_root / "index.db"
    if index_db.exists():
        conn = _connect_rw(index_db)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                if target.marker_input_targets and _table_exists(conn, "ingest_marker_witnesses"):
                    placeholders = ",".join("?" for _ in target.marker_input_targets)
                    cursor = conn.execute(
                        f"DELETE FROM ingest_marker_witnesses WHERE request_key IN ({placeholders})",
                        tuple(marker.identity for marker in target.marker_input_targets),
                    )
                    if existing_receipt is None:
                        counts["index_marker_witnesses"] = max(cursor.rowcount, 0)
                cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                if existing_receipt is None:
                    counts["index_sessions"] = max(cursor.rowcount, 0)
        finally:
            conn.close()

    if existing_receipt is not None:
        receipt_id, value = existing_receipt
        stored_counts = value.get("counts")
        stored_timestamp = value.get("excised_at_ms")
        stored_hashes = value.get("removed_blob_hashes")
        return ExcisionReceipt(
            session_id=session_id,
            found=True,
            reason=str(value.get("reason", reason)),
            actor=str(value.get("actor", actor)),
            excised_at_ms=int(stored_timestamp) if isinstance(stored_timestamp, int) else timestamp,
            receipt_assertion_id=receipt_id,
            removed_blob_hashes=(
                tuple(str(item) for item in stored_hashes) if isinstance(stored_hashes, (list, tuple)) else ()
            ),
            counts=dict(stored_counts) if isinstance(stored_counts, dict) else {},
            retained_hook_events=retained_hook_events,
            retained_source_containers=retained_source_containers,
        )

    return ExcisionReceipt(
        session_id=session_id,
        found=True,
        reason=reason,
        actor=actor,
        excised_at_ms=timestamp,
        receipt_assertion_id=receipt_id,
        removed_blob_hashes=tuple(removed_hashes),
        counts=counts,
        retained_hook_events=retained_hook_events,
        retained_source_containers=retained_source_containers,
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

    dependent_ids = find_lineage_dependents(archive_root, session_id)
    if dependent_ids and not cascade_lineage:
        raise LineageDependentsError(session_id=session_id, dependent_session_ids=dependent_ids)

    # Resolve every cascade member before touching any tier.  A sealed marker
    # carrier may mention a later dependent; discovering that it mixes a
    # retained session after an earlier member was already deleted would make
    # a refusal mutate the archive.  The complete cascade set makes a shared
    # parent/child carrier wholly targeted while still refusing any outsider.
    session_ids = (*dependent_ids, session_id)
    target_session_ids = frozenset(session_ids)
    targets = tuple(
        _resolve_session_excision_target(archive_root, candidate, target_session_ids=target_session_ids)
        for candidate in session_ids
    )
    target = targets[-1]
    if not target.found:
        return ExcisionReceipt(session_id=session_id, found=False)

    timestamp = now_ms if now_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
    cascaded_receipts = tuple(
        _apply_single_session_excision(
            archive_root,
            dependent_id,
            reason=reason,
            actor=actor,
            now_ms=timestamp,
            resolved_target=resolved_target,
        )
        for dependent_id, resolved_target in zip(dependent_ids, targets[:-1], strict=True)
    )
    primary = _apply_single_session_excision(
        archive_root, session_id, reason=reason, actor=actor, now_ms=timestamp, resolved_target=target
    )

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
        retained_hook_events=tuple(
            dict.fromkeys(
                [*primary.retained_hook_events, *(e for r in cascaded_receipts for e in r.retained_hook_events)]
            )
        ),
        retained_source_containers=tuple(
            dict.fromkeys(
                [
                    *primary.retained_source_containers,
                    *(label for r in cascaded_receipts for label in r.retained_source_containers),
                ]
            )
        ),
    )


__all__ = [
    "ExcisionPolicyError",
    "ExcisionPolicySnapshot",
    "ExcisionPlan",
    "ExcisionRawTarget",
    "ExcisionReceipt",
    "ExcisionTarget",
    "ContainerDisposition",
    "ContainerItem",
    "ContainerMember",
    "LineageDependentsError",
    "UnclassifiedSessionCarrierError",
    "apply_session_excision",
    "build_excision_policy_snapshot",
    "find_lineage_dependents",
    "plan_session_excision",
    "resolve_session_excision_target",
]
