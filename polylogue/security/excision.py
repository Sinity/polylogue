"""Standalone/off-mode local excision (polylogue-27m).

"The archive can forget on purpose": in standalone/off mode (no Sinex-backed
replica to reconcile), local excision is *authoritative*. Applying an
excision:

1. Deletes the session's vectors from ``embeddings.db`` (if embedded).
2. Deletes the session's ``blob_refs`` and ``raw_sessions`` rows from
   ``source.db`` (cascading to ``raw_session_memberships``/
   ``raw_membership_census``), then records a durable removed-hash marker in
   ``excised_content`` for every distinct blob hash grouped under that raw
   ingestion's ``ref_id`` -- not just the raw payload's own hash. ``blob_refs``
   names raw-owned ``ref_type IN ('raw_payload', 'attachment')`` records,
   so a session's inline attachments (whose content hash can
   differ from the raw payload's) get their own non-resurrection marker too.
   A hash that a session outside the excision still references is the
   exception (see "Shared blobs" below).
   That marker is what makes re-ingest non-resurrecting: both acquire-time
   raw-session write functions (``write_source_raw_session`` and
   ``write_source_raw_session_blob_ref`` -- the payload-in-memory and
   blob-ref/streaming routes respectively, shared by the CLI import path and
   the daemon watch path) refuse to re-store a payload whose blob hash is
   recorded there, even after an unrelated ``index.db`` rebuild.
3. Deletes the session's durable hook events (``raw_hook_events`` +
   ``hook_event_carriers`` + their ``hook_payload`` blob refs, through
   ``delete_source_hook_event``) and records an ``excised_content`` marker
   for every blob hash they owned that no other session references. Hook
   payloads are session-addressable by
   ``(origin, session_native_id)`` and carry no ``raw_sessions`` row, so no
   raw target above reaches them; without this step a completed excision
   left every PreToolUse/PostToolUse payload readable (polylogue-bhhsa).
   Every hook deletion is a declared Source effect; an undeclared write
   that would keep a payload readable (a reinstating trigger, say) is refused
   by the writer's authorizer and the whole excision fails closed.
4. Disposes the manifest containers holding those raw acquisitions per
   member (``source_item_raw_members`` + ``source_items``), and drops any
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

**Tool-output sidecars.** A Claude Code or gemini-cli tool result that
overflowed its inline envelope left the full output in a sidecar file
(``<session>/tool-results/`` or ``tool-outputs/session-<id>/``), retained as
its own ``tool_result_sidecar`` raw row that no session relation names.
``resolve_session_excision_target`` derives each transcript's sidecar scope
from its retained ``source_path`` and seeds every retained revision of the
files the session owns into the revision closure (polylogue-8j9rh).
Ownership follows the join's own rules: a file whose stem is a ``tool_id`` of
one of the session's ``tool_result`` blocks, or that a sidecar event of the
session names as its own (see :class:`_SidecarOwnership`). The scope
directory is shared (a Claude Code parent and its subagents; every chat of
one gemini-cli process), so a file another transcript owns stays, and a file
no transcript claimed stays too.

**Shared blobs.** Excision forgets the excised session, not every session
whose content shares a content-addressed blob with it. Two sessions with the
same tool output or the same attachment own one blob hash. After the excised
session's source rows are deleted, each hash they named is marked in
``excised_content`` only when
:func:`polylogue.storage.blob_liveness.inspect_session_blob_references`
finds no other session's reference to it (a retained raw, a live ledger row,
a hook event, a material, a retained container, or an index attachment
linked to another session). A shared hash stays unmarked and readable for
that session and is named on the receipt as ``shared_blob_hashes``;
excising the last session that references it marks it. Forgetting content
wherever it appears is the secret-scanning route's job, not this one.

**Attachments referenced from elsewhere.** ``attachment_refs.session_id``/
``message_id`` carry ``ON DELETE CASCADE`` to ``sessions``/``messages``, so
deleting the excised session's row already removes only *its own*
attachment references. A content-hash-deduplicated ``attachments`` row that
is still referenced by another, non-excised session's ``attachment_refs`` is
untouched -- excision never deletes shared attachment metadata still in
legitimate use elsewhere; it only unlinks the excised session's reference to
it, and by the rule above its blob hash is not marked.

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
the audited Excision operation refuses this by default
(:class:`LineageDependentsError`) and only proceeds with
``cascade_lineage=True``, which excises the whole transitive lineage
together so no dependent composed read is left broken.

Mirror/primary-mode lifecycle mechanics (durable request/outbox, fault
injection against a versioned contract fake) live in
:mod:`polylogue.security.lifecycle`. This module is the off/standalone path
only -- see ``docs/security.md`` for the full mode matrix and non-goals.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from builtins import BaseExceptionGroup
from collections.abc import Collection, Generator, Iterator, Mapping, Sequence, Set
from contextlib import AbstractContextManager, ExitStack, closing
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from pydantic import ConfigDict, TypeAdapter

from polylogue.archive.revision_authority import WORK_EVENT_RAW_ID_PREFIX
from polylogue.core.compute_cancel import compute_cancel_requested
from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility, BlockType, Origin, Provider
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.pipeline.ids import SIDECAR_BLOB_EVENT_TYPES
from polylogue.security.excision_carriers import (
    UnclassifiedSessionCarrierError,
    audit_session_carriers,
)
from polylogue.security.excision_policy import (
    ExcisionPolicyError,
    ExcisionPolicySnapshot,
    build_excision_policy_snapshot,
)
from polylogue.sources.live.gemini_tool_output_sidecars import resolve_tool_outputs_dir
from polylogue.sources.live.tool_result_sidecars import resolve_tool_results_dir
from polylogue.sources.origin_specs import artifact_rule_for_path
from polylogue.sources.parsers.claude.todos import session_and_agent_id_from_filename
from polylogue.storage.accepted_marker_inputs import (
    MarkerInputExcisionTarget,
    marker_input_excision_targets_sync,
)
from polylogue.storage.blob_gc_index_watermark import index_liveness_authority_blocker
from polylogue.storage.blob_liveness import (
    ConnectionSessionBlobLivenessRead,
    LivenessState,
    inspect_session_blob_references,
)
from polylogue.storage.sqlite.archive_tiers.source_write import (
    is_blob_hash_excised,
)
from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection

if TYPE_CHECKING:
    from polylogue.operations.mutation_actuators import SessionExcisionArgs
    from polylogue.operations.mutation_transaction import MutationPlan, RecoveryOperation, StartedBoundMutation
    from polylogue.storage.blob_liveness import BlobOwner
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.reference_seal import KnownTierCell, PreparedIndexMutation


# Excision opens and commits one tier at a time so a mid-apply failure leaves
# at most one tier mutated, never a half-written cross-tier transaction. The
# writer factory below is the declared no-sibling-attach route for that apply.


def _connect_rw(path: Path, *, archive_root: Path, foreign_keys: bool = False) -> sqlite3.Connection:
    """Open a one-shot writable tier connection, attaching no sibling tier."""
    from polylogue.storage.sqlite.connection_profile import ISOLATED_TIER_WRITE_PROFILE

    return open_isolated_write_connection(
        path,
        purpose=f"excision apply({path})",
        archive_root=archive_root,
        profile=replace(ISOLATED_TIER_WRITE_PROFILE, foreign_keys=foreign_keys),
    )


@dataclass(frozen=True, slots=True)
class ExcisionRawTarget:
    """One raw acquisition backing the excised session."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(
        strict=True, extra="forbid", ser_json_bytes="hex", val_json_bytes="hex"
    )

    raw_id: str
    blob_hash: bytes
    source_path: str


@dataclass(frozen=True, slots=True)
class ContainerMember:
    """One record of a manifest container that belongs to the excised session."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(
        strict=True, extra="forbid", ser_json_bytes="hex", val_json_bytes="hex"
    )

    source_generation_id: str
    source_item_id: str
    record_coordinate: str
    raw_blob_hash: bytes


@dataclass(frozen=True, slots=True)
class ContainerItem:
    """One ``source_items`` row touched by an excision."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(
        strict=True, extra="forbid", ser_json_bytes="hex", val_json_bytes="hex"
    )

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

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(
        strict=True, extra="forbid", ser_json_bytes="hex", val_json_bytes="hex"
    )

    members: tuple[ContainerMember, ...] = ()
    removable_items: tuple[ContainerItem, ...] = ()
    retained_items: tuple[ContainerItem, ...] = ()


@dataclass(frozen=True, slots=True)
class IndexMarkerExcisionTarget:
    """Original derived witness coordinates and an exact dispositions digest."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(
        strict=True, extra="forbid", ser_json_bytes="hex", val_json_bytes="hex"
    )

    request_key: str
    carrier_digest: str
    incarnation_id: str
    dispositions_sha256: bytes


@dataclass(frozen=True, slots=True)
class ExcisionTarget:
    """Rows resolved as in-scope for excising one session.

    Resolved once, up front, so a dry-run preview and the real mutation act
    on the identical row set (mirrors the ``reset --session`` fix, jnj.5).
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(
        strict=True, extra="forbid", ser_json_bytes="hex", val_json_bytes="hex"
    )

    session_id: str
    index_marker_witnesses: tuple[IndexMarkerExcisionTarget, ...] = field(kw_only=True)
    session_exists: bool = False
    session_content_hash: bytes | None = None
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
    #: Raw ids of the tool-output sidecars this session owns -- the Claude
    #: Code ``tool-results/`` and gemini-cli ``tool-outputs/session-<id>/``
    #: files its tool results overflowed into, retained as their own
    #: ``tool_result_sidecar`` raw rows that no session relation names
    #: (polylogue-8j9rh). Each also appears in ``raw_targets``; named
    #: separately so the plan and receipt can count them.
    sidecar_raw_ids: tuple[str, ...] = ()
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
            or self.material_ids
            or self.marker_input_targets
        )


_EXCISION_TARGET_REPLAY = TypeAdapter(ExcisionTarget)


def excision_target_replay(target: ExcisionTarget) -> dict[str, object]:
    """Retain the canonical target coordinates, never erased content bodies."""
    return cast(dict[str, object], json.loads(_EXCISION_TARGET_REPLAY.dump_json(target)))


def excision_target_from_replay(value: object) -> ExcisionTarget:
    """Decode exact frozen coordinates without resolving a newer generation."""
    target = _EXCISION_TARGET_REPLAY.validate_json(json.dumps(value, allow_nan=False), strict=True)
    if excision_target_replay(target) != value:
        raise ValueError("excision replay requires the complete canonical target coordinates")
    if target.session_content_hash is not None and len(target.session_content_hash) != 32:
        raise ValueError("excision replay requires the original SHA-256 session identity")
    if any(len(raw.blob_hash) != 32 for raw in target.raw_targets) or any(
        len(value) != 32 for value in target.material_blob_hashes
    ):
        raise ValueError("excision replay requires complete original blob identities")
    witness_keys: set[str] = set()
    markers = {marker.identity: marker.carrier_digest for marker in target.marker_input_targets}
    for witness in target.index_marker_witnesses:
        if (
            witness.request_key in witness_keys
            or markers.get(witness.request_key) != witness.carrier_digest
            or len(witness.dispositions_sha256) != 32
            or len(witness.request_key) != 64
            or len(witness.carrier_digest) != 64
            or any(char not in "0123456789abcdef" for char in witness.request_key + witness.carrier_digest)
            or len(witness.incarnation_id) != 36
        ):
            raise ValueError("excision replay requires exact unique original Index marker coordinates")
        witness_keys.add(witness.request_key)
    return target


def resolve_session_excision_target(archive: ArchiveStore, session_id: str) -> ExcisionTarget:
    """Resolve the exact rows an excision of ``session_id`` would touch."""

    return _resolve_session_excision_target(archive, session_id, target_session_ids=frozenset({session_id}))


def _resolve_session_excision_target(
    archive: ArchiveStore, session_id: str, *, target_session_ids: frozenset[str]
) -> ExcisionTarget:
    """Resolve one target against the full preflight cascade set."""

    index_db = archive.index_db_path
    source_db = archive.source_db_path
    index = archive._conn
    source = archive.source_connection

    raw_ids: list[str] = []
    session_exists = False
    session_content_hash: bytes | None = None
    message_ids: tuple[str, ...] = ()
    block_ids: tuple[str, ...] = ()
    sidecar_ownership = _SidecarOwnership()

    if index_db.exists():
        row = index.execute(
            "SELECT raw_id, content_hash FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is not None:
            session_content_hash = bytes(row[1]) if row[1] is not None else None
        if row is not None and row[0]:
            session_exists = True
            raw_ids.append(str(row[0]))
        elif row is not None:
            session_exists = True
        message_ids = tuple(
            str(r[0])
            for r in index.execute(
                "SELECT message_id FROM messages WHERE session_id = ? ORDER BY message_id",
                (session_id,),
            ).fetchall()
        )
        block_ids = tuple(
            str(r[0])
            for r in index.execute(
                "SELECT block_id FROM blocks WHERE session_id = ? ORDER BY block_id",
                (session_id,),
            ).fetchall()
        )
        sidecar_ownership = _session_sidecar_ownership(index, session_id)

    # sessions.raw_id names only the most recently applied revision. Every
    # superseded baseline and append fragment is retained live by default
    # (see storage/raw_retention.py), each with its own blob_refs rows and a
    # content hash the head's excised_content marker does not cover -- an
    # append revision is a byte-prefix of its successor but hashes
    # differently. Corroborate the head with the index's own revision
    # bookkeeping before crossing into the durable tier.
    if index_db.exists() and session_exists:
        raw_ids.extend(_index_revision_raw_ids(index, session_id))

    raw_targets: tuple[ExcisionRawTarget, ...] = ()
    hook_event_ids: tuple[str, ...] = ()
    fact_raw_ids: tuple[str, ...] = ()
    sidecar_raw_ids: tuple[str, ...] = ()
    containers = ContainerDisposition()
    material_ids: tuple[str, ...] = ()
    material_blob_hashes: tuple[bytes, ...] = ()
    marker_input_targets: tuple[MarkerInputExcisionTarget, ...] = ()
    if source_db.exists():
        # Fail closed before resolving anything: a session-keyed relation
        # with no declared excision reach must refuse, not be skipped into a
        # success receipt (polylogue-9lrqs).
        audit_session_carriers(source).raise_if_unreachable()
        # Fact-tier evidence carries no sessions row, so the index can never
        # seed it. Resolve it from its own declared identity and add it to the
        # seed set before the revision closure runs.
        fact_raw_ids = _session_fact_raw_ids(source, session_id)
        raw_ids.extend(fact_raw_ids)
        raw_ids.extend(_session_work_event_raw_ids(source, session_id))
        if sidecar_ownership and raw_ids:
            sidecar_raw_ids = _session_sidecar_raw_ids(
                source, session_id, _durable_revision_closure(source, raw_ids), sidecar_ownership
            )
            raw_ids.extend(sidecar_raw_ids)
        marker_target_raw_ids = frozenset(raw_ids)
        resolved = _durable_revision_closure(source, raw_ids) if raw_ids else ()
        marker_target_raw_ids = frozenset(resolved)
        has_marker_inputs = _table_exists(source, "pending_accepted_marker_inputs") and _table_exists(
            source, "accepted_marker_inputs"
        )
        if has_marker_inputs:
            marker_input_targets = marker_input_excision_targets_sync(
                source,
                target_session_ids=target_session_ids,
                target_raw_ids=marker_target_raw_ids,
            )
            raw_ids.extend(marker.raw_id for marker in marker_input_targets)
            resolved = _durable_revision_closure(source, raw_ids) if raw_ids else ()
            marker_target_raw_ids = frozenset(resolved)
        if raw_ids:
            placeholders = ",".join("?" for _ in resolved)
            rows = source.execute(
                f"SELECT raw_id, blob_hash, source_path FROM raw_sessions WHERE raw_id IN ({placeholders}) ORDER BY raw_id",
                resolved,
            ).fetchall()
            raw_targets = tuple(
                ExcisionRawTarget(raw_id=str(r[0]), blob_hash=bytes(r[1]), source_path=str(r[2])) for r in rows
            )
            containers = _resolve_container_disposition(source, tuple(t.raw_id for t in raw_targets))
        if has_marker_inputs:
            marker_input_targets = marker_input_excision_targets_sync(
                source,
                target_session_ids=target_session_ids,
                target_raw_ids=marker_target_raw_ids,
            )
        hook_event_ids = _session_hook_event_ids(source, session_id)
        material_ids, material_blob_hashes = _session_material_targets(source, session_id)

    index_marker_witnesses: list[IndexMarkerExcisionTarget] = []
    if index_db.exists() and marker_input_targets:
        incarnation = index.execute(
            "SELECT incarnation_id,device,inode FROM ingest_index_incarnation WHERE singleton=1"
        ).fetchone()
        physical = index_db.stat()
        for marker in sorted(marker_input_targets, key=lambda marker: marker.identity):
            row = index.execute(
                "SELECT carrier_digest,incarnation_id,dispositions_json FROM ingest_marker_witnesses WHERE request_key=?",
                (marker.identity,),
            ).fetchone()
            if row is None:
                continue
            if (
                incarnation is None
                or tuple(incarnation[1:]) != (physical.st_dev, physical.st_ino)
                or row[:2] != (marker.carrier_digest, incarnation[0])
                or not isinstance(row[2], str)
            ):
                raise ValueError("Excision marker witness differs from its Source carrier or Index incarnation")
            index_marker_witnesses.append(
                IndexMarkerExcisionTarget(
                    marker.identity, row[0], row[1], hashlib.sha256(row[2].encode("utf-8")).digest()
                )
            )

    return ExcisionTarget(
        session_id=session_id,
        index_marker_witnesses=tuple(index_marker_witnesses),
        session_exists=session_exists,
        session_content_hash=session_content_hash,
        raw_targets=raw_targets,
        message_ids=message_ids,
        block_ids=block_ids,
        hook_event_ids=hook_event_ids,
        fact_raw_ids=fact_raw_ids,
        sidecar_raw_ids=sidecar_raw_ids,
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
    while True:
        if compute_cancel_requested():
            raise asyncio.CancelledError("excision revision closure cancelled by its owner")
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


def _bind_cascade_container_disposition(
    archive: ArchiveStore, targets: tuple[ExcisionTarget, ...]
) -> tuple[ExcisionTarget, ...]:
    """Resolve shared container liveness against every raw in the cascade."""
    raw_ids = tuple(dict.fromkeys(raw.raw_id for target in targets for raw in target.raw_targets))
    source_db = archive.source_db_path
    if not raw_ids or not source_db.exists() or not targets:
        return targets
    disposition = _resolve_container_disposition(archive.source_connection, raw_ids)
    # Container disposition is applied before raw deletion. Put the union
    # disposition on one target so shared items are removed exactly once.
    return tuple(
        replace(target, containers=disposition if index == 0 else ContainerDisposition())
        for index, target in enumerate(targets)
    )


def _session_work_event_raw_ids(conn: sqlite3.Connection, session_id: str) -> tuple[str, ...]:
    """Raw ids of the agent work events retained for this session.

    Each work event is its own logical source, so neither the transcript's
    revision closure nor ``sessions.raw_id`` reaches it. Its raw row carries
    the annotated session's ``(origin, native_id)``, which is the link.
    """
    origin, _, native_id = session_id.partition(":")
    if not origin or not native_id:
        return ()
    return tuple(
        str(row[0])
        for row in conn.execute(
            "SELECT raw_id FROM raw_sessions WHERE raw_id GLOB ? AND origin = ? AND native_id = ?",
            (f"{WORK_EVENT_RAW_ID_PREFIX}*", origin, native_id),
        ).fetchall()
    )


#: Reason prefix both sidecar joins give the debt of a file no transcript
#: claims (``no_owning_tool_result_block``, ``no_owning_tool_call``). Every
#: other debt reason is recorded by the transcript that owns the file.
_OWNERLESS_SIDECAR_REASON_PREFIX = "no_owning_"


@dataclass(frozen=True, slots=True)
class _SidecarOwnership:
    """Which tool-output sidecar files a session's own join claimed.

    Read from the session's derived rows, by the join's own rules
    (``sources/live/tool_result_sidecars.py``,
    ``sources/live/gemini_tool_output_sidecars.py``): a file whose stem is a
    ``tool_id`` of one of the session's ``tool_result`` blocks, and every file
    a sidecar event of the session names -- matched, or debt the join
    recorded against a file it had already resolved to this transcript
    (oversize, unreadable, less complete than the inline text). Both joins
    record the one ownerless outcome with a ``no_owning_*`` reason: a file no
    transcript claims is owned by no session and stays. Both scopes are
    shared: a Claude Code ``tool-results/`` directory by the parent and its
    subagent transcripts, a gemini-cli ``tool-outputs/session-<id>/``
    directory by every chat of one CLI process.
    """

    tool_ids: frozenset[str] = frozenset()
    filenames: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.tool_ids or self.filenames)

    def owns(self, filename: str) -> bool:
        return filename in self.filenames or filename.rsplit(".", 1)[0] in self.tool_ids


def _session_sidecar_ownership(conn: sqlite3.Connection, session_id: str) -> _SidecarOwnership:
    """Read the session's sidecar ownership evidence from the index tier."""
    tool_ids = frozenset(
        str(row[0])
        for row in conn.execute(
            "SELECT DISTINCT tool_id FROM blocks WHERE session_id = ? AND block_type = ? AND tool_id IS NOT NULL",
            (session_id, BlockType.TOOL_RESULT.value),
        ).fetchall()
        if row[0]
    )
    event_types = tuple(sorted(SIDECAR_BLOB_EVENT_TYPES))
    placeholders = ",".join("?" for _ in event_types)
    filenames: set[str] = set()
    for (payload_json,) in conn.execute(
        f"SELECT payload_json FROM session_events WHERE session_id = ? AND event_type IN ({placeholders})",
        (session_id, *event_types),
    ).fetchall():
        payload = json.loads(str(payload_json)) if payload_json else None
        if not isinstance(payload, dict):
            continue
        reason = payload.get("reason")
        if payload.get("acquisition_status") == "debt" and (
            not isinstance(reason, str) or reason.startswith(_OWNERLESS_SIDECAR_REASON_PREFIX)
        ):
            continue
        filename = payload.get("filename")
        if isinstance(filename, str) and filename:
            filenames.add(filename)
    return _SidecarOwnership(tool_ids=tool_ids, filenames=frozenset(filenames))


def _path_prefix_range(prefix: str) -> tuple[str, str]:
    """Half-open ``source_path`` range covering every path starting with ``prefix``.

    A range comparison keeps ``idx_raw_sessions_source_path`` usable where a
    ``LIKE`` with the ``ESCAPE`` a path needs would not.
    """
    return prefix, prefix[:-1] + chr(ord(prefix[-1]) + 1)


def _session_sidecar_raw_ids(
    conn: sqlite3.Connection,
    session_id: str,
    transcript_raw_ids: Sequence[str],
    ownership: _SidecarOwnership,
) -> tuple[str, ...]:
    """Raw ids of every retained revision of the sidecars this session owns.

    The scope directory comes from each transcript's retained ``source_path``
    by the same path law acquisition and derivation use
    (``resolve_tool_results_dir`` / ``resolve_tool_outputs_dir``); a scope
    member is a direct child of it, as a directory listing would have
    yielded. Every retained revision of an owned file is returned, not only
    the latest: each one holds that output's bytes.
    """
    origin, _, native_id = session_id.partition(":")
    if not transcript_raw_ids or not native_id:
        return ()
    placeholders = ",".join("?" for _ in transcript_raw_ids)
    transcript_paths = sorted(
        {
            str(row[0])
            for row in conn.execute(
                f"SELECT source_path FROM raw_sessions WHERE raw_id IN ({placeholders}) AND origin = ?",
                (*transcript_raw_ids, origin),
            ).fetchall()
            if row[0]
        }
    )
    resolved: set[str] = set()
    for source_path in transcript_paths:
        if origin == Origin.CLAUDE_CODE_SESSION.value:
            if not source_path.endswith(".jsonl"):
                continue
            directory = resolve_tool_results_dir(source_path)
            if directory is None:
                continue
            resolved.update(_owned_scope_children(conn, directory.as_posix(), ownership, skip_prefix="hook-"))
        elif origin == Origin.GEMINI_CLI_SESSION.value:
            resolved.update(_gemini_owned_sidecar_raw_ids(conn, source_path, native_id, ownership))
    return tuple(sorted(resolved))


def _owned_scope_children(
    conn: sqlite3.Connection,
    directory: str,
    ownership: _SidecarOwnership,
    *,
    skip_prefix: str | None = None,
) -> set[str]:
    """Raw ids of the owned files directly inside one scope directory.

    ``skip_prefix`` names files the join never treats as sidecars (Claude
    Code's ``hook-*`` stdout captures, which hook-event excision owns).
    """
    low, high = _path_prefix_range(f"{directory}/")
    found: set[str] = set()
    for raw_id, source_path in conn.execute(
        "SELECT raw_id, source_path FROM raw_sessions WHERE source_path >= ? AND source_path < ?",
        (low, high),
    ):
        filename = str(source_path)[len(low) :]
        if not filename or "/" in filename:
            continue
        if skip_prefix is not None and filename.startswith(skip_prefix):
            continue
        if ownership.owns(filename):
            found.add(str(raw_id))
    return found


def _gemini_owned_sidecar_raw_ids(
    conn: sqlite3.Connection, source_path: str, native_id: str, ownership: _SidecarOwnership
) -> set[str]:
    """Owned sidecars of one gemini-cli chat snapshot.

    The scope is ``tool-outputs/session-<sessionId>/`` for the wire
    ``sessionId``, which the chat's native id begins with, followed by a
    colon (``gemini_cli_chat_identity`` composes
    ``<sessionId>:<kind>:<startTime>``). Every prefix of the native id that
    ends before a colon is a candidate, and ``resolve_tool_outputs_dir``, the
    path law the join itself uses, turns each into its directory.
    """
    directories: set[str] = set()
    for index, character in enumerate(native_id):
        if character != ":":
            continue
        directory = resolve_tool_outputs_dir(source_path, native_id[:index])
        if directory is not None:
            directories.add(directory.as_posix())
    found: set[str] = set()
    for scope_directory in sorted(directories):
        found.update(_owned_scope_children(conn, scope_directory, ownership))
    return found


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


def find_lineage_dependents(archive: ArchiveStore, session_id: str) -> tuple[str, ...]:
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

    index_db = archive.index_db_path
    if not index_db.exists():
        return ()

    dependents: list[str] = []
    seen = {session_id}
    frontier = [session_id]
    while frontier:
        parent_id = frontier.pop()
        rows = archive._conn.execute(
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


class ExcisionBlobReferenceUnknownError(RuntimeError):
    """Raised when excision cannot tell whether another session references a blob.

    Marking a blob another session still references would forget that
    session's content; leaving an unreferenced blob unmarked would let the
    excised session's bytes be re-acquired. With the answer blocked (an
    unknown ``blob_refs`` type, a missing owner table), the apply refuses and
    its source transaction rolls back.
    """

    def __init__(self, *, blob_hash: bytes, blockers: tuple[str, ...]) -> None:
        self.blob_hash = blob_hash
        self.blockers = blockers
        super().__init__(
            f"cannot decide whether blob {blob_hash.hex()} is referenced outside the excision: {'; '.join(blockers)}"
        )


class LineageDependentsError(RuntimeError):
    """Raised when excising a session would break composed reads of its lineage.

    ``session_id`` is a prefix-sharing lineage parent for the listed
    dependent sessions (see :func:`find_lineage_dependents`); excising it
    without also excising them would delete bytes their composed transcripts
    depend on, leaving a dangling ``branch_point_message_id`` with no
    warning. Pass ``cascade_lineage=True`` to the audited Excision operation
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
    targets: tuple[ExcisionTarget, ...] = ()
    user_frame_epoch: int | None = None
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
    #: Tool-output sidecar raw rows the session owns (polylogue-8j9rh).
    #: Already counted in ``source_raw_rows``, like ``source_fact_rows``.
    source_sidecar_rows: int = 0
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
    #: Content-free carrier digests identifying marker evidence in this plan.
    marker_input_digests: tuple[str, ...] = ()

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
            "source_sidecar_rows": self.source_sidecar_rows,
            "source_container_members": self.source_container_members,
            "source_container_items": self.source_container_items,
            "retained_source_containers": list(self.retained_source_containers),
            "source_materials": self.source_materials,
            "source_marker_inputs_pending": self.source_marker_inputs_pending,
            "source_marker_inputs_accepted": self.source_marker_inputs_accepted,
            "marker_input_digests": list(self.marker_input_digests),
        }


def _excision_user_frame_epoch(archive: ArchiveStore) -> int:
    """Retain the assertion population precondition in the frozen plan."""
    from polylogue.storage.io_phase_metrics import connection_cursor
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    with connection_cursor(
        archive._conn, "SELECT epoch FROM user_tier.query_unit_frame_state WHERE singleton=1"
    ) as cursor:
        frame = cursor.fetchone()
    if frame is None or type(frame[0]) is not int or frame[0] < 0:
        raise ReferenceSealError("Excision requires its canonical durable User assertion frame")
    return frame[0]


def plan_session_excision(archive: ArchiveStore, session_id: str, *, cascade_lineage: bool = False) -> ExcisionPlan:
    """Enumerate exactly what an apply would remove from an owned read snapshot."""

    dependent_ids = find_lineage_dependents(archive, session_id)
    if dependent_ids and not cascade_lineage:
        raise LineageDependentsError(session_id=session_id, dependent_session_ids=dependent_ids)
    target_session_ids = frozenset((*dependent_ids, session_id)) if cascade_lineage else frozenset({session_id})
    session_ids = (*dependent_ids, session_id) if cascade_lineage else (session_id,)
    targets = tuple(
        _resolve_session_excision_target(archive, candidate, target_session_ids=target_session_ids)
        for candidate in session_ids
    )
    if cascade_lineage:
        targets = _bind_cascade_container_disposition(archive, targets)
    owned_marker_keys: set[str] = set()
    owned_targets: list[ExcisionTarget] = []
    for current in targets:
        owned = tuple(
            witness for witness in current.index_marker_witnesses if witness.request_key not in owned_marker_keys
        )
        owned_marker_keys.update(witness.request_key for witness in owned)
        owned_targets.append(replace(current, index_marker_witnesses=owned))
    targets = tuple(owned_targets)
    target = targets[-1]
    user_frame_epoch = _excision_user_frame_epoch(archive)
    if not target.found:
        return ExcisionPlan(
            session_id=session_id,
            found=False,
            targets=targets,
            user_frame_epoch=user_frame_epoch,
            lineage_dependent_session_ids=dependent_ids,
        )

    raw_targets = tuple({raw.raw_id: raw for current in targets for raw in current.raw_targets}.values())
    message_ids = tuple({message_id for current in targets for message_id in current.message_ids})
    block_ids = tuple({block_id for current in targets for block_id in current.block_ids})
    material_ids = tuple({material_id for current in targets for material_id in current.material_ids})
    material_blob_hashes = tuple({blob_hash for current in targets for blob_hash in current.material_blob_hashes})
    marker_targets = tuple(
        {marker.identity: marker for current in targets for marker in current.marker_input_targets}.values()
    )
    refs = tuple(ref for current in targets for ref in _target_refs(current))

    source_db = archive.source_db_path
    connection = archive._conn
    source_connection = archive.source_connection

    source_blob_refs = 0
    already_excised: list[str] = []
    if source_db.exists() and (raw_targets or material_ids):
        for material_hash in material_blob_hashes:
            if is_blob_hash_excised(source_connection, material_hash):
                already_excised.append(material_hash.hex())
        for raw_target in raw_targets:
            row = source_connection.execute(
                "SELECT COUNT(*) FROM blob_refs WHERE ref_id = ?",
                (raw_target.raw_id,),
            ).fetchone()
            source_blob_refs += int(row[0]) if row else 0
            if is_blob_hash_excised(source_connection, raw_target.blob_hash):
                already_excised.append(raw_target.blob_hash.hex())

    embeddings_vectors = 0
    schema_versions = archive.operation_schema_versions or {}
    if "embeddings" not in schema_versions or not message_ids:
        embeddings_vectors = 0
    else:
        placeholders = ",".join("?" for _ in message_ids)
        # message_embeddings/message_embeddings_meta are content-addressed
        # (keyed by vector_derivation_hash, polylogue-q88p) and may be shared
        # with messages outside this target; count message-scoped references.
        row = connection.execute(
            f"SELECT COUNT(*) FROM embeddings_tier.message_embedding_refs WHERE message_id IN ({placeholders})",
            message_ids,
        ).fetchone()
        embeddings_vectors = int(row[0]) if row else 0

    placeholders = ",".join("?" for _ in refs)
    row = connection.execute(
        f"SELECT COUNT(*) FROM user_tier.assertions WHERE target_ref IN ({placeholders})",
        refs,
    ).fetchone()
    user_assertions = int(row[0]) if row else 0

    return ExcisionPlan(
        session_id=session_id,
        found=True,
        targets=targets,
        user_frame_epoch=user_frame_epoch,
        source_raw_rows=len(raw_targets),
        source_blob_refs=source_blob_refs,
        index_sessions=sum(current.session_exists for current in targets),
        index_messages=len(message_ids),
        index_blocks=len(block_ids),
        embeddings_vectors=embeddings_vectors,
        user_assertions=user_assertions,
        already_excised_blob_hashes=tuple(already_excised),
        lineage_dependent_session_ids=dependent_ids,
        source_hook_events=len({item for current in targets for item in current.hook_event_ids}),
        source_fact_rows=len({item for current in targets for item in current.fact_raw_ids}),
        source_sidecar_rows=len({item for current in targets for item in current.sidecar_raw_ids}),
        source_container_members=len(
            {
                (member.source_generation_id, member.source_item_id, member.record_coordinate)
                for current in targets
                for member in current.containers.members
            }
        ),
        source_container_items=len(
            {
                (item.source_generation_id, item.source_item_id)
                for current in targets
                for item in current.containers.removable_items
            }
        ),
        retained_source_containers=tuple(
            dict.fromkeys(item.label for current in targets for item in current.containers.retained_items)
        ),
        source_materials=len(material_ids),
        # A source-first interruption retains only terminal marker evidence.
        # Count it so recovery preview/prepare matches the retry receipt that
        # removes its rebuildable witness.
        source_marker_inputs_pending=sum(marker.state == "pending" for marker in marker_targets),
        source_marker_inputs_accepted=sum(marker.state == "accepted" for marker in marker_targets),
        marker_input_digests=tuple(dict.fromkeys(marker.carrier_digest for marker in marker_targets)),
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
    #: Blob hashes the excised session owned that a session outside the
    #: excision still references. They stay readable for that session and get
    #: no ``excised_content`` marker; excising the last referencing session
    #: marks them.
    shared_blob_hashes: tuple[str, ...] = ()
    marker_input_digests: tuple[str, ...] = ()
    counts: dict[str, int] = field(default_factory=dict)
    # Populated only when the audited Excision operation cascaded across a
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
            "shared_blob_hashes": list(self.shared_blob_hashes),
            "marker_input_digests": list(self.marker_input_digests),
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


def _load_excision_source_rows(
    seal: PreparedIndexMutation,
    table: str,
    predicate: str,
    parameters: tuple[object, ...],
) -> None:
    """Load exact read dependencies from the pinned original Source only.

    Callers supply canonical table/predicate pairs. Loading does not grant a
    writable role, and touched original images must never resurrect deletes
    or overwrite an earlier statement's selected postimage.
    """
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    after: int | None = None
    while True:
        with seal.original_rows(
            "source",
            f'SELECT rowid FROM "{table}" WHERE ({predicate}) AND (? IS NULL OR rowid>?) ORDER BY rowid LIMIT 256',
            (*parameters, after, after),
        ) as rows:
            page = tuple(int(row[0]) for row in rows)
        if not page:
            return
        for rowid in page:
            if seal.source_row_is_touched(table, rowid):
                continue
            image = seal.retain_tier_row("source", table, rowid)
            if image is None:
                raise ReferenceSealError("original Excision Source input disappeared inside its pinned snapshot")
            seal.load_source_row(image)
        after = page[-1]


# Complete incoming raw FK family in canonical Source1 and released003.
# These are hydration dependencies and physical cascade roles, not arbitrary
# table-level write authority. New schema siblings must update this family.
_EXCISION_RAW_FK_CHILDREN = (
    "raw_container_coordinates",
    "raw_capture_observations",
    "raw_session_memberships",
    "raw_membership_census",
    "raw_legacy_append_resynthesis_receipts",
    "raw_authority_parser_census",
    "raw_authority_verdicts",
    "raw_artifacts",
    "raw_profile_identity_receipts",
    "source_items",
    "source_item_raw_members",
)


def _load_excision_source_target(seal: PreparedIndexMutation, target: ExcisionTarget) -> None:
    """Load complete original owned inputs before any closure deletion.

    The frozen domain target is provenance; exact original physical row cells
    remain the authority used by Native capture and live effect comparison.
    All canonical incoming raw FK carriers are retained before the parent is
    removed, including Source profile receipts and SET NULL dispositions.
    """
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    if excision_target_replay(target) != excision_target_replay(seal.original_excision_target(target.session_id)):
        raise ReferenceSealError("Source cleanup differs from its exact retained begun Excision target")

    # The witness is created after durable begin. A legitimate Source writer
    # can have added another session-addressed owner in that gap, even while
    # every named target still exists unchanged. Named-row equality alone
    # cannot certify that the frozen preview still covers this population.
    origin, _, native_id = target.session_id.partition(":")
    with seal.original_rows(
        "source",
        "SELECT count(*) FROM raw_hook_events WHERE origin=? AND session_native_id=?",
        (origin, native_id),
    ) as rows:
        hook_count = rows.fetchone()[0]
    with seal.original_rows(
        "source", "SELECT count(*) FROM material_observations WHERE referrer_ref=?", (target.session_id,)
    ) as rows:
        material_count = rows.fetchone()[0]
    if hook_count != len(target.hook_event_ids) or material_count != len(target.material_ids):
        raise ReferenceSealError("session-addressed Source owners changed after the retained begun preview")

    def owned_rows(
        table: str,
        predicate: str,
        parameters: tuple[object, ...],
        *,
        hash_column: str,
        revision_column: str | None,
        prior_revision_value: str | None = None,
    ) -> int:
        _load_excision_source_rows(seal, table, predicate, parameters)
        count = 0
        after: int | None = None
        while True:
            with seal.original_rows(
                "source",
                f'SELECT rowid FROM "{table}" WHERE ({predicate}) AND (? IS NULL OR rowid>?) ORDER BY rowid LIMIT 256',
                (*parameters, after, after),
            ) as rows:
                page = tuple(int(row[0]) for row in rows)
            if not page:
                return count
            for rowid in page:
                image = seal.retain_tier_row("source", table, rowid)
                if image is None:
                    raise ReferenceSealError("frozen Excision owner disappeared from its original snapshot")
                fields = dict(zip(image.columns, image.cells, strict=True))
                hash_cell = fields[hash_column]
                if seal._literal_cell_metadata(hash_cell)[0] == "null":
                    continue
                prior = (
                    seal.retain_literal_scalar(prior_revision_value)
                    if prior_revision_value is not None
                    else fields[revision_column]
                    if revision_column is not None
                    else seal.retain_literal_scalar(None)
                )
                seal.record_excision_source_blob(target.session_id, hash_cell, prior)
                count += 1
            after = page[-1]

    for member in target.containers.members:
        predicate = "source_generation_id=? AND source_item_id=? AND record_coordinate=?"
        parameters = (member.source_generation_id, member.source_item_id, member.record_coordinate)
        if (
            owned_rows(
                "source_item_raw_members",
                predicate,
                parameters,
                hash_column="raw_blob_hash",
                revision_column=None,
                prior_revision_value=f"{member.source_item_id}:{member.record_coordinate}",
            )
            != 1
        ):
            raise ReferenceSealError("frozen Excision container member lacks its original exact coordinate")
        with seal.original_rows(
            "source", f"SELECT rowid FROM source_item_raw_members WHERE {predicate}", parameters
        ) as rows:
            member_row = rows.fetchone()
        assert member_row is not None
        image = seal.retain_tier_row("source", "source_item_raw_members", int(member_row[0]))
        assert image is not None
        fields = dict(zip(image.columns, image.cells, strict=True))
        if not seal._literal_scalar_equal(fields["raw_blob_hash"], member.raw_blob_hash):
            raise ReferenceSealError("frozen Excision container member differs from its original bytes")
        # A lineage cascade resolves container liveness over the whole frozen
        # closure, so a member can belong to another cascade target's raw.
        # Ownership is therefore checked against the begun closure, exactly as
        # removable items are below; without a cascade the closure is this
        # session's own raws.
        if not seal.excision_source_raw_in_closure(fields["raw_id"]):
            raise ReferenceSealError("frozen Excision container member differs from its original raw ownership")

    for item in (*target.containers.removable_items, *target.containers.retained_items):
        predicate = "source_generation_id=? AND source_item_id=?"
        item_parameters = (item.source_generation_id, item.source_item_id)
        _load_excision_source_rows(seal, "source_items", predicate, item_parameters)
        with seal.original_rows("source", f"SELECT rowid FROM source_items WHERE {predicate}", item_parameters) as rows:
            item_row = rows.fetchone()
        item_image = None if item_row is None else seal.retain_tier_row("source", "source_items", int(item_row[0]))
        if item_image is None:
            raise ReferenceSealError("frozen Excision container item lacks its original physical owner")
        item_fields = dict(zip(item_image.columns, item_image.cells, strict=True))
        if not seal._literal_scalar_equal(item_fields["blob_hash"], item.blob_hash):
            raise ReferenceSealError("frozen Excision container item differs from its original bytes")
        _load_excision_source_rows(seal, "source_item_raw_members", predicate, item_parameters)
        _load_excision_source_rows(seal, "source_item_member_dispositions", predicate, item_parameters)
        if item in target.containers.removable_items:
            after_member: int | None = None
            while True:
                with seal.original_rows(
                    "source",
                    f"SELECT rowid FROM source_item_raw_members WHERE {predicate} AND raw_id IS NOT NULL "
                    "AND (? IS NULL OR rowid>?) ORDER BY rowid LIMIT 256",
                    (*item_parameters, after_member, after_member),
                ) as rows:
                    member_page = tuple(int(row[0]) for row in rows)
                if not member_page:
                    break
                for rowid in member_page:
                    member_image = seal.retain_tier_row("source", "source_item_raw_members", rowid)
                    if member_image is None:
                        raise ReferenceSealError("original container member disappeared from its pinned view")
                    member_fields = dict(zip(member_image.columns, member_image.cells, strict=True))
                    if not seal.excision_source_raw_in_closure(member_fields["raw_id"]):
                        raise ReferenceSealError("frozen removable container gained an independent surviving raw owner")
                after_member = member_page[-1]
        if item in target.containers.removable_items and item.blob_hash is not None:
            owned_rows(
                "source_items",
                predicate,
                item_parameters,
                hash_column="blob_hash",
                revision_column=None,
                prior_revision_value=item.label,
            )

    for raw in target.raw_targets:
        _load_excision_source_rows(seal, "raw_sessions", "raw_id=?", (raw.raw_id,))
        with seal.original_rows("source", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw.raw_id,)) as rows:
            found = rows.fetchone()
        original = None if found is None else seal.retain_tier_row("source", "raw_sessions", int(found[0]))
        if original is None:
            raise ReferenceSealError("frozen Excision raw lacks its original physical owner")
        fields = dict(zip(original.columns, original.cells, strict=True))
        if not seal._literal_scalar_equal(fields["blob_hash"], raw.blob_hash) or not seal._literal_scalar_equal(
            fields["source_path"], raw.source_path
        ):
            raise ReferenceSealError("frozen Excision raw differs from its original byte/acquisition identity")
        seal.record_excision_source_blob(target.session_id, fields["blob_hash"], fields["raw_id"])
        owned_rows(
            "blob_refs",
            "ref_id=? AND ref_type IN ('raw_payload','attachment')",
            (raw.raw_id,),
            hash_column="blob_hash",
            revision_column="ref_id",
        )
        for table in _EXCISION_RAW_FK_CHILDREN:
            _load_excision_source_rows(seal, table, "raw_id=?", (raw.raw_id,))
        _load_excision_source_rows(seal, "raw_existence_changes", "raw_id=?", (raw.raw_id,))

    for hook_id in target.hook_event_ids:
        origin, _, native_id = target.session_id.partition(":")
        with seal.original_rows(
            "source", "SELECT rowid FROM raw_hook_events WHERE hook_event_id=?", (hook_id,)
        ) as rows:
            hook_row = rows.fetchone()
        hook_image = None if hook_row is None else seal.retain_tier_row("source", "raw_hook_events", int(hook_row[0]))
        if hook_image is None:
            raise ReferenceSealError("frozen Excision hook lacks its original physical owner")
        hook_fields = dict(zip(hook_image.columns, hook_image.cells, strict=True))
        if (
            not origin
            or not native_id
            or not all(
                seal._literal_scalar_equal(hook_fields[column], value)
                for column, value in (("origin", origin), ("session_native_id", native_id))
            )
        ):
            raise ReferenceSealError("frozen Excision hook differs from its original session ownership")
        if (
            owned_rows(
                "raw_hook_events",
                "hook_event_id=?",
                (hook_id,),
                hash_column="blob_hash",
                revision_column="hook_event_id",
            )
            == 0
        ):
            # NULL payload hashes do not make an existing logical event absent.
            with seal.original_rows(
                "source", "SELECT 1 FROM raw_hook_events WHERE hook_event_id=?", (hook_id,)
            ) as rows:
                if rows.fetchone() is None:
                    raise ReferenceSealError("frozen Excision hook lacks its original physical owner")
        owned_rows(
            "hook_event_carriers",
            "hook_event_id=?",
            (hook_id,),
            hash_column="blob_hash",
            revision_column="hook_event_id",
        )
        owned_rows(
            "blob_refs",
            "ref_type='hook_payload' AND ref_id=?",
            (hook_id,),
            hash_column="blob_hash",
            revision_column="ref_id",
        )

    material_hashes: set[bytes] = set()
    for material_id in target.material_ids:
        with seal.original_rows(
            "source",
            "SELECT blob_hash FROM material_observations WHERE material_id=? AND referrer_ref=?",
            (material_id, target.session_id),
        ) as rows:
            material_row = rows.fetchone()
        if material_row is None:
            raise ReferenceSealError("frozen Excision material lacks its original session ownership")
        if material_row[0] is not None:
            material_hashes.add(bytes(material_row[0]))
        if (
            owned_rows(
                "material_observations",
                "material_id=? AND referrer_ref=?",
                (material_id, target.session_id),
                hash_column="blob_hash",
                revision_column=None,
            )
            == 0
        ):
            with seal.original_rows(
                "source",
                "SELECT 1 FROM material_observations WHERE material_id=? AND referrer_ref=?",
                (material_id, target.session_id),
            ) as rows:
                if rows.fetchone() is None:
                    raise ReferenceSealError("frozen Excision material lacks its original session ownership")
        _load_excision_source_rows(seal, "material_evidence_links", "material_id=?", (material_id,))
        _load_excision_source_rows(seal, "material_observations", "supersedes_material_id=?", (material_id,))

    if material_hashes != set(target.material_blob_hashes):
        raise ReferenceSealError("frozen Excision materials differ from their original byte identities")

    for marker in target.marker_input_targets:
        if marker.tombstoned:
            raise ReferenceSealError("earlier marker tombstones cannot prove this begun Excision Source completion")
        if marker.state == "pending":
            table, predicate = "pending_accepted_marker_inputs", "request_key=? AND raw_id=? AND carrier_digest=?"
        elif marker.state == "accepted":
            table, predicate = "accepted_marker_inputs", "identity=? AND raw_id=? AND payload_sha256=?"
        else:
            raise ReferenceSealError("frozen Excision marker has no canonical carrier state")
        parameters = (marker.identity, marker.raw_id, marker.carrier_digest)
        _load_excision_source_rows(seal, table, predicate, parameters)
        with seal.original_rows("source", f'SELECT 1 FROM "{table}" WHERE {predicate}', parameters) as rows:
            if rows.fetchone() is None:
                raise ReferenceSealError("frozen Excision marker lacks its original sealed carrier")
        _load_excision_source_rows(seal, "excised_marker_inputs", "identity=?", (marker.identity,))


def _excision_source_writable_keys(
    seal: PreparedIndexMutation, table: str, predicate: str, parameters: tuple[object, ...]
) -> Iterator[tuple[str, tuple[KnownTierCell, ...]]]:
    """Declare exact selected physical keys only after each reader settles."""
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    _columns, keys = seal._known_tier_table_shape("source", table)
    after: int | None = None
    while True:
        with seal.source_rows(
            f'SELECT rowid FROM "{table}" WHERE ({predicate}) AND (? IS NULL OR rowid>?) ORDER BY rowid LIMIT 256',
            (*parameters, after, after),
        ) as rows:
            page = tuple(int(row[0]) for row in rows)
        if not page:
            return
        for rowid in page:
            image = seal.retain_source_row(table, rowid)
            if image is None:
                raise ReferenceSealError("selected Excision target disappeared before its exact writable role")
            yield table, seal._known_row_key_cells(image, keys)
        after = page[-1]


def _excision_source_equal_predicate(
    seal: PreparedIndexMutation, fields: tuple[tuple[str, str | bytes | int | None], ...]
) -> tuple[str, tuple[object, ...]]:
    """Spell canonical equality operands through original literal slots."""
    expressions: list[str] = []
    parameters: tuple[object, ...] = ()
    for column, value in fields:
        if not column.isidentifier():
            raise ValueError("Excision equality requires its canonical column")
        expression, bindings = seal.source_literal_expression(seal.retain_literal_scalar(value))
        expressions.append(f'"{column}" IS {expression}')
        parameters += bindings
    if not expressions:
        raise ValueError("Excision deletion cannot omit its exact target predicate")
    return " AND ".join(expressions), parameters


def _stage_excision_raw_delete(seal: PreparedIndexMutation, raw_id: str) -> int:
    """Execute the canonical raw deletion with exact incoming FK roles."""
    predicate, parameters = _excision_source_equal_predicate(seal, (("raw_id", raw_id),))

    def roles() -> Iterator[tuple[str, tuple[KnownTierCell, ...]]]:
        # The incoming FK family is canonical Source1/003. SET NULL children
        # retain their complete original cells; only the raw pointer changes.
        for table in ("raw_sessions", *_EXCISION_RAW_FK_CHILDREN):
            yield from _excision_source_writable_keys(seal, table, predicate, parameters)

    with seal.source_statement(
        f"DELETE FROM raw_sessions WHERE {predicate}", parameters, table="raw_sessions", writable_targets=roles()
    ) as cursor:
        return max(cursor.rowcount, 0)


def _stage_excision_source_target(
    seal: PreparedIndexMutation, target: ExcisionTarget, *, excised_at_ms: int
) -> dict[str, int]:
    """Capture one frozen target's cleanup before whole-closure liveness.

    Every target must have been loaded before the first call. Classification
    happens only after all calls, so a shared hash sees the surviving Source
    postimage for the entire begun closure. No new candidate is discovered
    from rows already deleted by an earlier target.
    """
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    if excision_target_replay(target) != excision_target_replay(seal.original_excision_target(target.session_id)):
        raise ReferenceSealError("Source effects differ from the retained begun Excision target")
    if type(excised_at_ms) is not int or excised_at_ms < 0:
        raise ReferenceSealError("Source effects require their declared Excision time")
    counts = {
        "source_blob_refs": 0,
        "source_raw_rows": 0,
        "source_raw_existence_changes": 0,
        "source_hook_events": 0,
        "source_fact_rows": 0,
        "source_sidecar_rows": 0,
        "source_container_members": 0,
        "source_container_items": 0,
        "source_materials": 0,
        "source_marker_inputs_pending": 0,
        "source_marker_inputs_accepted": 0,
    }
    for marker in target.marker_input_targets:
        if marker.tombstoned or marker.state not in {"pending", "accepted"}:
            raise ReferenceSealError("Source effects require this attempt's original marker carrier")
        values = {
            "identity": marker.identity,
            "raw_id": marker.raw_id,
            "carrier_digest": marker.carrier_digest,
            "state": marker.state,
            "stream_id": marker.stream_id,
            "accepted_sequence": marker.accepted_sequence,
            "excised_at_ms": excised_at_ms,
        }
        cells = {column: seal.retain_literal_scalar(value) for column, value in values.items()}
        expressions = ["?"]
        bindings: tuple[object, ...] = (None,)
        for cell in cells.values():
            expression, parameters = seal.source_literal_expression(cell)
            expressions.append(expression)
            bindings += parameters
        with seal.source_statement(
            "INSERT INTO excised_marker_inputs(rowid,identity,raw_id,carrier_digest,state,stream_id,"
            "accepted_sequence,excised_at_ms) VALUES (" + ",".join(expressions) + ") "
            "ON CONFLICT(identity) DO NOTHING",
            bindings,
            table="excised_marker_inputs",
            writable_targets=(("excised_marker_inputs", (cells["identity"],)),),
            prepared_cells=cells,
            allocation_parameter=0,
        ):
            pass
        key, digest = ("request_key", "carrier_digest") if marker.state == "pending" else ("identity", "payload_sha256")
        predicate, parameters = _excision_source_equal_predicate(
            seal, ((key, marker.identity), ("raw_id", marker.raw_id), (digest, marker.carrier_digest))
        )
        if marker.state == "pending":
            with seal.source_statement(
                f"DELETE FROM pending_accepted_marker_inputs WHERE {predicate}",
                parameters,
                table="pending_accepted_marker_inputs",
                writable_targets=_excision_source_writable_keys(
                    seal, "pending_accepted_marker_inputs", predicate, parameters
                ),
            ) as cursor:
                counts["source_marker_inputs_pending"] += max(cursor.rowcount, 0)
        else:
            with seal.source_statement(
                f"DELETE FROM accepted_marker_inputs WHERE {predicate}",
                parameters,
                table="accepted_marker_inputs",
                writable_targets=_excision_source_writable_keys(seal, "accepted_marker_inputs", predicate, parameters),
            ) as cursor:
                counts["source_marker_inputs_accepted"] += max(cursor.rowcount, 0)

    # Member disposition precedes raw FK SET NULL. Keep the original composite
    # coordinates even when several targets share one manifest container.
    for member in target.containers.members:
        predicate, parameters = _excision_source_equal_predicate(
            seal,
            (
                ("source_generation_id", member.source_generation_id),
                ("source_item_id", member.source_item_id),
                ("record_coordinate", member.record_coordinate),
            ),
        )
        with seal.source_statement(
            f"DELETE FROM source_item_raw_members WHERE {predicate}",
            parameters,
            table="source_item_raw_members",
            writable_targets=_excision_source_writable_keys(seal, "source_item_raw_members", predicate, parameters),
        ) as cursor:
            counts["source_container_members"] += max(cursor.rowcount, 0)
    for item in target.containers.removable_items:
        predicate, parameters = _excision_source_equal_predicate(
            seal, (("source_generation_id", item.source_generation_id), ("source_item_id", item.source_item_id))
        )

        def item_roles(
            predicate: str = predicate, parameters: tuple[object, ...] = parameters
        ) -> Iterator[tuple[str, tuple[KnownTierCell, ...]]]:
            for table in ("source_items", "source_item_raw_members", "source_item_member_dispositions"):
                yield from _excision_source_writable_keys(seal, table, predicate, parameters)

        with seal.source_statement(
            f"DELETE FROM source_items WHERE {predicate}",
            parameters,
            table="source_items",
            writable_targets=item_roles(),
        ) as cursor:
            counts["source_container_items"] += max(cursor.rowcount, 0)
    for raw in target.raw_targets:
        predicate, parameters = _excision_source_equal_predicate(seal, (("ref_id", raw.raw_id),))
        predicate += " AND ref_type IN ('raw_payload','attachment')"
        with seal.source_statement(
            f"DELETE FROM blob_refs WHERE {predicate}",
            parameters,
            table="blob_refs",
            writable_targets=_excision_source_writable_keys(seal, "blob_refs", predicate, parameters),
        ) as cursor:
            counts["source_blob_refs"] += max(cursor.rowcount, 0)
        removed_raw_rows = _stage_excision_raw_delete(seal, raw.raw_id)
        counts["source_raw_rows"] += removed_raw_rows
        # Fact and sidecar raws are raw rows of a named class; the receipt
        # accounts for each class it removes, as the plan previews them.
        if raw.raw_id in target.fact_raw_ids:
            counts["source_fact_rows"] += removed_raw_rows
        if raw.raw_id in target.sidecar_raw_ids:
            counts["source_sidecar_rows"] += removed_raw_rows
        predicate, parameters = _excision_source_equal_predicate(seal, (("raw_id", raw.raw_id),))
        with seal.source_statement(
            f"DELETE FROM raw_existence_changes WHERE {predicate}",
            parameters,
            table="raw_existence_changes",
            writable_targets=_excision_source_writable_keys(seal, "raw_existence_changes", predicate, parameters),
        ) as cursor:
            counts["source_raw_existence_changes"] += max(cursor.rowcount, 0)
    for hook_id in target.hook_event_ids:
        predicate, parameters = _excision_source_equal_predicate(seal, (("hook_event_id", hook_id),))
        with seal.source_statement(
            f"DELETE FROM hook_event_carriers WHERE {predicate}",
            parameters,
            table="hook_event_carriers",
            writable_targets=_excision_source_writable_keys(seal, "hook_event_carriers", predicate, parameters),
        ):
            pass
        with seal.source_statement(
            f"DELETE FROM raw_hook_events WHERE {predicate}",
            parameters,
            table="raw_hook_events",
            writable_targets=_excision_source_writable_keys(seal, "raw_hook_events", predicate, parameters),
        ) as cursor:
            counts["source_hook_events"] += max(cursor.rowcount, 0)
        predicate, parameters = _excision_source_equal_predicate(seal, (("ref_id", hook_id),))
        predicate += " AND ref_type='hook_payload'"
        with seal.source_statement(
            f"DELETE FROM blob_refs WHERE {predicate}",
            parameters,
            table="blob_refs",
            writable_targets=_excision_source_writable_keys(seal, "blob_refs", predicate, parameters),
        ):
            pass
    # Drop incoming predecessor links before deleting any material. A surviving
    # observation keeps all its other original cells and its own identity.
    for material_id in target.material_ids:
        predicate, parameters = _excision_source_equal_predicate(seal, (("supersedes_material_id", material_id),))
        with seal.source_statement(
            f"UPDATE material_observations SET supersedes_material_id=NULL WHERE {predicate}",
            parameters,
            table="material_observations",
            writable_targets=_excision_source_writable_keys(seal, "material_observations", predicate, parameters),
        ):
            pass
    for material_id in target.material_ids:
        predicate, parameters = _excision_source_equal_predicate(seal, (("material_id", material_id),))

        def material_roles(
            predicate: str = predicate, parameters: tuple[object, ...] = parameters
        ) -> Iterator[tuple[str, tuple[KnownTierCell, ...]]]:
            for table in ("material_observations", "material_evidence_links"):
                yield from _excision_source_writable_keys(seal, table, predicate, parameters)

        with seal.source_statement(
            f"DELETE FROM material_observations WHERE {predicate}",
            parameters,
            table="material_observations",
            writable_targets=material_roles(),
        ) as cursor:
            counts["source_materials"] += max(cursor.rowcount, 0)
    return counts


class _PreparedExcisionBlobSourceRead:
    """Canonical liveness reads on the same selected Excision Source state.

    Original matching rows are dependencies, never write permissions. The
    shared classifier remains the sole owner map and liveness decision.
    """

    def __init__(self, seal: PreparedIndexMutation) -> None:
        self._seal = seal

    def session_blob_global_blockers(self) -> tuple[str, ...]:

        return ConnectionSessionBlobLivenessRead(self._seal.observer("source")).session_blob_global_blockers()

    def session_blob_owner_available(self, owner: BlobOwner) -> bool:

        return ConnectionSessionBlobLivenessRead(self._seal.observer("source")).session_blob_owner_available(owner)

    def session_blob_direct_hashes(self, owner: BlobOwner, hashes: tuple[bytes, ...]) -> tuple[bytes, ...]:
        from polylogue.storage.blob_liveness import session_blob_direct_query

        sql, parameters = session_blob_direct_query(owner, hashes)
        if not hashes:
            return ()
        assert owner.blob_column is not None
        marks = ",".join("?" for _ in hashes)
        _load_excision_source_rows(self._seal, owner.table, f'"{owner.blob_column}" IN ({marks})', hashes)
        with self._seal.source_rows(sql, parameters) as rows:
            return tuple(bytes(row[0]) for row in rows)

    def session_blob_ledger_hashes(self, owner: BlobOwner, hashes: tuple[bytes, ...]) -> tuple[bytes, ...]:
        from polylogue.storage.blob_liveness import session_blob_ledger_query

        sql, parameters = session_blob_ledger_query(owner, hashes)
        if not hashes:
            return ()
        assert owner.ref_type is not None and owner.referent_column is not None
        marks = ",".join("?" for _ in hashes)
        _load_excision_source_rows(
            self._seal, "blob_refs", f"blob_hash IN ({marks}) AND ref_type=?", (*hashes, owner.ref_type)
        )
        _load_excision_source_rows(
            self._seal,
            owner.table,
            f"EXISTS (SELECT 1 FROM blob_refs AS ref WHERE ref.blob_hash IN ({marks}) "
            f'AND ref.ref_type=? AND ref.ref_id="{owner.table}"."{owner.referent_column}")',
            (*hashes, owner.ref_type),
        )
        with self._seal.source_rows(sql, parameters) as rows:
            return tuple(bytes(row[0]) for row in rows)


class _PreparedExcisionSessionClosure(Set[str]):
    """Read membership from the same original begun closure without a Python copy."""

    def __init__(self, seal: PreparedIndexMutation) -> None:
        self._seal = seal

    def _require_original_membership(self) -> None:
        from polylogue.storage.sqlite.reference_seal import ReferenceSealError

        self._seal._require_new_work()
        if self._seal._begun_excision is None:
            raise ReferenceSealError("Excision membership lacks its original begun relation")

    def __contains__(self, session_id: object) -> bool:
        self._require_original_membership()
        if not isinstance(session_id, str):
            return False
        with self._seal._owned_cursor(
            self._seal._scratch, "SELECT 1 FROM temp.begun_excision_sessions WHERE session_id=?", (session_id,)
        ) as rows:
            return rows.fetchone() is not None

    def __len__(self) -> int:
        self._require_original_membership()
        with self._seal._owned_cursor(self._seal._scratch, "SELECT count(*) FROM temp.begun_excision_sessions") as rows:
            return int(rows.fetchone()[0])

    def __iter__(self) -> Iterator[str]:
        self._require_original_membership()
        after = -1
        while True:
            with self._seal._owned_cursor(
                self._seal._scratch,
                "SELECT ordinal,session_id FROM temp.begun_excision_sessions WHERE ordinal>? ORDER BY ordinal LIMIT 256",
                (after,),
            ) as rows:
                page = tuple((int(row[0]), str(row[1])) for row in rows)
            if not page:
                return
            for ordinal, session_id in page:
                self._require_original_membership()
                yield session_id
                after = ordinal


def _stage_excision_source_blob_dispositions(
    seal: PreparedIndexMutation,
    *,
    excluding_session_ids: Set[str],
    reason: str,
    actor: str,
    excised_at_ms: int,
) -> int:
    """Classify the whole closure's surviving postimage, then mark removals.

    Only the shared descriptor classifier decides liveness. Native candidates
    preserve per-target membership, while pages visit each distinct hash once.
    An uncertain owner refuses preparation before any live Source mutation.
    """
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    seal._require_excision_source_bookkeeping()
    if not (
        type(excluding_session_ids) is frozenset
        or isinstance(excluding_session_ids, _PreparedExcisionSessionClosure)
        and excluding_session_ids._seal is seal
    ):
        raise ReferenceSealError("classification requires the exact original frozen closure membership")
    with seal._owned_cursor(seal._scratch, "SELECT count(*) FROM temp.begun_excision_sessions") as rows:
        closure_count = rows.fetchone()[0]
    if len(excluding_session_ids) != closure_count:
        raise ReferenceSealError("blob classification exclusions differ from the retained begun closure")
    with seal._owned_cursor(seal._scratch, "SELECT session_id FROM temp.begun_excision_sessions") as rows:
        for (session_id,) in rows:
            if session_id not in excluding_session_ids:
                raise ReferenceSealError("blob classification cannot withhold an independent Index owner")
    source = _PreparedExcisionBlobSourceRead(seal)
    index = seal.observer("index")
    authority_blocker = index_liveness_authority_blocker(
        blob_root=seal.archive_root / "blob", index_path=seal.index_path, index_conn=index, record=False
    )
    fixed = {
        "hash_kind": seal.retain_literal_scalar("blob_hash"),
        "reason": seal.retain_literal_scalar(reason),
        "actor": seal.retain_literal_scalar(actor),
        "span_start": seal.retain_literal_scalar(None),
        "span_end": seal.retain_literal_scalar(None),
        "excised_at_ms": seal.retain_literal_scalar(excised_at_ms),
    }
    reservations_removed = 0
    after: bytes | None = None
    while page := seal.excision_source_blob_page(after):
        references = inspect_session_blob_references(
            source,
            tuple(blob_hash for blob_hash, _prior in page),
            index_conn=index,
            excluding_session_ids=excluding_session_ids,
            index_authority_blocker=authority_blocker,
        )
        for blob_hash, prior_revision in page:
            reference = references[blob_hash]
            if reference.state is LivenessState.BLOCKED:
                raise ExcisionBlobReferenceUnknownError(blob_hash=blob_hash, blockers=reference.blockers)
            removed = reference.state is LivenessState.UNREFERENCED
            seal.record_excision_blob_disposition(blob_hash, removed=removed)
            if not removed:
                continue
            _load_excision_source_rows(
                seal, "excised_content", "removed_hash=? AND hash_kind='blob_hash'", (blob_hash,)
            )
            hash_cell = seal.retain_literal_scalar(blob_hash)
            cells = {
                "removed_hash": hash_cell,
                "hash_kind": fixed["hash_kind"],
                "reason": fixed["reason"],
                "actor": fixed["actor"],
                "prior_revision": prior_revision,
                "span_start": fixed["span_start"],
                "span_end": fixed["span_end"],
                "excised_at_ms": fixed["excised_at_ms"],
            }
            expressions = ["?"]
            bindings: tuple[object, ...] = (None,)
            for cell in cells.values():
                expression, parameters = seal.source_literal_expression(cell)
                expressions.append(expression)
                bindings += parameters
            with seal.source_statement(
                "INSERT INTO excised_content(rowid,removed_hash,hash_kind,reason,actor,prior_revision,"
                "span_start,span_end,excised_at_ms) VALUES (" + ",".join(expressions) + ") "
                "ON CONFLICT(removed_hash,hash_kind) DO NOTHING",
                bindings,
                table="excised_content",
                writable_targets=(("excised_content", (hash_cell, fixed["hash_kind"])),),
                prepared_cells=cells,
                allocation_parameter=0,
            ):
                pass
            _load_excision_source_rows(seal, "blob_publication_reservations", "blob_hash=?", (blob_hash,))
            expression, parameters = seal.source_literal_expression(hash_cell)
            predicate = f"blob_hash={expression}"
            with seal.source_statement(
                f"DELETE FROM blob_publication_reservations WHERE {predicate}",
                parameters,
                table="blob_publication_reservations",
                writable_targets=_excision_source_writable_keys(
                    seal, "blob_publication_reservations", predicate, parameters
                ),
            ) as cursor:
                removed_reservations = max(cursor.rowcount, 0)
                reservations_removed += removed_reservations
            if seal._excision_source_receipts_ready:
                seal.record_excision_source_reservations(blob_hash, removed_reservations)
        after = page[-1][0]
    return reservations_removed


def _verify_original_excision_index_markers(seal: PreparedIndexMutation) -> None:
    """Bind every selected request key to its original native row before effects."""
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    witnesses: dict[str, IndexMarkerExcisionTarget] = {}
    markers: dict[str, str] = {}
    for session_id in _PreparedExcisionSessionClosure(seal):
        target = seal.original_excision_target(session_id)
        for marker in target.marker_input_targets:
            prior = markers.setdefault(marker.identity, marker.carrier_digest)
            if prior != marker.carrier_digest:
                raise ReferenceSealError("frozen Source markers disagree on their exact carrier")
        for witness in target.index_marker_witnesses:
            if witness.request_key in witnesses or markers.get(witness.request_key) != witness.carrier_digest:
                raise ReferenceSealError("frozen Index marker witness has duplicate or foreign ownership")
            witnesses[witness.request_key] = witness
    with seal._owned_cursor(
        seal._scratch,
        "CREATE TEMP TABLE excision_index_marker_current(request_key TEXT PRIMARY KEY,present INTEGER NOT NULL) STRICT",
    ):
        pass
    for key in markers:
        with seal.original_rows(
            "index", "SELECT rowid FROM ingest_marker_witnesses WHERE request_key=?", (key,)
        ) as rows:
            row = rows.fetchone()
        selected_witness = witnesses.get(key)
        with seal._owned_cursor(
            seal._scratch, "INSERT INTO temp.excision_index_marker_current VALUES (?,?)", (key, int(row is not None))
        ):
            pass
        if row is None:
            if selected_witness is not None and not seal._excision_recovery_user_committed:
                raise ReferenceSealError("original Index marker witness disappeared before its deletion")
            continue
        if selected_witness is None:
            raise ReferenceSealError("Index marker witness appeared outside its frozen completion")
        image = seal.retain_tier_row("index", "ingest_marker_witnesses", row[0])
        if image is None:
            raise ReferenceSealError("original Index marker disappeared inside its pinned snapshot")
        cells = dict(zip(image.columns, image.cells, strict=True))
        expected = {
            "request_key": selected_witness.request_key,
            "carrier_digest": selected_witness.carrier_digest,
            "incarnation_id": selected_witness.incarnation_id,
        }
        for name, value in expected.items():
            kind, length, _fixed = seal._literal_cell_metadata(cells[name])
            if kind != "text" or length != len(value.encode("utf-8")):
                raise ReferenceSealError("original Index marker changed its retained scalar")
            if b"".join(seal._literal_cell_chunks(cells[name])) != value.encode("utf-8"):
                raise ReferenceSealError("original Index marker changed its retained scalar")
        with seal.original_rows(
            "index", "SELECT incarnation_id,device,inode FROM ingest_index_incarnation WHERE singleton=1"
        ) as rows:
            incarnation = rows.fetchone()
        if (
            incarnation is None
            or incarnation[0] != selected_witness.incarnation_id
            or tuple(incarnation[1:]) != seal.index_identity[:2]
        ):
            raise ReferenceSealError("Index marker belongs to a stale physical incarnation")
        kind, _length, _fixed = seal._literal_cell_metadata(cells["dispositions_json"])
        if kind != "text":
            raise ReferenceSealError("original Index marker dispositions are not native text")
        digest = hashlib.sha256()
        for chunk in seal._literal_cell_chunks(cells["dispositions_json"]):
            digest.update(chunk)
        if digest.digest() != selected_witness.dispositions_sha256:
            raise ReferenceSealError("original Index marker changed its exact dispositions bytes")


def _verify_excision_source_target_terminal(
    seal: PreparedIndexMutation, target: ExcisionTarget, *, recovered: bool = False
) -> None:
    """Prove every frozen deletion family against the selected Source postimage."""
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    if recovered:
        if not seal._original_reads_active or not seal._excision_source_completion_staged:
            raise ReferenceSealError("recovered Source terminal proof requires its original event read window")
    else:
        seal._require_excision_source_bookkeeping()

    def read(sql: str, parameters: tuple[object, ...]) -> AbstractContextManager[sqlite3.Cursor]:
        return seal.original_rows("source", sql, parameters) if recovered else seal.source_rows(sql, parameters)

    def absent(table: str, predicate: str, parameters: tuple[object, ...]) -> None:
        with read(f"SELECT 1 FROM {table} WHERE {predicate} LIMIT 1", parameters) as rows:
            if rows.fetchone() is not None:
                raise ReferenceSealError("Excision Source terminal proof retains an owned deletion target")

    origin, separator, native_id = target.session_id.partition(":")
    if separator and origin and native_id:
        absent("raw_hook_events", "origin=? AND session_native_id=?", (origin, native_id))
    absent("material_observations", "referrer_ref=?", (target.session_id,))
    for raw in target.raw_targets:
        absent("raw_sessions", "raw_id=?", (raw.raw_id,))
        absent("blob_refs", "ref_id=? AND ref_type IN ('raw_payload','attachment')", (raw.raw_id,))
        absent("raw_existence_changes", "raw_id=?", (raw.raw_id,))
    for hook_id in target.hook_event_ids:
        absent("raw_hook_events", "hook_event_id=?", (hook_id,))
        absent("hook_event_carriers", "hook_event_id=?", (hook_id,))
        absent("blob_refs", "ref_id=? AND ref_type='hook_payload'", (hook_id,))
    for member in target.containers.members:
        absent(
            "source_item_raw_members",
            "source_generation_id=? AND source_item_id=? AND record_coordinate=?",
            (member.source_generation_id, member.source_item_id, member.record_coordinate),
        )
    for item in target.containers.removable_items:
        for table in ("source_items", "source_item_raw_members", "source_item_member_dispositions"):
            absent(
                table, "source_generation_id=? AND source_item_id=?", (item.source_generation_id, item.source_item_id)
            )
    for material_id in target.material_ids:
        absent("material_observations", "material_id=?", (material_id,))
        absent("material_evidence_links", "material_id=?", (material_id,))
        absent("material_observations", "supersedes_material_id=?", (material_id,))
    for marker in target.marker_input_targets:
        table, key = (
            ("pending_accepted_marker_inputs", "request_key")
            if marker.state == "pending"
            else ("accepted_marker_inputs", "identity")
        )
        absent(table, f"{key}=?", (marker.identity,))
        with read(
            "SELECT raw_id,carrier_digest,state,stream_id,accepted_sequence FROM excised_marker_inputs WHERE identity=?",
            (marker.identity,),
        ) as rows:
            terminal = rows.fetchone()
        if terminal is None or tuple(terminal) != (
            marker.raw_id,
            marker.carrier_digest,
            marker.state,
            marker.stream_id,
            marker.accepted_sequence,
        ):
            raise ReferenceSealError("Excision marker terminal proof differs from its frozen carrier")


def _stage_excision_source_closure(seal: PreparedIndexMutation, *, reason: str, actor: str, excised_at_ms: int) -> None:
    """Delete every frozen owner before classifying or sealing Source completion."""
    seal._require_excision_source_bookkeeping()
    session_ids = _PreparedExcisionSessionClosure(seal)
    for session_id in session_ids:
        target = seal.original_excision_target(session_id)
        _load_excision_source_target(seal, target)
    for session_id in session_ids:
        target = seal.original_excision_target(session_id)
        counts = _stage_excision_source_target(seal, target, excised_at_ms=excised_at_ms)
        seal.record_excision_source_target_counts(target.session_id, counts)
    for session_id in session_ids:
        target = seal.original_excision_target(session_id)
        _verify_excision_source_target_terminal(seal, target)
    _stage_excision_source_blob_dispositions(
        seal, excluding_session_ids=session_ids, reason=reason, actor=actor, excised_at_ms=excised_at_ms
    )
    seal.stage_excision_source_completion(occurred_at_ms=excised_at_ms)


def _stage_excision_user_receipt(
    seal: PreparedIndexMutation,
    target: ExcisionTarget,
    receipt: ExcisionReceipt,
    *,
    operation_id: str,
    attempt_id: str,
    plan_hash: str,
) -> ExcisionReceipt:
    """Capture canonical assertion cleanup on the already begun original owner.

    The caller owns the original read window and User producer. Counts are
    captured from actual staged DML; publication must consume that complete
    tape under the same physical removal plan before these become a receipt.
    """
    from polylogue.storage.sqlite.archive_tiers.user_write import assertion_upsert_statement, prepare_assertion_row
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    if seal._begun_excision != (operation_id, attempt_id, plan_hash):
        raise ReferenceSealError("User receipt requires its exact original begun Excision attempt")
    frozen_target = seal.original_excision_target(target.session_id)
    if excision_target_replay(target) != excision_target_replay(frozen_target):
        raise ReferenceSealError("User receipt coordinates differ from its retained begun Excision preview")
    if receipt.session_id != target.session_id or not target.found or receipt.excised_at_ms is None:
        raise ReferenceSealError("User receipt differs from its frozen Excision target")
    timestamp = receipt.excised_at_ms
    receipt_id = _receipt_assertion_id(target.session_id, timestamp)

    # Page physical identities before retaining complete native cells. The
    # same pinned original snapshot supplies both membership and payload.
    for ref in _target_refs(target):
        after: int | None = None
        while True:
            with seal.original_rows(
                "user",
                "SELECT rowid FROM assertions WHERE target_ref=? AND (? IS NULL OR rowid>?) ORDER BY rowid LIMIT 256",
                (ref, after, after),
            ) as rows:
                page = tuple(int(row[0]) for row in rows)
            if not page:
                break
            for rowid in page:
                image = seal.retain_tier_row("user", "assertions", rowid)
                if image is None:
                    raise ReferenceSealError("original Excision assertion disappeared inside its pinned view")
                seal.load_user_row(image)
            after = page[-1]

    def writable_keys(
        predicate: str, parameters: tuple[object, ...]
    ) -> Iterator[tuple[str, tuple[KnownTierCell, ...]]]:
        after: int | None = None
        while True:
            with seal.user_rows(
                f"SELECT rowid FROM assertions WHERE {predicate} AND (? IS NULL OR rowid>?) ORDER BY rowid LIMIT 256",
                (*parameters, after, after),
            ) as rows:
                page = tuple(int(row[0]) for row in rows)
            if not page:
                return
            for rowid in page:
                image = seal.retain_user_row("assertions", rowid)
                if image is None:
                    raise ReferenceSealError("selected Excision assertion disappeared before its declared effect")
                cells = dict(zip(image.columns, image.cells, strict=True))
                yield "assertions", (cells["assertion_id"],)
            after = page[-1]

    removed = tombstoned = 0
    for ref in _target_refs(target):
        ref_cell = seal.retain_literal_scalar(ref)
        expression, parameters = seal.source_literal_expression(ref_cell)
        predicate = f"target_ref={expression} AND assertion_id LIKE 'marker-%'"

        # Resolve only original anchors, then retain standalone exact SQL.
        # Replay cannot borrow the preparation connection's TEMP relations.
        for _table, marker_key in writable_keys(predicate, parameters):
            if compute_cancel_requested():
                raise asyncio.CancelledError("marker tombstone preparation cancelled by its owner")
            marker_id = marker_key[0]
            marker_expression, marker_parameters = seal.source_literal_expression(marker_id)
            with seal.user_rows(
                f"SELECT rowid FROM assertions WHERE assertion_id={marker_expression}", marker_parameters
            ) as rows:
                marker_rowid = int(rows.fetchone()[0])
            marker_image = seal.retain_user_row("assertions", marker_rowid)
            if marker_image is None:
                raise ReferenceSealError("original marker disappeared before its exact tombstone")
            assignments = ""
            if seal.marker_reference_in_excision(marker_image, "author_ref"):
                # Self on a deleted tombstone means retired provenance, not
                # self-authorship or default operator authorship.
                assignments += "author_ref='assertion:' || assertion_id, "
            if seal.marker_reference_in_excision(marker_image, "scope_ref"):
                assignments += "scope_ref=NULL, "
            with seal.user_statement(
                "UPDATE assertions SET target_ref='assertion:' || assertion_id, value_json='{}', body_text=NULL, "
                + assignments
                + "evidence_refs_json='[]', status='deleted', updated_at_ms=? WHERE assertion_id="
                + marker_expression,
                (timestamp, *marker_parameters),
                table="assertions",
                writable_targets=(("assertions", marker_key),),
            ) as cursor:
                tombstoned += max(cursor.rowcount, 0)
        predicate = f"target_ref={expression} AND kind NOT IN ('suppression','excision_record','excision_request')"
        with seal.user_statement(
            "DELETE FROM assertions WHERE " + predicate,
            parameters,
            table="assertions",
            writable_targets=writable_keys(predicate, parameters),
        ) as cursor:
            removed += max(cursor.rowcount, 0)

    source_counts = (
        seal.excision_source_target_counts(target.session_id) if seal._excision_source_completion_staged else {}
    )
    counts = {
        **receipt.counts,
        **source_counts,
        "user_assertions_removed": removed,
        "user_assertions_tombstoned": tombstoned,
    }
    scalar_fields = {
        "reason": receipt.reason,
        "actor": receipt.actor,
        "mode": "standalone",
        "operation_id": operation_id,
        "attempt_id": attempt_id,
        "plan_hash": plan_hash,
        "counts": counts,
        "excised_at_ms": timestamp,
    }

    def array_items(key: str) -> Generator[str, None, None]:
        if key == "marker_input_digests":
            yield from receipt.marker_input_digests
        elif seal._excision_source_completion_staged:
            with closing(
                seal.excision_source_target_hashes(target.session_id, removed=key == "removed_blob_hashes")
            ) as hashes:
                for blob_hash in hashes:
                    yield blob_hash.hex()
        else:
            yield from receipt.removed_blob_hashes if key == "removed_blob_hashes" else receipt.shared_blob_hashes

    def value_chunks() -> Generator[bytes, None, None]:
        # This is the same sorted, compact JSON value as prepare_assertion_row;
        # only the hash/marker arrays stay on their original native iterator.
        yield b"{"
        for position, key in enumerate(
            sorted((*scalar_fields, "removed_blob_hashes", "shared_blob_hashes", "marker_input_digests"))
        ):
            if position:
                yield b","
            yield json.dumps(key, ensure_ascii=False).encode("utf-8") + b":"
            if key in scalar_fields:
                yield json.dumps(scalar_fields[key], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
                continue
            yield b"["
            with closing(array_items(key)) as items:
                for ordinal, item in enumerate(items):
                    if ordinal:
                        yield b","
                    yield json.dumps(item, ensure_ascii=False).encode("utf-8")
            yield b"]"
        yield b"}"

    from polylogue.storage.sqlite.literal_cells import owned_literal_stream

    value_length = 0
    with owned_literal_stream(value_chunks()) as chunks:
        for chunk in chunks:
            value_length += len(chunk)
    with owned_literal_stream(value_chunks()) as chunks:
        value_cell = seal.retain_literal_stream("text", value_length, chunks)
    values = prepare_assertion_row(
        seal.observer("user"),
        assertion_id=receipt_id,
        target_ref=f"session:{target.session_id}",
        kind=AssertionKind.EXCISION_RECORD,
        value={},
        author_ref="user:local",
        author_kind="user",
        status=AssertionStatus.ACTIVE,
        visibility=AssertionVisibility.PRIVATE,
        context_policy={"inject": False},
        now_ms=timestamp,
    )
    columns, _keys = seal._known_tier_table_shape("user", "assertions")
    cells = {column: seal.retain_literal_scalar(value) for column, value in zip(columns, values, strict=True)}
    cells["value_json"] = value_cell
    expressions = ["?"]
    bindings: list[object] = [None]
    for column in columns:
        expression, parameters = seal.source_literal_expression(cells[column])
        expressions.append(expression)
        bindings.extend(parameters)
    with seal.user_statement(
        assertion_upsert_statement(tuple(expressions)),
        tuple(bindings),
        table="assertions",
        writable_targets=(("assertions", (cells["assertion_id"],)),),
        prepared_cells=cells,
        allocation_parameter=0,
    ):
        pass
    return replace(receipt, counts=counts, receipt_assertion_id=receipt_id)


class _BoundExcisionSessions(Collection[str]):
    """Borrow the exact authenticated plan's existing target population."""

    def __init__(self, plan: MutationPlan) -> None:
        self._plan = plan

    def __len__(self) -> int:
        return len(self._plan.target_refs)

    def __iter__(self) -> Iterator[str]:
        from polylogue.storage.sqlite.reference_seal import ReferenceSealError

        for ref in self._plan.target_refs:
            if not ref.startswith("session:"):
                raise ReferenceSealError("begun Excision contains a non-session target")
            yield ref.removeprefix("session:")

    def __contains__(self, session_id: object) -> bool:
        return isinstance(session_id, str) and f"session:{session_id}" in self._plan.target_refs


def _deliver_started_no_effect_excision(
    started: StartedBoundMutation, args: SessionExcisionArgs
) -> Mapping[str, object]:
    """Deliver the full canonical empty product without asserting a paid commit."""
    import asyncio

    from polylogue.storage.sqlite.audit_continuity import CanonicalAuditLiteral
    from polylogue.storage.sqlite.literal_cells import canonical_json_text_chunks, owned_literal_stream
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    if (
        started.operation_id is None
        or started.plan.operation != "mutate-session-excision"
        or started.plan.context.get("found") is not False
        or args.session_id != started.plan.context["session_id"]
        or args.reason != started.plan.context["reason"]
        or args.actor != started.plan.context["actor"]
        or args.cascade_lineage != started.plan.context["cascade_lineage"]
    ):
        raise ReferenceSealError("no-effect Excision differs from its original Started evidence")
    if args.result_sink is None:
        raise ReferenceSealError("Excision requires its original result delivery owner")
    receipt = ExcisionReceipt(args.session_id, False, reason=args.reason, actor=args.actor)
    fields = receipt.as_dict()
    array_fields = (
        "removed_blob_hashes",
        "shared_blob_hashes",
        "marker_input_digests",
        "cascaded_session_ids",
        "retained_hook_events",
        "retained_source_containers",
    )
    summary = {key: value for key, value in fields.items() if key not in array_fields}
    summary.update((field + "_count", 0) for field in array_fields)
    active = True

    def chunks() -> Generator[bytes, None, None]:
        if not active:
            raise ReferenceSealError("no-effect Excision delivery owner has retired")
        yield b"{"
        for ordinal, key in enumerate(sorted(fields)):
            if compute_cancel_requested():
                raise asyncio.CancelledError("no-effect Excision delivery cancelled")
            if ordinal:
                yield b","
            yield json.dumps(key).encode("utf-8") + b":"
            value = fields[key]
            if isinstance(value, str):

                def text_chunks(value: str = value) -> Generator[bytes, None, None]:
                    for offset in range(0, len(value), 4096):
                        if compute_cancel_requested():
                            raise asyncio.CancelledError("no-effect Excision token cancelled")
                        yield value[offset : offset + 4096].encode("utf-8")

                yield from canonical_json_text_chunks(text_chunks())
            else:
                yield json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        yield b"}"
        if not active:
            raise ReferenceSealError("no-effect Excision delivery owner has retired")

    digest = hashlib.sha256()
    byte_length = 0
    with owned_literal_stream(chunks()) as stream:
        for chunk in stream:
            digest.update(chunk)
            byte_length += len(chunk)
    try:
        args.result_sink(summary, CanonicalAuditLiteral(byte_length, digest.hexdigest(), chunks))
    finally:
        active = False
    return summary


def _apply_started_session_excision(
    started: StartedBoundMutation, args: SessionExcisionArgs, *, actuator: object
) -> Mapping[str, object]:
    """Use the executor's actual Started carrier, without reconstructing it."""
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    if started.operation_id is None:
        raise ReferenceSealError("Excision requires its exact durable Started operation")
    return _apply_original_session_excision(started.plan, started.operation_id, args, actuator=actuator)


def _apply_original_session_excision(
    plan: MutationPlan,
    operation_id: str,
    args: SessionExcisionArgs,
    *,
    actuator: object,
    recovery: RecoveryOperation | None = None,
) -> Mapping[str, object]:
    """Prepare and publish one whole closure from the executor's actual begin."""
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.operations.audit import AuditRepository
    from polylogue.operations.mutation_transaction import _authorized_removal_apply
    from polylogue.storage.archive_identity import resolve_active_index_path
    from polylogue.storage.blob_publication import exclude_archive_blob_publishers
    from polylogue.storage.sqlite.archive_tiers.archive import stage_index_session_deletions
    from polylogue.storage.sqlite.connection_profile import (
        NativeSQLCustodyOwner,
        _close_failed_native_construction,
    )
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealError

    if plan.operation != "mutate-session-excision":
        raise ReferenceSealError("Excision requires the exact durable Started carrier")
    if (
        args.session_id != plan.context["session_id"]
        or args.reason != plan.context["reason"]
        or args.actor != plan.context["actor"]
        or args.cascade_lineage != plan.context["cascade_lineage"]
    ):
        raise ReferenceSealError("Excision arguments differ from their exact Started plan")
    sink_failure: BaseException | None = None
    session_ids = _BoundExcisionSessions(plan)
    if args.session_id not in session_ids:
        raise ReferenceSealError("Excision primary is outside its actual Started closure")
    timestamp = int(datetime.fromisoformat(plan.prepared_at).timestamp() * 1000)
    primary: ExcisionReceipt | None = None
    total_counts: dict[str, int] = {}
    index_path = resolve_active_index_path(args.archive_root)
    # Publisher exclusion covers both original preparation and all physical
    # publication. Targets come only from the retained begun preview.
    with exclude_archive_blob_publishers(args.archive_root / "source.db"):
        with PreparedIndexMutation.for_excision(
            index_path,
            archive_root=args.archive_root,
            input_demand=args.input_demand,
        ) as seal:
            if recovery is not None and (
                recovery.operation_id != operation_id
                or recovery.operation != plan.operation
                or recovery.operation_version != plan.operation_version
                or recovery.plan_hash != plan.plan_hash
                or recovery.target_digest != plan.target_digest
                or recovery.attempt_id is None
                or not recovery.target_evidence_complete
                or recovery.expected_target_count != len(plan.target_refs)
                or recovery.reconstructed_target_count != len(plan.target_refs)
                or tuple(target.ref for target in recovery.targets) != plan.target_refs
            ):
                raise ReferenceSealError("Excision recovery differs from its recorded original operation and plan")
            attempt_id = seal.bind_begun_excision(
                operation_id,
                plan.plan_hash,
                session_ids,
                recovery_attempt_id=None if recovery is None else recovery.attempt_id,
            )
            source_completed = False
            with seal.original_read_snapshot():
                _verify_original_excision_index_markers(seal)
                event = seal.original_excision_source_completion() if recovery is not None else None
                if event is None:
                    if seal._excision_recovery_user_committed:
                        raise ReferenceSealError("User completion lacks its exact original Source completion")
                    seal.prepare_excision_embeddings_intent()
                    with seal.source_producer():
                        _stage_excision_source_closure(
                            seal,
                            reason=args.reason,
                            actor=args.actor,
                            excised_at_ms=timestamp,
                        )
                else:
                    if event[1] != timestamp:
                        raise ReferenceSealError("Source event time differs from its original frozen operation")
                    seal.restore_excision_source_completion(event[0], occurred_at_ms=event[1])
                    for session_id in session_ids:
                        _verify_excision_source_target_terminal(
                            seal, seal.original_excision_target(session_id), recovered=True
                        )
                    source_completed = True
            source_permit = None if source_completed else seal.prepare_source_mutation()
            embeddings = seal.prepare_excision_embeddings_child()
            if source_completed:
                with seal.original_read_snapshot():
                    paid_completed = embeddings.recover_committed()
                    if not paid_completed:
                        seal.enroll_restored_excision_embeddings_inputs()
                if seal._excision_recovery_user_committed and not paid_completed:
                    raise ReferenceSealError("User completion cannot substitute for a missing atomic paid completion")
            if not embeddings._completed and "embeddings" in seal._capabilities:
                embeddings._verify_inputs(seal.observer("embeddings"))
                seal.validate_observers_current()
            with seal.original_read_snapshot(), ExitStack() as preparation:
                if not seal._excision_recovery_user_committed:
                    preparation.enter_context(seal.user_producer())
                for session_id in _PreparedExcisionSessionClosure(seal):
                    target = seal.original_excision_target(session_id)
                    candidate = ExcisionReceipt(
                        session_id=session_id,
                        found=True,
                        reason=args.reason,
                        actor=args.actor,
                        excised_at_ms=timestamp,
                        counts={
                            **seal.excision_embeddings_expected_counts(session_id),
                            "index_sessions": int(target.session_exists),
                            "index_messages": len(target.message_ids),
                            "index_blocks": len(target.block_ids),
                            "index_marker_witnesses": len(target.index_marker_witnesses),
                        },
                        marker_input_digests=tuple(
                            dict.fromkeys(marker.carrier_digest for marker in target.marker_input_targets)
                        ),
                        retained_hook_events=(),
                        retained_source_containers=tuple(item.label for item in target.containers.retained_items),
                    )
                    if seal._excision_recovery_user_committed:
                        staged = replace(
                            candidate,
                            counts=seal.original_excision_user_completion_counts(session_id),
                            receipt_assertion_id=_receipt_assertion_id(session_id, timestamp),
                        )
                    else:
                        staged = _stage_excision_user_receipt(
                            seal,
                            target,
                            candidate,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            plan_hash=plan.plan_hash,
                        )
                    for key, value in staged.counts.items():
                        total_counts[key] = total_counts.get(key, 0) + value
                    if session_id == args.session_id:
                        primary = staged
            user_permit = None if seal._excision_recovery_user_committed else seal.prepare_user_mutation()

            if primary is None:
                raise ReferenceSealError("prepared Excision lost its original primary receipt")
            # Result staging borrows this original witness until the existing
            # request owner has installed the complete canonical document.
            from polylogue.storage.sqlite.audit_continuity import CanonicalAuditLiteral
            from polylogue.storage.sqlite.literal_cells import canonical_json_text_chunks, owned_literal_stream
            from polylogue.storage.sqlite.reference_seal import KnownTierCell

            array_fields = (
                "cascaded_session_ids",
                "marker_input_digests",
                "removed_blob_hashes",
                "retained_hook_events",
                "retained_source_containers",
                "shared_blob_hashes",
            )
            with seal._owned_cursor(
                seal._scratch,
                "CREATE TEMP TABLE excision_product_items ("
                "ordinal INTEGER PRIMARY KEY, field TEXT NOT NULL, value TEXT NOT NULL, "
                "cell_id INTEGER NOT NULL, UNIQUE(field,value))",
            ):
                pass

            def stage_item(field: str, value: str) -> None:
                with seal._owned_cursor(
                    seal._scratch, "SELECT 1 FROM temp.excision_product_items WHERE field=? AND value=?", (field, value)
                ) as existing:
                    if existing.fetchone() is not None:
                        return
                cell = seal.retain_literal_scalar(value)
                with seal._owned_cursor(
                    seal._scratch,
                    "INSERT INTO temp.excision_product_items(field,value,cell_id) VALUES(?,?,?)",
                    (field, value, cell._cell_id),
                ):
                    pass

            with seal.original_read_snapshot():
                for session_id in _PreparedExcisionSessionClosure(seal):
                    target = seal.original_excision_target(session_id)
                    if session_id != args.session_id:
                        stage_item("cascaded_session_ids", session_id)
                    for marker in target.marker_input_targets:
                        stage_item("marker_input_digests", marker.carrier_digest)
                    for item in target.containers.retained_items:
                        stage_item("retained_source_containers", item.label)
                    for is_removed, field in ((True, "removed_blob_hashes"), (False, "shared_blob_hashes")):
                        with closing(seal.excision_source_target_hashes(session_id, removed=is_removed)) as hashes:
                            for blob_hash in hashes:
                                stage_item(field, blob_hash.hex())
            summary: dict[str, object] = {
                "session_id": primary.session_id,
                "found": primary.found,
                "reason": primary.reason,
                "actor": primary.actor,
                "excised_at_ms": primary.excised_at_ms,
                "receipt_assertion_id": primary.receipt_assertion_id,
                "counts": total_counts,
            }
            for field in array_fields:
                with seal._owned_cursor(
                    seal._scratch, "SELECT count(*) FROM temp.excision_product_items WHERE field=?", (field,)
                ) as count:
                    summary[field + "_count"] = count.fetchone()[0]
            summary["complete"] = not (
                summary["retained_hook_events_count"] or summary["retained_source_containers_count"]
            )
            scalar_cells = {
                key: seal.retain_literal_scalar(cast("str | None", value))
                for key, value in summary.items()
                if key in {"session_id", "reason", "actor", "receipt_assertion_id"}
            }

            def product_chunks() -> Generator[bytes, None, None]:
                seal._require_new_work()
                yield b"{"
                fields = sorted((set(summary) - {field + "_count" for field in array_fields}) | set(array_fields))
                for ordinal, key in enumerate(fields):
                    if ordinal:
                        yield b","
                    yield json.dumps(key, ensure_ascii=False).encode("utf-8") + b":"
                    if key in array_fields:
                        yield b"["
                        with seal._owned_cursor(
                            seal._scratch,
                            "SELECT cell_id FROM temp.excision_product_items WHERE field=? ORDER BY ordinal",
                            (key,),
                        ) as cells:
                            for index, (cell_id,) in enumerate(cells):
                                if index:
                                    yield b","
                                yield from canonical_json_text_chunks(
                                    seal._literal_cell_chunks(KnownTierCell(seal, cell_id))
                                )
                        yield b"]"
                    elif key in scalar_cells and summary[key] is not None:
                        yield from canonical_json_text_chunks(seal._literal_cell_chunks(scalar_cells[key]))
                    else:
                        yield json.dumps(
                            summary[key], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                yield b"}"

            product_length = 0
            product_digest = hashlib.sha256()
            with owned_literal_stream(product_chunks()) as chunks:
                for chunk in chunks:
                    product_length += len(chunk)
                    product_digest.update(chunk)
            product_cell = seal.retain_literal_stream("text", product_length, product_chunks())

            def retained_product_chunks() -> Generator[bytes, None, None]:
                seal._require_new_work()
                yield from seal._literal_cell_chunks(product_cell)
                seal._require_new_work()

            product = CanonicalAuditLiteral(product_length, product_digest.hexdigest(), retained_product_chunks)
            if args.result_sink is None:
                raise ReferenceSealError("Excision requires its original result delivery owner")

            def publish() -> None:
                with _authorized_removal_apply(plan, args.archive_root, actuator, args):
                    index = _connect_rw(index_path, archive_root=args.archive_root, foreign_keys=True)
                    owner = NativeSQLCustodyOwner(index, terminal_parent=seal)
                    try:
                        with seal.mutation_scope(index) as scope:
                            for session_id in _PreparedExcisionSessionClosure(seal):
                                scope.authorize_session_removal((session_id,))
                                with seal.original_read_snapshot():
                                    target = seal.original_excision_target(session_id)
                                    if seal._excision_recovery_user_committed:
                                        with seal.original_rows(
                                            "index", "SELECT 1 FROM sessions WHERE session_id=?", (session_id,)
                                        ) as rows:
                                            expected = rows.fetchone() is not None
                                    else:
                                        expected = target.session_exists
                                for witness in target.index_marker_witnesses:
                                    with seal._owned_cursor(
                                        seal._scratch,
                                        "SELECT present FROM temp.excision_index_marker_current WHERE request_key=?",
                                        (witness.request_key,),
                                    ) as rows:
                                        marker_present = rows.fetchone()
                                    if marker_present is None:
                                        raise ReferenceSealError(
                                            "Index marker deletion lacks its exact original postimage input"
                                        )
                                    with seal._owned_cursor(
                                        index,
                                        "DELETE FROM ingest_marker_witnesses WHERE request_key=?",
                                        (witness.request_key,),
                                    ) as cursor:
                                        if cursor.rowcount != marker_present[0]:
                                            raise ReferenceSealError(
                                                "Index marker deletion differs from its original witness"
                                            )
                                deleted = stage_index_session_deletions(index, scope, (session_id,))
                                if len(deleted) != int(expected):
                                    raise ReferenceSealError("Index deletion differs from original frozen membership")
                            scope.preflight_reachability()
                            # User's actual guarded DML is staged, then held
                            # uncommitted through Source and Embeddings. Any
                            # User byte/effect refusal precedes durable effects.
                            with ExitStack() as user_lifetime:
                                # One physical User transaction stays alive;
                                # only the custody's transient SQL authority
                                # changes between already prepared tiers.
                                user = None
                                if user_permit is not None:
                                    with user_permit.hold_authority():
                                        user = user_lifetime.enter_context(user_permit.mutation_connection())
                                        with seal._owned_cursor(user, "BEGIN IMMEDIATE"):
                                            pass
                                        user_permit.apply_user_statements(user)
                                # Only TEMP effect bookkeeping changed while the
                                # retained User attachment froze witness MAIN.
                                if user_permit is not None:
                                    seal._settle_frozen_literal_bookkeeping()
                                if source_permit is not None:
                                    with source_permit.hold_authority(), source_permit.mutation_connection() as source:
                                        with seal._owned_cursor(source, "BEGIN IMMEDIATE"):
                                            pass
                                        source_permit.apply_source_statements(source)
                                        source_permit.allow_commit(source)
                                        source.commit()
                                        seal.accept_known_tier_commit(source_permit.committed())
                                if not embeddings._completed:
                                    embeddings.apply()
                                for session_id in _PreparedExcisionSessionClosure(seal):
                                    if embeddings.completed_counts(
                                        session_id
                                    ) != seal.excision_embeddings_expected_counts(session_id):
                                        raise ReferenceSealError(
                                            "User receipt omitted physically completed Embeddings effects"
                                        )
                                if user_permit is not None and user is not None:
                                    with user_permit.hold_authority():
                                        seal.validate_observers_current()
                                        user_permit.allow_commit(user)
                                        user.commit()
                                        seal.accept_known_tier_commit(user_permit.committed())
                            scope.commit()
                    except BaseException as failure:
                        _close_failed_native_construction(owner, failure)
                        raise
                    else:
                        owner.close()

            admit_stage_write("operation.session-excision.publish", publish)
            try:
                args.result_sink(summary, product)
            except BaseException as failure:
                # Effects are physically settled. Delivery failure cannot
                # undo them or suppress their canonical completion event.
                sink_failure = failure
        # Projection changes Audit currency, so it follows retirement of the
        # original witness. Recovery reads this same durable command/event.
        try:
            admit_stage_write(
                "operation.session-excision.continuity",
                lambda: AuditRepository.for_archive_root(args.archive_root).reconcile_continuity(),
            )
        except BaseException as failure:
            if sink_failure is not None:
                raise BaseExceptionGroup(
                    "Excision delivery and continuity failed", [sink_failure, failure]
                ) from sink_failure
            raise
        if sink_failure is not None:
            raise sink_failure
    return summary


__all__ = [
    "ExcisionBlobReferenceUnknownError",
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
    "build_excision_policy_snapshot",
    "find_lineage_dependents",
    "plan_session_excision",
    "resolve_session_excision_target",
]
