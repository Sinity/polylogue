"""Reconstruct a fingerprint-stale managed Index from its retained Source."""

from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, closing
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.errors import SchemaSkew
from polylogue.core.stage_admission import stage_write_admission
from polylogue.logging import emit
from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation
from polylogue.operations.raw_observation_owner import RawObservationArchiveWork
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation, TierFileIdentity
from polylogue.storage.index_generation import (
    IndexGeneration,
    IndexGenerationStore,
    canonical_active_index_path,
    rebuild_source_acquisition_snapshot,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.schema_identity import (
    DerivedTier,
    derived_schema_identity,
    read_schema_identity,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import locate_composed_message
from polylogue.storage.sqlite.audit_leaf import VerifiedAuditLeaf
from polylogue.storage.sqlite.connection_profile import assert_tier_schema_supported, open_readonly_connection
from polylogue.storage.sqlite.reference_seal import IndexMutationDestination, PreparedIndexMutation
from polylogue.storage.sqlite.schema_manifest import SchemaManifest, canonical_schema_manifest, schema_manifest_diff

if TYPE_CHECKING:
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.operations.raw_observation_owner import RawObservationReplacement

T = TypeVar("T")
_OWNER = "daemon:retained-index-startup"
_RAW_PAGE = 128


class IndexReconvergenceRefusedError(ValueError):
    """Retained evidence cannot safely replace the observed Index."""

    code = "index_reconvergence_refused"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _require_shape(conn: sqlite3.Connection, tier: ArchiveTier) -> None:
    expected = canonical_schema_manifest(tier)
    actual = SchemaManifest.from_connection(conn, tier)
    if actual.version != expected.version or any(schema_manifest_diff(expected, actual).values()):
        raise IndexReconvergenceRefusedError(f"unprovable_{tier.value}_shape")


def _require_empty_bootstrap_predecessor(conn: sqlite3.Connection, *, owner_id: str) -> None:
    """Admit an owned fresh-format empty generation without reading old material.

    Derived DDL may change between releases. Only the earlier bootstrap
    owner's completely empty material can cross that boundary; metadata
    controls retain their canonical shape and every unknown table is checked.
    """
    expected = canonical_schema_manifest(ArchiveTier.INDEX)
    actual = SchemaManifest.from_connection(conn, ArchiveTier.INDEX)
    if owner_id != "daemon:empty-index-startup" or actual.version != expected.version:
        raise IndexReconvergenceRefusedError("unprovable_index_shape")
    controls = {"raw_existence_journal_control", "session_profile_demand_state", "query_unit_frame_state"}
    expected_controls = {obj for obj in expected.objects if obj[0] == "table" and obj[1] in controls}
    actual_controls = {obj for obj in actual.objects if obj[0] == "table" and obj[1] in controls}
    if actual_controls != expected_controls:
        raise IndexReconvergenceRefusedError("unprovable_index_shape")
    for schema, name, kind, *_ in conn.execute("PRAGMA table_list").fetchall():
        check_compute_cancelled()
        if schema != "main" or name.startswith("sqlite_") or name in controls | {"schema_identity"} or kind == "shadow":
            continue
        quoted = '"' + name.replace('"', '""') + '"'
        try:
            populated = conn.execute(f"SELECT 1 FROM {quoted} LIMIT 1").fetchone() is not None
        except sqlite3.DatabaseError as exc:
            raise IndexReconvergenceRefusedError("unprovable_index_population") from exc
        if populated:
            raise IndexReconvergenceRefusedError("nonempty_predecessor_index")


def reconverge_managed_index_on_startup(
    root: Path,
    *,
    archive_owner: OwnedArchiveLocation,
    compute_adapter: BoundedComputeAdapter,
    write_admission: Callable[[str], AbstractContextManager[object]],
    retained: list[RawObservationReplacement],
    run_replay: Callable[[Callable[[], object], int], object],
) -> str | None:
    """Prepare off-gate, replay retained bytes, then publish under the writer.

    No current-schema writer opens the predecessor. Completed durable pages
    resume from the same candidate; an interrupted page replays idempotently.
    """
    from polylogue.maintenance.candidate_capacity import require_candidate_capacity
    from polylogue.operations.durable_change_train import assert_holds_archive_ownership
    from polylogue.operations.reset_safety import archive_tiers_are_closed

    assert_holds_archive_ownership(archive_owner, root)
    if not archive_tiers_are_closed(root):
        raise IndexReconvergenceRefusedError("startup_tiers_not_closed")
    if not (root / ".index-active-pointer").exists():
        return None
    location = ArchiveLocation.resolve(root)
    index = location.active_index_path.resolve(strict=True)
    generations = canonical_active_index_path(location).parent / ".index-generations"
    if index.parent.parent != generations.resolve() or index.name != "index.db":
        canonical = canonical_active_index_path(location)
        if not canonical.is_symlink() and canonical.resolve(strict=True) == index:
            return None  # The separate verified-empty bootstrap owner handles regular Index.
        raise IndexReconvergenceRefusedError("managed_index_escapes_archive")
    store = IndexGenerationStore(location, repair_anchor=False)
    parent = store.load(index.parent.name)
    if parent.state not in {"active", "promoting"} or Path(parent.index_path).resolve() != index:
        raise IndexReconvergenceRefusedError("unprovable_active_generation")
    recovering = None
    empty_predecessor = False
    with VerifiedAuditLeaf(index.parent, filename=index.name) as leaf:
        with closing(open_readonly_connection(leaf.anchored_path, validate_schema=False)) as conn:
            try:
                assert_tier_schema_supported(conn, index, ArchiveTier.INDEX)
            except SchemaSkew:
                pass
            else:
                active = store.load(index.parent.name)
                if active.owner_id == _OWNER and active.state == "active":
                    return None
                if active.owner_id != _OWNER or active.state != "promoting":
                    return None
                recovering = active
            if read_schema_identity(conn, DerivedTier.INDEX) is None:
                raise IndexReconvergenceRefusedError("missing_index_identity")
            try:
                _require_shape(conn, ArchiveTier.INDEX)
            except IndexReconvergenceRefusedError:
                _require_empty_bootstrap_predecessor(conn, owner_id=parent.owner_id)
                empty_predecessor = True
        leaf.assert_unchanged()
    if store.load(index.parent.name) != parent or (parent.state != "active" and parent != recovering):
        raise IndexReconvergenceRefusedError("unprovable_active_generation")

    with PreparedIndexMutation(index, archive_root=root) as original, ExitStack() as custody:
        with original.original_read_snapshot():
            if empty_predecessor:
                _require_empty_bootstrap_predecessor(original.observer("index"), owner_id=parent.owner_id)
            for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT):
                conn = original.observer(tier.value)
                try:
                    assert_tier_schema_supported(conn, root / f"{tier.value}.db", tier)
                except SchemaSkew as exc:
                    raise IndexReconvergenceRefusedError(f"unsupported_{tier.value}_schema") from exc
                _require_shape(conn, tier)
        binding = location.configured_tier("embeddings")
        entry = binding.configured_path.lstat()
        leaf = custody.enter_context(
            VerifiedAuditLeaf(binding.resolved_path.parent, filename=binding.resolved_path.name)
        )
        embeddings = custody.enter_context(closing(open_readonly_connection(leaf.anchored_path, validate_schema=False)))
        try:
            assert_tier_schema_supported(embeddings, binding.resolved_path, ArchiveTier.EMBEDDINGS)
        except SchemaSkew as exc:
            raise IndexReconvergenceRefusedError("unsupported_embeddings_schema") from exc
        _require_shape(embeddings, ArchiveTier.EMBEDDINGS)
        version = int(embeddings.execute("PRAGMA data_version").fetchone()[0])
        conserved_bindings = (
            *(location.configured_tier(name) for name in ("source", "user", "audit")),
            TierFileIdentity.resolve("index", index),
        )

        def require_current() -> None:
            check_compute_cancelled()
            for conserved in conserved_bindings:
                if TierFileIdentity.resolve(conserved.name, conserved.configured_path) != conserved:
                    raise IndexReconvergenceRefusedError(f"changed_{conserved.name}_binding")
            # Source interpretation legitimately commits through the Raw owner.
            # Its original binding stays pinned; the remaining tiers may not change.
            with original.verified_namespace():
                for name in ("index", "user", "audit"):
                    observer = original.observer(name)
                    if int(observer.execute("PRAGMA data_version").fetchone()[0]) != original.observer_version(name):
                        raise IndexReconvergenceRefusedError(f"changed_{name}_custody")
            leaf.assert_unchanged()
            current = binding.configured_path.lstat()
            if TierFileIdentity.resolve("embeddings", binding.configured_path) != binding or (
                current.st_dev,
                current.st_ino,
            ) != (entry.st_dev, entry.st_ino):
                raise IndexReconvergenceRefusedError("changed_embeddings_binding")
            if int(embeddings.execute("PRAGMA data_version").fetchone()[0]) != version:
                raise IndexReconvergenceRefusedError("changed_embeddings_custody")

        snapshot = rebuild_source_acquisition_snapshot(root)
        if recovering is not None:
            if snapshot != recovering.source_snapshot:
                raise IndexReconvergenceRefusedError("changed_source_evidence")
            original.validate_observers_current()
            with write_admission("daemon.index_reconvergence.recover_promotion"):
                original.validate_observers_current()
                require_current()
                store.complete_promotion_recovery(recovering.generation_id)
            return recovering.generation_id
        operation_id = str(uuid.uuid4())
        from polylogue.sources.origin_specs import parser_semantic_authority_fingerprint

        recipe = (
            derived_schema_identity(DerivedTier.INDEX),
            derived_schema_identity(DerivedTier.OPS),
            parser_semantic_authority_fingerprint(),
        )
        custody_digest = hashlib.sha256(
            json.dumps(
                {
                    "tiers": [item.as_dict() for item in (*conserved_bindings, binding)],
                    "predecessor_frame": [
                        tuple(row)
                        for row in original.observer("index").execute(
                            "SELECT * FROM query_unit_frame_state ORDER BY relation"
                        )
                    ],
                    "assertions_epoch": tuple(
                        original.observer("user")
                        .execute("SELECT epoch FROM query_unit_frame_state WHERE singleton=1")
                        .fetchone()
                    ),
                    "audit_head": tuple(
                        original.observer("audit")
                        .execute("SELECT generation,head_sha256 FROM audit_continuity_head WHERE singleton=1")
                        .fetchone()
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        def source_sequence() -> int:
            with closing(open_readonly_connection(root / "source.db")) as source:
                return int(
                    source.execute(
                        "SELECT MAX((SELECT COALESCE(MAX(sequence),0) FROM raw_existence_changes), retained_floor) "
                        "FROM raw_existence_journal_control WHERE singleton=1"
                    ).fetchone()[0]
                )

        def reusable(candidate: IndexGeneration) -> bool:
            identity = Path(candidate.index_path).stat()
            if (
                candidate.source_snapshot != snapshot
                or (
                    candidate.reconstruction_index_identity,
                    candidate.reconstruction_ops_identity,
                    candidate.reconstruction_parser_identity,
                )
                != recipe
                or candidate.reconstruction_predecessor_id != parent.generation_id
                or candidate.reconstruction_custody_digest != custody_digest
                or (candidate.reconstruction_device, candidate.reconstruction_inode)
                != (identity.st_dev, identity.st_ino)
            ):
                return False
            with closing(
                open_readonly_connection(Path(candidate.index_path), validate_schema=False)
            ) as candidate_reader:
                return read_schema_identity(candidate_reader, DerivedTier.INDEX) == recipe[0]

        def rewind_frontier(candidate: IndexGeneration) -> str:
            # An interrupted page can refine an earlier shared Raw before its
            # Index publication. Keep the candidate, but replay that key. Loss
            # of the journal or lost Source WAL commits requires a full scan into
            # the same file; an empty prefix owns no stale completion claim.
            with closing(open_readonly_connection(root / "source.db")) as source:
                floor = int(
                    source.execute(
                        "SELECT retained_floor FROM raw_existence_journal_control WHERE singleton=1"
                    ).fetchone()[0]
                )
                if (
                    floor > candidate.reconstruction_source_sequence
                    or source_sequence() < candidate.reconstruction_source_sequence
                ):
                    return ""
                changed = source.execute(
                    "SELECT MIN(raw_id) FROM raw_existence_changes WHERE sequence>? AND raw_id<=?",
                    (candidate.reconstruction_source_sequence, candidate.reconstruction_raw_id),
                ).fetchone()[0]
                if changed is None:
                    return candidate.reconstruction_raw_id
                predecessor = source.execute(
                    "SELECT raw_id FROM raw_sessions WHERE raw_id<? ORDER BY raw_id DESC LIMIT 1", (changed,)
                ).fetchone()
                return str(predecessor[0]) if predecessor is not None else ""

        require_current()
        original.validate_observers_current()
        with write_admission("daemon.index_reconvergence.create"):
            original.validate_observers_current()
            require_current()
            generation = None
            # Only this startup owner's never-published candidates are reclaimed.
            for path in sorted(store.generations_root.glob("gen-*/generation.json")):
                abandoned = store.load(path.parent.name)
                if abandoned.owner_id != _OWNER:
                    continue
                if abandoned.state == "promoting":
                    store.discard_unpublished_promotion(abandoned)
                elif abandoned.state == "inactive":
                    if generation is None and reusable(abandoned):
                        generation = abandoned
                    else:
                        store.discard_if_inactive(abandoned)
            if generation is None:
                require_candidate_capacity(root, operation_id=operation_id, baseline_digest=snapshot)
                generation = store.create(owner_id=_OWNER, source_snapshot=snapshot)
                generation = store.begin_reconstruction(
                    generation,
                    index_identity=recipe[0],
                    ops_identity=recipe[1],
                    parser_identity=recipe[2],
                    predecessor_id=parent.generation_id,
                    custody_digest=custody_digest,
                    source_sequence=source_sequence(),
                )
            else:
                require_candidate_capacity(
                    root,
                    operation_id=operation_id,
                    baseline_digest=snapshot,
                    existing_candidate_generation_id=generation.generation_id,
                )
                frontier = rewind_frontier(generation)
                if frontier < generation.reconstruction_raw_id or (
                    not frontier and source_sequence() < generation.reconstruction_source_sequence
                ):
                    generation = store.checkpoint_reconstruction(
                        generation,
                        raw_id=frontier,
                        source_sequence=source_sequence(),
                        rewind=True,
                    )
            # Establish the reader-index deferral before any retained
            # preparation seals this candidate's SQLite incarnation.
            with ArchiveStore.open_owned_inactive_generation(
                Path(generation.index_path).parent,
                generation_id=generation.generation_id,
                owner_id=generation.owner_id,
                defer_secondary_indexes=not generation.reconstruction_raw_id,
                preserve_secondary_index_layout=bool(generation.reconstruction_raw_id),
            ):
                pass
        emit(
            "daemon.index_reconvergence.startup",
            outcome="degraded",
            state="building",
            generation_id=generation.generation_id,
        )
        work = RawObservationArchiveWork(root, compute_adapter=compute_adapter)

        def admit(actor: str, function: Callable[[], T]) -> T:
            with write_admission(actor):
                return function()

        frontier = generation.reconstruction_raw_id
        # The controller alone owns the original observers and promotion seal.
        # Page payloads live on disk; only a shared-admission window is offered.
        width = compute_adapter.snapshot().by_class("incremental-background").ceiling_slots
        with stage_write_admission(admit):
            while True:
                require_current()
                with closing(open_readonly_connection(root / "source.db")) as source:
                    rows = source.execute(
                        "SELECT raw_id FROM raw_sessions WHERE raw_id>? ORDER BY raw_id LIMIT ?",
                        (frontier, min(_RAW_PAGE, width)),
                    ).fetchall()
                if not rows:
                    break
                selected = tuple(str(row[0]) for row in rows)

                async def prepare_page(selected: tuple[str, ...], generation: IndexGeneration) -> None:
                    adapter = make_raw_observation_derivation(
                        root,
                        compute_adapter=compute_adapter,
                        index_db_path=Path(generation.index_path),
                        owned_generation=generation,
                    )
                    destination = IndexMutationDestination.owned_inactive(generation)
                    scope = pickle.dumps(selected, protocol=pickle.HIGHEST_PROTOCOL)
                    selection = work.sidecar_owner_selector(selected)
                    async with work.prepared_neutral_page(
                        selected,
                        destination=lambda: (adapter, Path(generation.index_path), destination),
                        require_authority=work.require_source_frontier_authority,
                        selection=selection,
                    ) as page:
                        replay = work.retained_replay_operation(
                            scope,
                            retained=retained,
                            destination=lambda: (adapter, Path(generation.index_path), destination),
                            require_authority=work.require_source_frontier_authority,
                            select_retained_raw_ids=selection,
                            on_terminal_refusal=None,
                            on_dependency_refusal=None,
                            on_membership_refusal=None,
                            before_publication=None,
                            neutral_page=page,
                        )
                        # The daemon retains each physical publisher creator.
                        # No original observer is passed into this operation.
                        run_replay(lambda: replay().outcome.require_complete(), len(scope))

                asyncio.run(prepare_page(selected, generation))
                require_current()
                with write_admission("daemon.index_reconvergence.checkpoint"):
                    require_current()
                    generation = store.checkpoint_reconstruction(
                        generation,
                        raw_id=str(rows[-1][0]),
                        source_sequence=source_sequence(),
                    )
                frontier = generation.reconstruction_raw_id
                emit(
                    "daemon.index_reconvergence.progress",
                    outcome="degraded",
                    rows=len(rows),
                    generation_id=generation.generation_id,
                )
        with (
            write_admission("daemon.index_reconvergence.readiness"),
            ArchiveStore.open_owned_inactive_generation(
                Path(generation.index_path).parent,
                generation_id=generation.generation_id,
                owner_id=generation.owner_id,
                preserve_secondary_index_layout=True,
            ) as candidate,
        ):
            candidate.run_generation_readiness_pass()
        require_current()
        if rebuild_source_acquisition_snapshot(root) != snapshot:
            raise IndexReconvergenceRefusedError("changed_source_evidence")
        with store.prepare_promotion(generation) as prepared:
            if rebuild_source_acquisition_snapshot(root) != snapshot:
                raise IndexReconvergenceRefusedError("changed_source_evidence")
            prepared.reference_seal.validate_observers_current()
            if prepared.missing_session_count:
                raise IndexReconvergenceRefusedError("active_reference_coverage_incomplete")
            with (
                prepared.reference_seal.original_read_snapshot(),
                closing(open_readonly_connection(Path(generation.index_path))) as candidate_reader,
            ):
                for row in embeddings.execute("SELECT session_id,message_id FROM message_embedding_refs"):
                    session_id, message_id = str(row[0]), str(row[1])
                    was_present = locate_composed_message(
                        prepared.reference_seal.observer("index"), session_id, message_id
                    )
                    if (
                        was_present is not None
                        and locate_composed_message(candidate_reader, session_id, message_id) is None
                    ):
                        raise IndexReconvergenceRefusedError("purchased_reference_coverage_incomplete")
            with write_admission("daemon.index_reconvergence.promote"):
                require_current()
                promoted = store.promote(generation, prepared)
        emit("daemon.index_reconvergence.startup", outcome="ok", state="complete", generation_id=promoted.generation_id)
        return promoted.generation_id
