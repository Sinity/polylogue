"""Admit current derived schema after an empty managed build was interrupted."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, closing
from pathlib import Path

from polylogue.core.compute_cancel import compute_cancel_requested
from polylogue.core.errors import SchemaSkew
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation, TierFileIdentity
from polylogue.storage.index_generation import (
    IndexGenerationStore,
    canonical_active_index_path,
    rebuild_source_evidence_snapshot,
)
from polylogue.storage.sqlite.archive_tiers.schema_identity import DerivedTier, read_schema_identity
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.audit_leaf import VerifiedAuditLeaf
from polylogue.storage.sqlite.connection_profile import assert_tier_schema_supported, open_readonly_connection
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
from polylogue.storage.sqlite.schema_manifest import SchemaManifest, canonical_schema_manifest, schema_manifest_diff


class EmptyIndexTransitionRefusedError(ValueError):
    """The observed archive cannot prove an empty-generation transition safe."""

    code = "empty_index_transition_refused"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _checkpoint() -> None:
    if compute_cancel_requested():
        raise asyncio.CancelledError("empty-generation preparation cancelled")


def _require_matching_shape(conn: sqlite3.Connection, tier: ArchiveTier) -> None:
    canonical = canonical_schema_manifest(tier)
    observed = SchemaManifest.from_connection(conn, tier)
    if observed.version != canonical.version or any(schema_manifest_diff(canonical, observed).values()):
        raise EmptyIndexTransitionRefusedError(f"unprovable_{tier.value}_shape")


def _require_empty_population(seal: PreparedIndexMutation, tier: str) -> None:
    # These rows describe format or change-journal state, never acquired or
    # parsed material. All other physical tables, including pending custody
    # and unknown tables, must be empty. Virtual shadow tables belong to their
    # checked parent rather than representing independent material.
    controls = {"raw_existence_journal_control", "schema_identity", "sqlite_sequence"}
    if tier == "source":
        controls.add("audit_continuity_control")
    else:
        controls.update({"session_profile_demand_state", "query_unit_frame_state"})
    with seal.original_rows(tier, "PRAGMA table_list") as rows:
        tables = [(str(row[1]), str(row[2])) for row in rows if row[0] == "main"]
    for name, kind in tables:
        _checkpoint()
        if name.startswith("sqlite_") or name in controls or kind == "shadow":
            continue
        quoted = '"' + name.replace('"', '""') + '"'
        with seal.original_rows(tier, f"SELECT 1 FROM {quoted} LIMIT 1") as rows:
            if rows.fetchone() is not None:
                raise EmptyIndexTransitionRefusedError(f"nonempty_{tier}_population")
    if tier == "source":
        with seal.original_rows(tier, "SELECT pending_mutation_id FROM audit_continuity_control") as rows:
            control = rows.fetchone()
            if control is None or control[0] is not None:
                raise EmptyIndexTransitionRefusedError("unsettled_source_continuity")


def _require_empty_blob_custody(root: Path) -> None:
    blob = root / "blob"
    if not blob.exists():
        if blob.is_symlink():
            raise EmptyIndexTransitionRefusedError("unprovable_blob_custody")
        return
    if blob.is_symlink() or not blob.is_dir():
        raise EmptyIndexTransitionRefusedError("unprovable_blob_custody")
    pending = [os.scandir(blob)]
    try:
        while pending:
            _checkpoint()
            child = next(pending[-1], None)
            if child is None:
                pending.pop().close()
            elif child.is_dir(follow_symlinks=False):
                pending.append(os.scandir(child.path))
            else:
                raise EmptyIndexTransitionRefusedError("nonempty_blob_custody")
    finally:
        for iterator in reversed(pending):
            iterator.close()


def replace_empty_managed_index_on_startup(
    root: Path,
    *,
    archive_owner: OwnedArchiveLocation,
    write_admission: Callable[[str], AbstractContextManager[object]],
) -> str | None:
    """Promote fresh empty DDL only from proved-empty original custody.

    The admitted startup worker prepares outside writer custody. It enters
    the existing writer bridge only for generation creation and pointer
    publication. No old Index is deleted or stamped; normal generation
    retention owns its later lifetime. Populated or structurally changed
    archives retain their original schema-blocked route.
    """
    from polylogue.operations.durable_change_train import assert_holds_archive_ownership
    from polylogue.operations.reset_safety import archive_tiers_are_closed

    assert_holds_archive_ownership(archive_owner, root)
    if not archive_tiers_are_closed(root):
        raise EmptyIndexTransitionRefusedError("startup_tiers_not_closed")
    if not (root / ".index-active-pointer").exists():
        return None
    location = ArchiveLocation.resolve(root)
    index = location.active_index_path.resolve(strict=True)
    canonical_index = canonical_active_index_path(location)
    generations_root = canonical_index.parent / ".index-generations"
    bootstrap = (
        not canonical_index.is_symlink()
        and canonical_index.resolve(strict=True) == index
        and location.configured_tier("index").resolved_path == index
    )
    if not bootstrap and (index.parent.parent != generations_root.resolve() or index.name != "index.db"):
        raise EmptyIndexTransitionRefusedError("managed_index_escapes_archive")
    with VerifiedAuditLeaf(index.parent, filename=index.name) as leaf:
        with closing(open_readonly_connection(leaf.anchored_path, validate_schema=False)) as conn:
            try:
                assert_tier_schema_supported(conn, index, ArchiveTier.INDEX)
            except SchemaSkew:
                pass
            else:
                return None
            if read_schema_identity(conn, DerivedTier.INDEX) is None:
                raise EmptyIndexTransitionRefusedError("missing_index_identity")
            _require_matching_shape(conn, ArchiveTier.INDEX)
        leaf.assert_unchanged()

    # This existing anchored owner resolves every durable User/Audit target
    # against the actual old generation even though its derived identity is
    # stale. It retains those observers through the final publication check.
    with PreparedIndexMutation(index, archive_root=root) as original, ExitStack() as embedding_custody:
        with original.original_read_snapshot():
            for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT):
                conn = original.observer(tier.value)
                try:
                    assert_tier_schema_supported(conn, root / f"{tier.value}.db", tier)
                except SchemaSkew as exc:
                    raise EmptyIndexTransitionRefusedError(f"unsupported_{tier.value}_schema") from exc
                _require_matching_shape(conn, tier)
            _require_empty_population(original, "source")
            _require_empty_population(original, "index")
        embedding_binding = location.configured_tier("embeddings")
        embedding = embedding_binding.resolved_path
        embedding_entry = embedding_binding.configured_path.lstat()
        embedding_entry_identity = (embedding_entry.st_dev, embedding_entry.st_ino)
        embedding_leaf = embedding_custody.enter_context(VerifiedAuditLeaf(embedding.parent, filename=embedding.name))
        embedding_conn = embedding_custody.enter_context(
            closing(open_readonly_connection(embedding_leaf.anchored_path, validate_schema=False))
        )
        with closing(embedding_conn.execute("PRAGMA data_version")) as rows:
            embedding_version = int(rows.fetchone()[0])
        try:
            assert_tier_schema_supported(embedding_conn, embedding, ArchiveTier.EMBEDDINGS)
        except SchemaSkew as exc:
            raise EmptyIndexTransitionRefusedError("unsupported_embeddings_schema") from exc
        _require_matching_shape(embedding_conn, ArchiveTier.EMBEDDINGS)

        def require_embedding_current() -> None:
            embedding_leaf.assert_unchanged()
            current_entry = embedding_binding.configured_path.lstat()
            if (
                TierFileIdentity.resolve("embeddings", embedding_binding.configured_path) != embedding_binding
                or (current_entry.st_dev, current_entry.st_ino) != embedding_entry_identity
            ):
                raise EmptyIndexTransitionRefusedError("changed_embeddings_binding")
            with closing(embedding_conn.execute("PRAGMA data_version")) as rows:
                if int(rows.fetchone()[0]) != embedding_version:
                    raise EmptyIndexTransitionRefusedError("changed_embeddings_custody")

        require_embedding_current()
        _require_empty_blob_custody(root)
        source_snapshot = rebuild_source_evidence_snapshot(root)
        original.validate_observers_current()
        with write_admission("daemon.empty_index.create"):
            original.validate_observers_current()
            require_embedding_current()
            store = IndexGenerationStore(location, repair_anchor=False)
            if not bootstrap:
                parent = store.load(index.parent.name)
                if parent.state != "active" or Path(parent.index_path).resolve() != index:
                    raise EmptyIndexTransitionRefusedError("unprovable_active_generation")
            generation = store.create(owner_id="daemon:empty-index-startup", source_snapshot=source_snapshot)
        with store.prepare_promotion(generation) as prepared:
            if prepared.missing_session_count:
                raise EmptyIndexTransitionRefusedError("active_reference_coverage_incomplete")
            with write_admission("daemon.empty_index.promote"):
                original.validate_observers_current()
                require_embedding_current()
                _require_empty_blob_custody(root)
                promoted = store.promote(generation, prepared)
        return promoted.generation_id
