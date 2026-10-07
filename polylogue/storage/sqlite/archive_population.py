"""Populate authenticated detached archives under fresh destination authority."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
from collections.abc import Callable, Mapping
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.sqlite import durable_change_train, migration_runner
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_BASELINE_VERSION_BY_TIER, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive_plan import (
    ARCHIVE_FORMAT_MARKER_NAME,
    _assert_archive_format_tier_lineage,
    _read_archive_format_birth,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    ARCHIVE_TIER_SPECS,
    _initialize_population_archive_stage,
    initialize_active_archive_root,
    invalidate_active_archive_bootstrap,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import DurableChangeTrainState
from polylogue.storage.sqlite.write_lease import ARCHIVE_WRITE_CUSTODY_LOCK_NAME

_HISTORY = Path(".maintenance-state/durable-change-trains")
_PROVENANCE = Path(".archive-population-provenance")
_DURABLE = (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT)


class ArchivePopulationError(ValueError):
    """An authenticated archive recipe cannot be populated under the current schema."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ArchivePopulationProof:
    replaced_paths: frozenset[str]
    source_manifest_id: str
    new_derived_tiers: frozenset[str] = frozenset()


def populate_authenticated_archive(
    source: Path,
    destination: Path,
    *,
    source_manifest_id: str,
    source_files: tuple[tuple[str, int, str], ...],
    retained_artifact_reference: bool,
    validate_source_files: Callable[[], None],
    original_tier_identities: Mapping[ArchiveTier, str] | None = None,
) -> ArchivePopulationProof | None:
    """Populate under one exact fence, including authenticated fixture clones."""
    from polylogue.storage.sqlite.population_admission import require_population_admission

    if not (source / ARCHIVE_FORMAT_MARKER_NAME).is_file():
        return None
    if destination.is_symlink():
        raise ArchivePopulationError("destination_symlink")
    destination = destination.resolve(strict=True)

    def populate() -> ArchivePopulationProof:
        proof = _populate_authenticated_archive(
            source,
            destination,
            source_manifest_id=source_manifest_id,
            source_files=source_files,
            retained_artifact_reference=retained_artifact_reference,
            validate_source_files=validate_source_files,
            original_tier_identities=original_tier_identities,
        )
        assert proof is not None
        for relative, size, digest in source_files:
            if relative not in proof.replaced_paths and _literal_file_evidence(destination / relative) != (
                size,
                digest,
            ):
                raise ArchivePopulationError("destination_unowned_file_changed")
        return proof

    # Every producer reserves and fences the actual destination before copy.
    # This owner consumes only that live capability, never an ambient directory.
    require_population_admission(destination)
    return populate()


def _literal_file_evidence(path: Path) -> tuple[int, str]:
    if not path.is_file() or path.is_symlink():
        raise ArchivePopulationError("invalid_literal_file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _populate_authenticated_archive(
    source: Path,
    destination: Path,
    *,
    source_manifest_id: str,
    source_files: tuple[tuple[str, int, str], ...],
    retained_artifact_reference: bool,
    validate_source_files: Callable[[], None],
    original_tier_identities: Mapping[ArchiveTier, str] | None = None,
) -> ArchivePopulationProof | None:
    """Consume an already authenticated detached copy under its source lock.

    The caller proves the entire original file set before entry and again
    proves all unchanged files afterwards. This owner accounts for changed
    durable pages and regenerated history with exact SQLite evidence. Generic
    authenticated file-tree recipes without archive format authority remain byte copies.
    """
    if not (source / ARCHIVE_FORMAT_MARKER_NAME).is_file():
        return None
    if any(relative.endswith((".db-wal", ".db-shm", ".db-journal")) for relative, _, _ in source_files):
        raise ArchivePopulationError("unsealed_source_sqlite_sidecars")
    birth = _read_archive_format_birth(source)
    history_paths = durable_change_train.durable_train_manifest_paths(source / _HISTORY)
    trains = tuple(durable_change_train.load_durable_change_train_manifest(path) for path in history_paths)
    if any(train.state is not DurableChangeTrainState.RELEASED for train in trains):
        raise ArchivePopulationError("unreleased_source_history")
    replaced = {
        relative
        for relative, _, _ in source_files
        if relative == ARCHIVE_FORMAT_MARKER_NAME or Path(relative).is_relative_to(_HISTORY)
    }
    # These exact files belong to the held destination offline/physical owner,
    # including callers that reserved the fence before copying the recipe.
    replaced.update({"daemon.pid", ".archive-ownership.lock", ARCHIVE_WRITE_CUSTODY_LOCK_NAME})
    with ExitStack() as stack:
        sources: dict[ArchiveTier, sqlite3.Connection] = {}
        evidence: dict[ArchiveTier, tuple[str, str]] = {}
        versions: dict[ArchiveTier, int] = {}
        for tier in _DURABLE:
            path = source / ARCHIVE_TIER_SPECS[tier].filename
            conn = stack.enter_context(closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)))
            conn.execute("BEGIN")
            _assert_archive_format_tier_lineage(source, tier, conn, birth)
            sources[tier] = conn
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if not ARCHIVE_BASELINE_VERSION_BY_TIER[tier] <= version <= ARCHIVE_VERSION_BY_TIER[tier]:
                raise ArchivePopulationError("unsupported_source_target")
            versions[tier] = version
            inventory = migration_runner.capture_durable_schema_inventory(conn)
            canonical = durable_change_train._canonical_schema_inventory(tier, version)
            if inventory.sha256 != canonical.sha256:
                raise ArchivePopulationError("unsupported_source_schema")
            tier_trains = {train.target_version: train for train in trains if train.tier is tier}
            floor = durable_change_train.DURABLE_MIGRATION_ADOPTION_FLOORS[tier]
            if version > floor:
                durable_change_train._require_released_train_chain(
                    tier,
                    tier_trains,
                    current_version=version,
                    floor=floor,
                )
            for train in tier_trains.values():
                if original_tier_identities is None:
                    durable_change_train._verify_released_train_live_tier(
                        conn,
                        train,
                        expected_live_schema_inventory_sha256=inventory.sha256,
                    )
                else:
                    # The authenticated backup names the original physical
                    # owner. Preserve its historical bindings; this relocated
                    # source is never admitted as that live archive.
                    durable_change_train._historical_schema_evidence(train)
                    if (
                        train.apply_evidence is None
                        or train.apply_evidence.post.archive_identity_digest != original_tier_identities.get(tier)
                        or durable_change_train._released_live_schema_inventory_sha256(tier, version, tier_trains)
                        != inventory.sha256
                    ):
                        raise ArchivePopulationError("detached_source_history_binding_mismatch")
            if tuple(row[0] for row in conn.execute("PRAGMA integrity_check")) != ("ok",):
                raise ArchivePopulationError("source_integrity_failed")
            if tuple(conn.execute("PRAGMA foreign_key_check")):
                raise ArchivePopulationError("source_foreign_keys_failed")
            evidence[tier] = (inventory.sha256, migration_runner._durable_literal_rows_digest(conn))
            replaced.add(ARCHIVE_TIER_SPECS[tier].filename)

        from polylogue.storage.sqlite.population_admission import _bound_population_stage

        # Validate and bind the complete authenticated target map before any
        # destination provenance or durable inode is changed.
        with _bound_population_stage(destination, {tier.value: version for tier, version in versions.items()}):
            validate_source_files()
            provenance = destination / _PROVENANCE / hashlib.sha256(source_manifest_id.encode()).hexdigest()
            if provenance.exists():
                raise ArchivePopulationError("provenance_namespace_conflict")
            provenance.mkdir(parents=True)
            from polylogue.core.errors import SchemaSkew
            from polylogue.storage.sqlite.connection_profile import (
                assert_tier_schema_supported,
                open_readonly_connection,
            )

            new_derived: set[str] = set()
            for tier in (ArchiveTier.INDEX, ArchiveTier.OPS):
                leaf = destination / ARCHIVE_TIER_SPECS[tier].filename
                if not leaf.exists() and not leaf.is_symlink():
                    new_derived.add(leaf.name)
                    continue
                metadata = leaf.lstat()
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise ArchivePopulationError("invalid_derived_leaf")
                try:
                    with closing(open_readonly_connection(leaf, validate_schema=False)) as derived:
                        if derived.execute("PRAGMA user_version").fetchone()[0] != ARCHIVE_TIER_SPECS[tier].version:
                            raise ArchivePopulationError("unsupported_derived_version")
                        assert_tier_schema_supported(derived, leaf, tier)
                except SchemaSkew:
                    # Derived identity changes are not durable evolution.
                    # Preserve the original copied file as detached evidence;
                    # the canonical constructor creates its current replacement.
                    original_derived = provenance / "original-derived"
                    original_derived.mkdir(exist_ok=True)
                    leaf.replace(original_derived / leaf.name)
                    replaced.add(leaf.name)
                    new_derived.add(leaf.name)

            original_history = destination / _HISTORY
            history_files = tuple(
                item
                for item in source_files
                if Path(item[0]).is_relative_to(_HISTORY) or item[0] == ARCHIVE_FORMAT_MARKER_NAME
            )
            if not os.path.lexists(original_history):
                # A source that copied no released train and no bootstrap
                # receipt carries no history directory; nothing is detached.
                pass
            elif retained_artifact_reference:
                # Cached recipes already retain these immutable bytes in their
                # authenticated artifact owner; keep its reference rather than a
                # separate per-clone copy of the complete train history.
                shutil.rmtree(original_history)
            else:
                original_history.replace(provenance / "original-history")
            (provenance / "source.json").write_text(
                json.dumps(
                    {
                        "source_manifest_id": source_manifest_id,
                        "owning_artifact": str(source) if retained_artifact_reference else None,
                        "original_receipts": history_files,
                        "regenerated_history": str(_HISTORY),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            birth_marker = destination / ARCHIVE_FORMAT_MARKER_NAME
            if retained_artifact_reference:
                birth_marker.unlink()
            else:
                birth_marker.replace(provenance / "original-format.json")
            for tier in _DURABLE:
                path = destination / ARCHIVE_TIER_SPECS[tier].filename
                path.unlink()
                for suffix in ("-wal", "-shm"):
                    path.with_name(path.name + suffix).unlink(missing_ok=True)
            invalidate_active_archive_bootstrap(destination)
            backup_root = destination / ".maintenance-state" / "pre-migration-backups"

            def backup_files() -> set[str]:
                if backup_root.is_symlink():
                    raise ArchivePopulationError("invalid_literal_file")
                files: set[str] = set()
                for path in backup_root.rglob("*"):
                    if path.is_symlink():
                        raise ArchivePopulationError("invalid_literal_file")
                    if path.is_file():
                        _literal_file_evidence(path)
                        files.add(str(path.relative_to(destination)))
                return files

            original_backup_paths = backup_files()
            _initialize_population_archive_stage(destination)
            # Canonical ownership admission rewrites this exact lock's owner
            # record for the destination. Other locks and fixture files retain
            # their authenticated original bytes.
            replaced.add(".archive-ownership.lock")
            for tier, source_conn in sources.items():
                path = destination / ARCHIVE_TIER_SPECS[tier].filename
                identity = (path.stat().st_dev, path.stat().st_ino)
                with closing(sqlite3.connect(path)) as target:
                    source_conn.backup(target)
                    observed = (
                        migration_runner.capture_durable_schema_inventory(target).sha256,
                        migration_runner._durable_literal_rows_digest(target),
                    )
                    if observed != evidence[tier]:
                        raise ArchivePopulationError("destination_row_or_schema_mismatch")
                    if int(target.execute("PRAGMA user_version").fetchone()[0]) != versions[tier]:
                        raise ArchivePopulationError("destination_version_mismatch")
                    if tuple(target.execute("PRAGMA foreign_key_check")):
                        raise ArchivePopulationError("destination_foreign_keys_failed")
                if (path.stat().st_dev, path.stat().st_ino) != identity:
                    raise ArchivePopulationError("destination_inode_changed")
            validate_source_files()
            invalidate_active_archive_bootstrap(destination)
            # This is the ordinary released-state consumer, after real source
            # population; copied original receipts never participate in admission.
            initialize_active_archive_root(destination)
            # Both canonical constructor phases own their new verified backup
            # packages. Existing detached backup bytes retain source authority.
            replaced.update(backup_files() - original_backup_paths)
    replaced.update(
        str(path.relative_to(destination))
        for root in (destination / _HISTORY, provenance)
        for path in root.rglob("*")
        if path.is_file()
    )
    return ArchivePopulationProof(frozenset(replaced), source_manifest_id, frozenset(new_derived))
