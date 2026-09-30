"""Populate synthetic clones without executing another archive's train receipts."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Callable
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.sqlite import durable_change_train, migration_runner
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive_plan import (
    ARCHIVE_FORMAT_MARKER_NAME,
    assert_archive_format_lineage,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    ARCHIVE_TIER_SPECS,
    initialize_active_archive_root,
    invalidate_active_archive_bootstrap,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import DurableChangeTrainState

_HISTORY = Path(".maintenance-state/durable-change-trains")
_PROVENANCE = Path(".fixture-archive-provenance")
_DURABLE = (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT)


class FixtureArchiveCloneError(ValueError):
    """A declared archive recipe cannot be populated under the current schema."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class FixtureArchiveCloneProof:
    replaced_paths: frozenset[str]
    source_manifest_id: str


def populate_authenticated_archive_clone(
    source: Path,
    destination: Path,
    *,
    source_manifest_id: str,
    source_files: tuple[tuple[str, int, str], ...],
    retained_artifact_reference: bool,
    validate_source_files: Callable[[], None],
) -> FixtureArchiveCloneProof | None:
    """Consume an already authenticated detached copy under its source lock.

    The caller proves the entire original file set before entry and again
    proves all unchanged files afterwards. This owner accounts for changed
    durable pages and regenerated history with exact SQLite evidence. Generic
    file-tree recipes without archive format authority remain byte copies.
    """
    if not (source / ARCHIVE_FORMAT_MARKER_NAME).is_file():
        return None
    if any(relative.endswith((".db-wal", ".db-shm", ".db-journal")) for relative, _, _ in source_files):
        raise FixtureArchiveCloneError("unsealed_source_sqlite_sidecars")
    assert_archive_format_lineage(source)
    history_paths = durable_change_train._durable_train_manifest_paths(source / _HISTORY)
    trains = tuple(durable_change_train.load_durable_change_train_manifest(path) for path in history_paths)
    if any(train.state is not DurableChangeTrainState.RELEASED for train in trains):
        raise FixtureArchiveCloneError("unreleased_source_history")
    replaced = {
        relative
        for relative, _, _ in source_files
        if relative == ARCHIVE_FORMAT_MARKER_NAME or Path(relative).is_relative_to(_HISTORY)
    }
    with ExitStack() as stack:
        sources: dict[ArchiveTier, sqlite3.Connection] = {}
        evidence: dict[ArchiveTier, tuple[str, str]] = {}
        for tier in _DURABLE:
            path = source / ARCHIVE_TIER_SPECS[tier].filename
            conn = stack.enter_context(closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)))
            conn.execute("BEGIN")
            sources[tier] = conn
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version != ARCHIVE_VERSION_BY_TIER[tier]:
                raise FixtureArchiveCloneError("unsupported_source_target")
            inventory = migration_runner.capture_durable_schema_inventory(conn)
            canonical = durable_change_train._canonical_schema_inventory(tier, version)
            if inventory.sha256 != canonical.sha256:
                raise FixtureArchiveCloneError("unsupported_source_schema")
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
                durable_change_train._verify_released_train_live_tier(
                    conn,
                    train,
                    expected_live_schema_inventory_sha256=inventory.sha256,
                )
            if tuple(row[0] for row in conn.execute("PRAGMA integrity_check")) != ("ok",):
                raise FixtureArchiveCloneError("source_integrity_failed")
            if tuple(conn.execute("PRAGMA foreign_key_check")):
                raise FixtureArchiveCloneError("source_foreign_keys_failed")
            evidence[tier] = (inventory.sha256, migration_runner._durable_literal_rows_digest(conn))
            replaced.add(ARCHIVE_TIER_SPECS[tier].filename)

        validate_source_files()
        provenance = destination / _PROVENANCE / hashlib.sha256(source_manifest_id.encode()).hexdigest()
        if provenance.exists():
            raise FixtureArchiveCloneError("provenance_namespace_conflict")
        provenance.mkdir(parents=True)
        original_history = destination / _HISTORY
        history_files = tuple(
            item
            for item in source_files
            if Path(item[0]).is_relative_to(_HISTORY) or item[0] == ARCHIVE_FORMAT_MARKER_NAME
        )
        if retained_artifact_reference:
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
        initialize_active_archive_root(destination)
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
                    raise FixtureArchiveCloneError("destination_row_or_schema_mismatch")
                if int(target.execute("PRAGMA user_version").fetchone()[0]) != ARCHIVE_VERSION_BY_TIER[tier]:
                    raise FixtureArchiveCloneError("destination_version_mismatch")
                if tuple(target.execute("PRAGMA foreign_key_check")):
                    raise FixtureArchiveCloneError("destination_foreign_keys_failed")
            if (path.stat().st_dev, path.stat().st_ino) != identity:
                raise FixtureArchiveCloneError("destination_inode_changed")
        validate_source_files()
        invalidate_active_archive_bootstrap(destination)
        # This is the ordinary released-state consumer, after real source
        # population; copied original receipts never participate in admission.
        initialize_active_archive_root(destination)
    replaced.update(
        str(path.relative_to(destination))
        for root in (destination / _HISTORY, provenance)
        for path in root.rglob("*")
        if path.is_file()
    )
    return FixtureArchiveCloneProof(frozenset(replaced), source_manifest_id)
