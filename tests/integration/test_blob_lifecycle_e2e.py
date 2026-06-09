"""End-to-end blob integrity lifecycle (#818).

Pins the operator-facing closure for blob orphan cleanup and integrity
verification. Each test exercises a real ``BlobStore`` against a real
SQLite ``raw_sessions`` table — no mocks of the storage layer —
and asserts the round-trip:

  ingest a referenced blob and an orphaned blob
  → ``detect_orphans``       reports the orphan
  → ``cleanup_orphans`` dry  reports what would be deleted, file untouched
  → ``cleanup_orphans`` live deletes the orphan
  → re-detect               reports zero orphans
  → ``verify_all``           confirms the referenced blob's integrity

This test guards against regressions where one of these surfaces is
silently disconnected (e.g., maintenance target unregistered, repair
handler routed to a stale set of orphan hashes).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypeAlias

CliWorkspace: TypeAlias = dict[str, Path]


_REFERENCED_BLOB = b"referenced raw session content"
_ORPHAN_BLOB = b"this blob has no raw_sessions row"


def _seed_raw_row(db_path: Path, raw_id: str, source_path: Path, blob_size: int) -> None:
    """Insert a minimum-viable raw_sessions row pointing at *raw_id*."""
    source_db_path = db_path.with_name("source.db")
    with sqlite3.connect(source_db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (raw_id, "unknown-export", raw_id, str(source_path), bytes.fromhex(raw_id), blob_size, 1_746_830_400_000),
        )
        conn.commit()


class TestBlobOrphanLifecycle:
    """Direct ``BlobStore`` API end-to-end against a real DB."""

    def test_detect_dry_run_apply_verify_roundtrip(self, cli_workspace: CliWorkspace) -> None:
        from polylogue.storage.blob_store import get_blob_store

        store = get_blob_store()
        referenced_hash, ref_size = store.write_from_bytes(_REFERENCED_BLOB)
        orphan_hash, orphan_size = store.write_from_bytes(_ORPHAN_BLOB)

        assert referenced_hash != orphan_hash
        assert store.exists(referenced_hash) is True
        assert store.exists(orphan_hash) is True

        # Only the referenced blob is registered in the DB. The orphan
        # is intentionally absent — the GC contract is "anything on disk
        # without a raw_sessions.raw_id row is an orphan".
        _seed_raw_row(
            cli_workspace["db_path"],
            referenced_hash,
            cli_workspace["inbox_dir"] / "ingested.json",
            ref_size,
        )

        db_referenced_ids = {referenced_hash}

        # Detect — orphan_count == 1, sample contains the orphan hash,
        # bytes match the orphan's on-disk size.
        detect = store.detect_orphans(db_referenced_ids)
        assert detect.orphan_count == 1
        assert detect.orphan_bytes == orphan_size
        assert orphan_hash in set(detect.orphan_samples)

        # Dry-run cleanup — reports what would be deleted, file untouched.
        dry = store.cleanup_orphans({orphan_hash}, dry_run=True)
        assert dry.dry_run is True
        assert dry.would_delete_count == 1
        assert dry.would_delete_bytes == orphan_size
        assert dry.deleted_count == 0
        assert store.exists(orphan_hash) is True, "dry-run must not delete"

        # Live cleanup — orphan gone, referenced blob untouched.
        applied = store.cleanup_orphans({orphan_hash}, dry_run=False)
        assert applied.dry_run is False
        assert applied.deleted_count == 1
        assert applied.deleted_bytes == orphan_size
        assert applied.errors == 0
        assert store.exists(orphan_hash) is False
        assert store.exists(referenced_hash) is True

        # Re-detect — zero orphans now.
        post = store.detect_orphans(db_referenced_ids)
        assert post.orphan_count == 0
        assert post.orphan_bytes == 0
        assert post.orphan_samples == ()

        # Integrity check — referenced blob still hashes correctly.
        verify = store.verify_all()
        assert verify.passed is True
        assert verify.checked == 1
        assert verify.failed_count == 0

    def test_cleanup_apply_is_idempotent_when_orphan_already_gone(self, cli_workspace: CliWorkspace) -> None:
        """TOCTOU: caller passes a hash that vanished between detect and cleanup.

        ``cleanup_orphans(dry_run=False)`` must not raise on a missing path —
        it should report the deletion as a no-op and continue. Critical for
        concurrent ingest scenarios where a blob may be re-referenced (or
        re-deleted) between the two calls.
        """
        from polylogue.storage.blob_store import get_blob_store

        store = get_blob_store()
        gone_hash, gone_size = store.write_from_bytes(b"will vanish")

        # Caller manually removes the blob first (simulating concurrent GC
        # or a dropped raw row → blob unlink elsewhere).
        store.remove(gone_hash)
        assert store.exists(gone_hash) is False

        result = store.cleanup_orphans({gone_hash}, dry_run=False)
        assert result.deleted_count == 0
        assert result.errors == 0
        assert result.error_details == ()
        assert result.deleted_bytes == 0


class TestRepairOrphanedBlobsHandler:
    """The maintenance-target wiring (`repair_orphaned_blobs`).

    Pins 818-A6: ``polylogue doctor --repair --target orphaned_blobs``
    must route through this handler and converge an orphaned-blob state
    to a clean state.
    """

    def test_dry_run_then_apply_clears_orphans(self, cli_workspace: CliWorkspace) -> None:
        from polylogue.config import get_config
        from polylogue.storage.blob_store import get_blob_store
        from polylogue.storage.repair import repair_orphaned_blobs

        store = get_blob_store()
        referenced_hash, ref_size = store.write_from_bytes(_REFERENCED_BLOB)
        orphan_hash, _ = store.write_from_bytes(_ORPHAN_BLOB)
        _seed_raw_row(
            cli_workspace["db_path"],
            referenced_hash,
            cli_workspace["inbox_dir"] / "ingested.json",
            ref_size,
        )

        config = get_config()

        # Dry-run: handler reports one would-be deletion, doesn't delete.
        dry = repair_orphaned_blobs(config, dry_run=True)
        assert dry.name == "orphaned_blobs"
        assert dry.success is True
        assert dry.repaired_count == 1
        assert "Would: delete 1" in dry.detail
        assert store.exists(orphan_hash) is True

        # Live: handler deletes the orphan and reports it.
        applied = repair_orphaned_blobs(config, dry_run=False)
        assert applied.name == "orphaned_blobs"
        assert applied.success is True
        assert applied.repaired_count == 1
        assert "Deleted 1" in applied.detail
        assert store.exists(orphan_hash) is False
        assert store.exists(referenced_hash) is True

        # Subsequent run: clean state, no work.
        clean = repair_orphaned_blobs(config, dry_run=False)
        assert clean.name == "orphaned_blobs"
        assert clean.success is True
        assert clean.repaired_count == 0
        assert "No orphaned blobs" in clean.detail

    def test_clean_state_returns_no_op(self, cli_workspace: CliWorkspace) -> None:
        from polylogue.config import get_config
        from polylogue.storage.blob_store import get_blob_store
        from polylogue.storage.repair import repair_orphaned_blobs

        store = get_blob_store()
        referenced_hash, ref_size = store.write_from_bytes(_REFERENCED_BLOB)
        _seed_raw_row(
            cli_workspace["db_path"],
            referenced_hash,
            cli_workspace["inbox_dir"] / "ingested.json",
            ref_size,
        )

        config = get_config()
        result = repair_orphaned_blobs(config, dry_run=False)
        assert result.name == "orphaned_blobs"
        assert result.success is True
        assert result.repaired_count == 0


class TestOrphanedBlobsCatalogRouting:
    """Pin the wiring from the maintenance-target catalog to the handler.

    818-A6: ``polylogue doctor --repair --target orphaned_blobs`` resolves
    via ``MaintenanceTargetCatalog`` to the ``orphaned_blobs`` spec and
    the ``_REPAIR_HANDLERS["orphaned_blobs"]`` entry. A regression that
    deletes the spec, deletes the handler dict entry, drops the destructive
    flag, or splits the name across the two structures would not be
    caught by handler-only unit tests.
    """

    def test_catalog_resolves_orphaned_blobs_target(self) -> None:
        from polylogue.maintenance.models import MaintenanceCategory
        from polylogue.maintenance.targets import (
            MaintenanceTargetMode,
            build_maintenance_target_catalog,
        )

        catalog = build_maintenance_target_catalog()
        spec = catalog.resolve_name("orphaned_blobs")
        assert spec is not None, "orphaned_blobs target must be registered"
        assert spec.name == "orphaned_blobs"
        assert spec.mode is MaintenanceTargetMode.CLEANUP
        assert spec.category is MaintenanceCategory.ARCHIVE_CLEANUP
        assert spec.destructive is True, (
            "blob deletion must be marked destructive so doctor requires explicit --repair / --cleanup confirmation"
        )

    def test_repair_handler_dict_routes_orphaned_blobs(self) -> None:
        """``_REPAIR_HANDLERS["orphaned_blobs"]`` must dispatch to the
        function under test in ``TestRepairOrphanedBlobsHandler``.
        """
        from polylogue.storage import repair

        assert "orphaned_blobs" in repair._REPAIR_HANDLERS
        assert repair._REPAIR_HANDLERS["orphaned_blobs"] is repair.repair_orphaned_blobs
        assert "orphaned_blobs" in repair._PREVIEW_HANDLERS
        assert repair._PREVIEW_HANDLERS["orphaned_blobs"] is repair.preview_orphaned_blobs

    def test_catalog_resolution_drives_end_to_end_cleanup(self, cli_workspace: CliWorkspace) -> None:
        """End-to-end via catalog → handler dict → handler call.

        Mirrors the doctor command's actual dispatch sequence: resolve a
        target name through the catalog, look up the handler in the
        repair-handler dict, invoke it against a real archive. This is
        the routing path that was previously untested.
        """
        from polylogue.config import get_config
        from polylogue.maintenance.targets import build_maintenance_target_catalog
        from polylogue.storage import repair
        from polylogue.storage.blob_store import get_blob_store

        store = get_blob_store()
        referenced_hash, ref_size = store.write_from_bytes(_REFERENCED_BLOB)
        orphan_hash, _ = store.write_from_bytes(_ORPHAN_BLOB)
        _seed_raw_row(
            cli_workspace["db_path"],
            referenced_hash,
            cli_workspace["inbox_dir"] / "ingested.json",
            ref_size,
        )

        catalog = build_maintenance_target_catalog()
        spec = catalog.resolve_name("orphaned_blobs")
        assert spec is not None
        handler = repair._REPAIR_HANDLERS[spec.name]

        result = handler(get_config(), dry_run=False)
        assert result.name == "orphaned_blobs"
        assert result.repaired_count == 1
        assert result.success is True
        assert store.exists(orphan_hash) is False
        assert store.exists(referenced_hash) is True
