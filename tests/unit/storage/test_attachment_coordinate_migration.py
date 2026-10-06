"""Equal-content coordinates survive the numbered durable Source replacement."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.enums import Origin, Provider
from polylogue.core.errors import SchemaSkew
from polylogue.core.write_lease import write_lease
from polylogue.daemon.durable_migrations import apply_declared_durable_migrations
from polylogue.operations.durable_change_train import acquire_durable_archive_ownership
from polylogue.storage.blob_liveness import LivenessState, inspect_blob_liveness
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceBlobRef,
    write_source_blob_refs,
    write_source_raw_session,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from tests.infra.durable_tier_fixtures import bootstrap_baseline_archive


def test_populated_attachment_migration_requires_real_backup_and_preserves_rowids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing backup or a copy omitting rowid makes this production route red."""
    bootstrap_baseline_archive(tmp_path, monkeypatch)
    source = sqlite3.connect(tmp_path / "source.db")
    store = BlobStore(tmp_path / "blob")
    for ordinal in range(2):
        store.write_from_bytes(f'{{"synthetic_record":{ordinal}}}\n'.encode())
    raw_ids = tuple(
        write_source_raw_session(
            source,
            origin=Origin.CODEX_SESSION,
            capture_mode=Provider.CODEX,
            source_path="/synthetic/windowless.jsonl",
            canonical_source_path="/synthetic/windowless.jsonl",
            source_index=-1,
            payload=f'{{"synthetic_record":{ordinal}}}\n'.encode(),
            acquired_at_ms=1,
        )
        for ordinal in range(2)
    )
    assert source.execute("PRAGMA user_version").fetchone()[0] == 1
    blob_hash, size = store.write_from_bytes(b"identical provider attachment")
    first = ArchiveSourceBlobRef(
        blob_hash=bytes.fromhex(blob_hash),
        ref_type="attachment",
        source_path="attachment:file-a",
        size_bytes=size,
        acquired_at_ms=1,
    )
    write_source_blob_refs(source, raw_ids[0], lambda: (first,))
    source.execute("UPDATE blob_refs SET rowid=-7 WHERE ref_type='attachment'")
    source.commit()
    source.close()
    with pytest.raises(SchemaSkew):
        with closing(open_readonly_connection(tmp_path / "source.db", tier=ArchiveTier.SOURCE)):
            pass
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        before = source.execute("SELECT rowid, * FROM blob_refs ORDER BY rowid").fetchall()
        with pytest.raises(migration_runner.MigrationError):
            migration_runner.migrate_archive_tier(source, ArchiveTier.SOURCE, backup_manifest=None)
        with pytest.raises(migration_runner.MigrationError):
            migration_runner.migrate_archive_tier(
                source,
                ArchiveTier.SOURCE,
                backup_manifest=None,
                allow_pristine_source_baseline=True,
            )
        assert source.execute("SELECT rowid, * FROM blob_refs ORDER BY rowid").fetchall() == before

    with acquire_durable_archive_ownership(tmp_path, owner_id="fixture.attachment-migration") as owner:
        applied = apply_declared_durable_migrations(
            tmp_path,
            archive_owner=owner,
            write_lease=lambda actor: write_lease(actor, archive_root=tmp_path),
        )
    assert [(step.target_version, step.requires_backup) for step in applied] == [
        (2, False),
        (3, True),
        (4, True),
        (5, False),
        (6, True),
    ]
    manifests = list((tmp_path / ".maintenance-state/pre-migration-backups").rglob("manifest.json"))
    assert len(manifests) == 3
    assert all((manifest.parent / "verification-receipt.json").is_file() for manifest in manifests)
    assert (manifests[0].parent / "verification-receipt.json").is_file()
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute("SELECT rowid, * FROM blob_refs ORDER BY rowid").fetchall() == before
        assert source.execute("PRAGMA user_version").fetchone()[0] == 6
        write_source_blob_refs(
            source,
            raw_ids[0],
            lambda: (
                ArchiveSourceBlobRef(
                    blob_hash=first.blob_hash,
                    ref_type="attachment",
                    source_path="attachment:file-b",
                    size_bytes=size,
                    acquired_at_ms=1,
                ),
            ),
        )
        assert source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type='attachment'").fetchone()[0] == 2
        assert inspect_blob_liveness(source, blob_hash).state is LivenessState.LIVE
    initialize_active_archive_root(tmp_path)
