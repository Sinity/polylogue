"""Source coordinates retain their exact keys and physical cleanup obligations."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.sources.prepared_jsonl import PreparedJsonl
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    NativeSQLCustodyOwner,
    native_sql_children,
    open_source_tier_write_connection,
)
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealError
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.sqlite_cursor_settlement import ControlledConnection


@pytest.mark.parametrize("index_damage", ["missing", "wrong_predicate", "wrong_coordinate"])
def test_source_coordinate_key_requires_published_partial_indexes(tmp_path: Path, index_damage: str) -> None:
    with write_lease("test.source-coordinate-index", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as source:
            with connection_cursor(source, "DROP INDEX idx_blob_refs_attachment_identity"):
                pass
            if index_damage != "missing":
                predicate = (
                    "ref_type != 'attachment'" if index_damage == "wrong_predicate" else "ref_type = 'attachment'"
                )
                coordinate = "source_path" if index_damage == "wrong_coordinate" else "coalesce(source_path, '')"
                with connection_cursor(
                    source,
                    "CREATE UNIQUE INDEX idx_blob_refs_attachment_identity "
                    f"ON blob_refs(blob_hash, ref_type, ref_id, {coordinate}) WHERE {predicate}",
                ):
                    pass
            source.commit()
        with PreparedIndexMutation.source_only(archive_root=tmp_path) as seal, pytest.raises(ReferenceSealError):
            seal._known_tier_table_shape("source", "blob_refs")


@pytest.mark.parametrize("failed_close", [False, True])
def test_artifact_blob_discard_requires_physical_child_settlement(tmp_path: Path, failed_close: bool) -> None:
    with write_lease("test.artifact-blob-retirement", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        seal = PreparedIndexMutation.source_only(archive_root=tmp_path)
        store = BlobStore(tmp_path / "blob")
        publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
        prepared_blob = store.prepare_from_bytes(b"neutral artifact publication")
        claim = publisher.prepare_claim(prepared_blob)
        publisher.queue_prepared(prepared_blob, claim=claim)
        artifact = PreparedJsonl(claim.receipt.blob_hash, None, None, publication_publisher=publisher)
        artifact._blob_publication.publisher = publisher
        artifact._blob_publication.seal = seal
        artifact._blob_publication.started = True
        artifact._blob_publication.page = (claim,)
        connection = sqlite3.connect(":memory:", factory=ControlledConnection)
        owner = NativeSQLCustodyOwner(connection, terminal_parent=seal)
        try:
            if failed_close:
                connection.close_failure = OSError("synthetic original native close failure")
                with pytest.raises(NativeConnectionSettlementError) as failed:
                    owner.close()
                assert failed.value.owner is owner
                assert not owner._settled and owner in native_sql_children(seal)
                with pytest.raises(NativeConnectionSettlementError) as retained:
                    artifact.discard()
                assert retained.value.owner is owner
                assert artifact._blob_publication.page == (claim,)
                assert claim.prepared_path.exists()
                connection.close_failure = None
            owner.close()
            assert owner._settled and owner.close_required and owner in native_sql_children(seal)
            artifact.discard()
            assert artifact._blob_publication.retired
            assert not claim.prepared_path.exists()
            assert owner in native_sql_children(seal)
            seal.close()
            assert not native_sql_children(seal)
        finally:
            connection.close_failure = None
            owner.close()
            seal.close()
            artifact.discard()


def test_source_coordinate_keys_match_partial_unique_index_semantics(tmp_path: Path) -> None:
    with write_lease("test.source-coordinate-keys", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as source:
            with connection_cursor(
                source,
                "INSERT INTO blob_refs(rowid,blob_hash,ref_id,ref_type,source_path,size_bytes,acquired_at_ms) "
                "VALUES(?,?,'neutral-raw',?,?,1,1)",
                (1, b"a" * 32, "attachment", None),
            ):
                pass
            with (
                pytest.raises(sqlite3.IntegrityError),
                connection_cursor(
                    source,
                    "INSERT INTO blob_refs(rowid,blob_hash,ref_id,ref_type,source_path,size_bytes,acquired_at_ms) "
                    "VALUES(2,?,'neutral-raw','attachment','',1,1)",
                    (b"a" * 32,),
                ),
            ):
                pass
            for rowid, ref_type, coordinate in (
                (3, "attachment", "attachment:file-b"),
                (4, "raw_payload", "capture-a"),
            ):
                with connection_cursor(
                    source,
                    "INSERT INTO blob_refs(rowid,blob_hash,ref_id,ref_type,source_path,size_bytes,acquired_at_ms) "
                    "VALUES(?,?,'neutral-raw',?,?,1,1)",
                    (rowid, b"a" * 32, ref_type, coordinate),
                ):
                    pass
            source.commit()
        with PreparedIndexMutation.source_only(archive_root=tmp_path) as seal, seal.original_read_snapshot():
            columns, keys = seal._known_tier_table_shape("source", "blob_refs")
            assert columns[keys[-1]] == "source_path"
            original = seal.retain_tier_row("source", "blob_refs", 1)
            other = seal.retain_tier_row("source", "blob_refs", 3)
            raw = seal.retain_tier_row("source", "blob_refs", 4)
            assert original is not None and other is not None and raw is not None
            assert seal._literal_scalar_equal(seal._known_row_key_cells(original, keys)[-1], "")
            assert seal._literal_scalar_equal(seal._known_row_key_cells(raw, keys)[-1], None)
            assert seal._literal_key_digest(original, keys) != seal._literal_key_digest(other, keys)
            assert original.rowid == 1 and other.rowid == 3 and raw.rowid == 4
