"""Exercise material preparation, publication and transactional admission."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.materials import (
    MaterialObservation,
    PreparedMaterial,
    admit_material,
    publish_prepared_materials,
)
from polylogue.storage.sqlite.write_lease import write_lease


def material_publisher(conn: sqlite3.Connection, store: BlobStore) -> ArchiveBlobPublisher:
    source_path = next(Path(row[2]) for row in conn.execute("PRAGMA database_list") if row[1] == "main" and row[2])
    return ArchiveBlobPublisher(source_path, store.root)


def apply_material_preparation(
    conn: sqlite3.Connection,
    *,
    prepared: PreparedMaterial,
    observed_at_ms: int,
    supersedes_material_id: str | None = None,
    commit: bool = True,
) -> MaterialObservation:
    # The argument has already been prepared before acquiring writer custody.
    assert not conn.in_transaction, "publication must precede the Source transaction"
    with write_lease("synthetic-material-test", archive_root=prepared.publisher.source_db_path.parent):
        publish_prepared_materials((prepared,))
        return admit_material(
            conn,
            prepared=prepared,
            observed_at_ms=observed_at_ms,
            supersedes_material_id=supersedes_material_id,
            commit=commit,
        )
