"""Drive attachment convergence must not resurrect durably excised bytes.

The whole-archive convergence stage selects any ``unfetched``
``upload_origin='drive'`` attachment reference, downloads it, and hashes it
back to the exact hash the operator excised. Before this guard it published
those bytes, wrote a live ``blob_refs`` row for them, and flipped the
attachment to ``acquired``.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from polylogue.operations.attachment_convergence import converge_drive_attachments
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.unit.daemon.test_attachment_convergence import _open_index, _session


def test_drive_backfill_refuses_bytes_the_operator_excised(tmp_path: Path) -> None:
    """A re-downloadable attachment the operator excised stays excised.

    Anti-vacuity: removing the ``is_blob_hash_excised`` check in
    ``converge_drive_attachments`` (or the matching gate in
    ``write_source_blob_refs``) restores the ``blob_refs`` row and the
    ``acquired`` status, failing the durable-state assertions below.
    Disabling the stage entirely does not make this green either: a
    non-excised sibling is asserted to still be acquired in the same pass.
    """
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_parsed_session_to_archive(index, _session("excised-one", file_id="drive-excised"), raw_id="excised-raw")
    write_parsed_session_to_archive(index, _session("kept-one", file_id="drive-kept"), raw_id="kept-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)

    excised_payload = b"the attachment the operator excised"
    kept_payload = b"an unrelated attachment"
    excised_hash = hashlib.sha256(excised_payload).digest()
    kept_hash = hashlib.sha256(kept_payload).digest()
    with source:
        source.execute(
            "INSERT INTO excised_content (removed_hash, hash_kind, reason, actor, excised_at_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (excised_hash, "blob_hash", "operator excision", "operator", 1000),
        )

    payloads = {"drive-excised": excised_payload, "drive-kept": kept_payload}

    result = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=lambda file_id: payloads[file_id],
        limit=10,
    )

    assert result.excised == 1
    assert result.acquired == 1, "an unrelated attachment must still be acquired in the same pass"

    statuses = {
        str(row["attachment_id"]): (row["blob_hash"], str(row["acquisition_status"]))
        for row in index.execute("SELECT attachment_id, blob_hash, acquisition_status FROM attachments")
    }
    assert sorted(status for _hash, status in statuses.values()) == ["acquired", "unavailable"]
    assert all(row_hash != excised_hash for row_hash, _status in statuses.values())

    ref_hashes = {bytes(row["blob_hash"]) for row in source.execute("SELECT blob_hash FROM blob_refs")}
    assert excised_hash not in ref_hashes, "a live blob_refs row was recreated for excised content"
    assert kept_hash in ref_hashes

    # The refusal happens before publication, so nothing reserves the excised
    # hash -- a reservation would make it permanently GC-immune.
    reserved = {
        bytes(row["blob_hash"]) for row in source.execute("SELECT blob_hash FROM blob_publication_reservations")
    }
    assert excised_hash not in reserved
    index.close()
    source.close()
