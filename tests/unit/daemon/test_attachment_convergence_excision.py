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
from typing import Any

import pytest

from polylogue.operations.attachment_convergence import AttachmentConvergenceResult, converge_drive_attachments
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.index_writer import write_fixture_index_session
from tests.unit.daemon.test_attachment_convergence import _into, _open_index, _retain_raws, _session


def _converge(index: sqlite3.Connection, source: sqlite3.Connection, **kwargs: Any) -> AttachmentConvergenceResult:
    """A standalone convergence pass: its write sections run under the caller's lease."""
    with write_lease("test.attachment-convergence", archive_root=kwargs["archive_root"]):
        return converge_drive_attachments(index, source, **kwargs)


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
    write_fixture_index_session(index, _session("excised-one", file_id="drive-excised"), raw_id="excised-raw")
    write_fixture_index_session(index, _session("kept-one", file_id="drive-kept"), raw_id="kept-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "excised-raw", "kept-raw")

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

    result = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(lambda file_id: payloads[file_id]),
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


def test_an_excision_committed_before_the_flush_marks_the_attachment_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flush that refuses freshly excised bytes ends the row ``unavailable``.

    Anti-vacuity (Codex P2, #5696): keep the refused download in
    ``acquired_refs`` after the flush and ``write_source_blob_refs`` raises
    ``ContentExcisedError``, aborting the pass before any row is updated.
    """
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_fixture_index_session(index, _session("raced-one", file_id="drive-raced"), raw_id="raced-raw")
    write_fixture_index_session(index, _session("kept-one", file_id="drive-kept"), raw_id="kept-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "raced-raw", "kept-raw")
    raced_payload = b"bytes excised while the pass was downloading"
    raced_hash = hashlib.sha256(raced_payload).digest()
    payloads = {"drive-raced": raced_payload, "drive-kept": b"an unrelated attachment"}
    original_flush = ArchiveBlobPublisher.flush

    def excise_then_flush(self: ArchiveBlobPublisher) -> object:
        with sqlite3.connect(tmp_path / "source.db") as excision:
            excision.execute(
                "INSERT OR IGNORE INTO excised_content (removed_hash, hash_kind, reason, actor, excised_at_ms) "
                "VALUES (?, 'blob_hash', 'operator excision', 'operator', 1000)",
                (raced_hash,),
            )
        return original_flush(self)

    monkeypatch.setattr(ArchiveBlobPublisher, "flush", excise_then_flush)

    result = _converge(
        index, source, archive_root=tmp_path, download_into=_into(lambda file_id: payloads[file_id]), limit=10
    )

    assert result.excised == 1
    assert result.acquired == 1
    statuses = sorted(str(row[0]) for row in index.execute("SELECT acquisition_status FROM attachments"))
    assert statuses == ["acquired", "unavailable"]
    ref_hashes = {bytes(row["blob_hash"]) for row in source.execute("SELECT blob_hash FROM blob_refs")}
    assert raced_hash not in ref_hashes
    index.close()
    source.close()


def test_an_excision_committed_after_the_flush_marks_the_attachment_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An excision landing after a successful flush still ends the row ``unavailable``.

    Anti-vacuity (Codex P2, #5696): consult only the flush's own refusals and
    the reference writer meets the ledger entry and aborts the pass.
    """
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_fixture_index_session(index, _session("raced-one", file_id="drive-raced"), raw_id="raced-raw")
    write_fixture_index_session(index, _session("kept-one", file_id="drive-kept"), raw_id="kept-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "raced-raw", "kept-raw")
    raced_payload = b"bytes excised while the pass was downloading"
    raced_hash = hashlib.sha256(raced_payload).digest()
    payloads = {"drive-raced": raced_payload, "drive-kept": b"an unrelated attachment"}
    original_flush = ArchiveBlobPublisher.flush

    def excise_then_flush(self: ArchiveBlobPublisher) -> object:
        published = original_flush(self)
        with sqlite3.connect(tmp_path / "source.db") as excision:
            excision.execute(
                "INSERT OR IGNORE INTO excised_content (removed_hash, hash_kind, reason, actor, excised_at_ms) "
                "VALUES (?, 'blob_hash', 'operator excision', 'operator', 1000)",
                (raced_hash,),
            )
        return published

    monkeypatch.setattr(ArchiveBlobPublisher, "flush", excise_then_flush)

    result = _converge(
        index, source, archive_root=tmp_path, download_into=_into(lambda file_id: payloads[file_id]), limit=10
    )

    assert result.excised == 1
    assert result.acquired == 1
    statuses = sorted(str(row[0]) for row in index.execute("SELECT acquisition_status FROM attachments"))
    assert statuses == ["acquired", "unavailable"]
    ref_hashes = {bytes(row["blob_hash"]) for row in source.execute("SELECT blob_hash FROM blob_refs")}
    assert raced_hash not in ref_hashes
    index.close()
    source.close()
