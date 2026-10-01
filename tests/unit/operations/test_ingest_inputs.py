"""Logical paths of machine-ingest inputs frozen from a staged import."""

from __future__ import annotations

from pathlib import Path

from polylogue.operations.ingest_inputs import discover_ingest_input_spool, retain_input_page
from polylogue.sources.source_staging import stage_source_input
from polylogue.storage.blob_publication import ArchiveBlobPublisher


def _retain(path: Path, source_path: str | None, tmp_path: Path) -> set[tuple[str, str]]:
    spool = discover_ingest_input_spool(path, source_path=source_path, check_stop=lambda: None)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    try:
        page = retain_input_page(spool, after_coordinate=None, publisher=publisher, check_stop=lambda: None)
    finally:
        publisher.discard_pending()
        spool.unlink(missing_ok=True)
    return {(item.coordinate, item.source_path) for item in page}


def test_staged_directory_members_are_keyed_under_the_callers_path(tmp_path: Path) -> None:
    """03.F045: a staged directory import is acquired where the caller's export lives.

    ``polylogue import <dir>`` stages outside the watched inbox, so the ingest
    operation is its only route; each member keeps its relative path under the
    submitted ``source_path``.

    Anti-vacuity: refuse a directory with a ``source_path`` again and this
    raises; key members on the staged physical path and the logical paths
    name ``import-staging`` instead of ``/exports/account``.
    """
    original = tmp_path / "exports" / "account"
    (original / "nested").mkdir(parents=True)
    (original / "conversations.json").write_text("[]")
    (original / "nested" / "chat.jsonl").write_text("{}\n")
    staged = stage_source_input(original, tmp_path / "import-staging", check_stop=lambda: None)

    assert _retain(staged, str(original), tmp_path) == {
        ("conversations.json", str(original / "conversations.json")),
        ("nested/chat.jsonl", str(original / "nested" / "chat.jsonl")),
    }


def test_file_intake_requires_the_captured_original_declaration(tmp_path: Path) -> None:
    import pytest

    original = tmp_path / "exports" / "session.jsonl"
    original.parent.mkdir()
    original.write_text("{}\n")
    staged = stage_source_input(original, tmp_path / "import-staging", check_stop=lambda: None)
    assert _retain(staged, str(original), tmp_path) == {("input:0", str(original))}
    with pytest.raises(ValueError):
        _retain(staged, "/unproved/session.jsonl", tmp_path)
    assert _retain(original, None, tmp_path) == {("input:0", str(original))}


def test_machine_zip_enumeration_preserves_the_accepted_decoder_identity(tmp_path: Path) -> None:
    """Machine preparation carries its accepted decoder into coordinate and raw ID."""
    import json
    import zipfile

    from polylogue.core.raw_coordinates import captured_zip_member_raw_id
    from polylogue.operations.ingest_inputs import PreparedSourceRecord, enumerate_ingest_input
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    source = tmp_path / "export.zip"
    payload = {
        "id": "synthetic-conversation",
        "title": "Synthetic",
        "mapping": {
            "message": {
                "id": "message",
                "parent": None,
                "children": [],
                "message": {
                    "id": "message",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hello"]},
                },
            }
        },
    }
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("conversations.json", json.dumps([payload]))
    spool = discover_ingest_input_spool(source, source_path=None, check_stop=lambda: None)
    publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
    try:
        (item,) = retain_input_page(spool, after_coordinate=None, publisher=publisher, check_stop=lambda: None)
        identities: list[str] = []
        for fingerprint in ("b" * 64, "c" * 64):
            records = [
                record
                for record in enumerate_ingest_input(
                    item,
                    source_generation_id="accepted-machine",
                    enumeration_fingerprint=fingerprint,
                    publisher=publisher,
                    acquired_at_ms=1,
                    check_stop=lambda: None,
                )
                if isinstance(record, PreparedSourceRecord)
            ]
            assert len(records) == 1
            record = records[0]
            coordinate = record.plan.request.captured_zip_coordinate
            assert coordinate is not None
            assert coordinate.decoder_fingerprint == fingerprint
            assert coordinate.container_blob_hash == item.blob_hash
            assert item.captured_identity is not None
            assert coordinate.canonical_container == item.captured_identity.canonical_source_path
            assert record.plan.raw_id == captured_zip_member_raw_id(coordinate, record.plan.request.blob_hash.hex())
            identities.append(record.plan.raw_id)
        assert identities[0] != identities[1]
    finally:
        publisher.discard_pending()
        spool.unlink(missing_ok=True)
