"""Logical paths of machine-ingest inputs frozen from a staged import."""

from __future__ import annotations

from pathlib import Path

from polylogue.operations.ingest_inputs import discover_ingest_input_spool, retain_input_page
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
    staged = tmp_path / "import-staging" / "account"
    (staged / "nested").mkdir(parents=True)
    (staged / "conversations.json").write_text("[]")
    (staged / "nested" / "chat.jsonl").write_text("{}\n")

    assert _retain(staged, "/exports/account", tmp_path) == {
        ("conversations.json", "/exports/account/conversations.json"),
        ("nested/chat.jsonl", "/exports/account/nested/chat.jsonl"),
    }


def test_a_file_input_is_keyed_on_its_source_path_or_its_physical_path(tmp_path: Path) -> None:
    staged = tmp_path / "import-staging" / "session.jsonl"
    staged.parent.mkdir(parents=True)
    staged.write_text("{}\n")

    assert _retain(staged, "/exports/session.jsonl", tmp_path) == {("input:0", "/exports/session.jsonl")}
    assert _retain(staged, None, tmp_path) == {("input:0", str(staged))}
