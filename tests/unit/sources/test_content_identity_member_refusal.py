"""A member refused for identity is a recorded gap, not a failed acquisition."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core import content_identity
from polylogue.core.enums import Provider
from polylogue.sources.source_acquisition import iter_source_raw_data
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload


def test_a_refused_member_is_recorded_and_its_siblings_are_acquired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: let ``ContentIdentityRefusal`` escape the member loop and
    the ZIP yields nothing and records no named gap for the refused member."""
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 64)
    source_root = tmp_path / "inbox"
    source_root.mkdir()
    zip_path = source_root / "export.zip"
    kept = json.dumps([{"id": "kept", "mapping": {}}]).encode()
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("a.json", b'{"n": 1.' + b"2" * 200 + b"}")
        archive.writestr("b.json", kept)

    cursor_state: CursorStatePayload = {}
    records = list(
        iter_source_raw_data(
            Source(name="chatgpt", path=source_root),
            blob_store=BlobStore(tmp_path / "archive" / "blob"),
            cursor_state=cursor_state,
        )
    )

    acquired = {record.source_path for record in records}
    assert f"{zip_path}:b.json" in acquired
    assert f"{zip_path}:a.json" not in acquired
    failures = cursor_state.get("failed_files", [])
    assert any(failure["path"] == f"{zip_path}:a.json" and "number token" in failure["error"] for failure in failures)


def test_a_refused_member_is_a_typed_production_baseline_fault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: let ``ContentIdentityRefusal`` escape ``_archive_members``
    and capturing the baseline raises instead of naming the member fault."""
    from polylogue.sources.live.production_baseline import capture_production_source_baseline
    from polylogue.sources.live.watcher import WatchSource

    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 64)
    root = tmp_path / "chatgpt"
    root.mkdir()
    zip_path = root / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("a.json", b'{"n": 1.' + b"2" * 200 + b"}")
        archive.writestr("b.json", json.dumps([{"id": "kept", "mapping": {}}]).encode())

    baseline = capture_production_source_baseline(
        (WatchSource("chatgpt", root, suffixes=(".zip",)),), operation_id="identity-refusal"
    )

    faults = {row.path: row.reason for row in baseline.decisions if row.disposition == "fault"}
    assert faults.keys() == {f"{zip_path}:a.json"}
    assert faults[f"{zip_path}:a.json"].startswith("content_identity_refused:")
    assert f"{zip_path}:b.json" in {row.path for row in baseline.accepted}


def test_a_live_zip_refusal_is_recorded_debt_until_the_zip_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: log and drop the refused member (the previous handling)
    and no ``live_ingest_admission`` debt names the gap once the ZIP's cursor
    advances past it."""
    from types import SimpleNamespace
    from typing import Any, cast

    from polylogue.core.enums import Provider
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.watcher import WatchSource

    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 64)
    root = tmp_path / "inbox"
    root.mkdir()
    zip_path = root / "export.zip"
    kept = json.dumps([{"id": "kept", "mapping": {}}]).encode()
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("a.json", b'{"n": 1.' + b"2" * 200 + b"}")
        archive.writestr("b.json", kept)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="chatgpt", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    def extract() -> set[str]:
        records, _bytes = processor._extract_zip_member_records(
            zip_path,
            blob_store=BlobStore(tmp_path / "blob"),
            fallback_provider=Provider.CHATGPT,
            file_mtime="2026-09-28T00:00:00+00:00",
        )
        return {record.source_path for _raw_id, record in records}

    assert extract() == {f"{zip_path}:b.json"}
    debt = cursor.list_convergence_debt(stage="live_ingest_admission")
    assert [(row.subject_id, "a.json" in (row.last_error or "")) for row in debt] == [(str(zip_path), True)]

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("b.json", kept)
    assert extract() == {f"{zip_path}:b.json"}
    assert cursor.list_convergence_debt(stage="live_ingest_admission") == []


def test_a_refused_split_element_does_not_drop_the_elements_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: let the refusal escape the split enumeration at the refused
    element and the session after it is never acquired."""
    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 16)
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 64)
    source_root = tmp_path / "inbox"
    source_root.mkdir()
    zip_path = source_root / "export.zip"
    first, second, after = (json.dumps({"id": name, "mapping": {}}).encode() for name in ("first", "second", "after"))
    # Split elements are re-serialized from the decoded record, so the
    # overlong token that survives is a key.
    refused = b'{"id": "refused", "mapping": {}, "' + b"k" * 200 + b'": 1}'
    document = b"[" + b", ".join((first, second, refused, after)) + b"]"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("conversations.json", document)

    cursor_state: CursorStatePayload = {}
    records = list(
        iter_source_raw_data(
            Source(name="chatgpt", path=source_root),
            blob_store=BlobStore(tmp_path / "archive" / "blob"),
            cursor_state=cursor_state,
        )
    )

    blob_store = BlobStore(tmp_path / "archive" / "blob")
    acquired_ids = {json.loads(blob_store.read_all(record.blob_hash))["id"] for record in records}
    assert acquired_ids == {"first", "second", "after"}
    assert sorted(record.source_index for record in records) == [0, 1, 3]
    failures = cursor_state.get("failed_files", [])
    assert any(
        failure["path"] == f"{zip_path}:conversations.json" and "object key" in failure["error"] for failure in failures
    )

    # Replay keeps every acquired element as a candidate beside the gap.
    from polylogue.sources.source_acquisition_components import (
        ZipEntryReadContext,
        replay_zip_entry_acquisition_payloads,
    )

    with zipfile.ZipFile(zip_path) as archive:
        context = ZipEntryReadContext(
            source=Source(name="chatgpt", path=source_root),
            zip_path=zip_path,
            entry=archive.getinfo("conversations.json"),
            file_mtime=None,
            provider_hint=Provider.CHATGPT,
            blob_store=BlobStore(tmp_path / "replay-blob"),
        )
        replayed = list(replay_zip_entry_acquisition_payloads(archive, context))
    assert [payload.source_index for payload in replayed] == [0, 1, 3]


def test_a_refused_grouped_element_still_takes_its_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: leave the index unadvanced for a refused grouped element
    and the next session is stored at the refused element's coordinate."""
    from polylogue.sources.source_acquisition_components import SplitPayloadBuffer

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 16)
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 64)
    buffer = SplitPayloadBuffer()
    emitted = [*buffer.add(Provider.CHATGPT, b'{"id": "a"}'), *buffer.add(Provider.CHATGPT, b'{"id": "b"}')]
    assert buffer.add_grouped(Provider.CHATGPT, b'{"' + b"k" * 200 + b'": 1}') is None
    emitted.extend(buffer.add(Provider.CHATGPT, b'{"id": "c"}'))
    assert [payload.source_index for payload in emitted] == [0, 1, 3]
    assert len(buffer.refusals) == 1


def test_a_refused_preserved_member_leaves_no_queued_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: propagate the refusal without discarding the queued blob
    and the next flush reserves bytes no raw record references."""
    from polylogue.sources.source_acquisition_components import (
        ZipEntryReadContext,
        stream_preserved_zip_entry_raw_data,
    )
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 64)
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("a.json", b'{"n": 1.' + b"2" * 200 + b"}")
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    with zipfile.ZipFile(zip_path) as archive, pytest.raises(content_identity.ContentIdentityRefusal):
        stream_preserved_zip_entry_raw_data(
            archive,
            ZipEntryReadContext(
                source=Source(name="chatgpt", path=tmp_path),
                zip_path=zip_path,
                entry=archive.getinfo("a.json"),
                file_mtime=None,
                provider_hint=Provider.CHATGPT,
                blob_store=publisher,
            ),
            provider_hint=Provider.CHATGPT,
        )
    assert not publisher.has_pending
    assert not any(path.is_file() for path in (tmp_path / "blob").rglob("*"))
