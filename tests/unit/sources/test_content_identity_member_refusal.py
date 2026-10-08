"""A member refused for identity is a recorded gap, not a failed acquisition."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core import content_identity
from polylogue.core.enums import Provider
from polylogue.core.raw_coordinates import zip_member_source_index
from polylogue.sources.source_acquisition import iter_source_acquisition_records
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload
from tests.infra.source_builders import acquired_payloads, live_zip_capture


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
        acquired_payloads(
            iter_source_acquisition_records(
                Source(name="chatgpt", path=source_root),
                blob_store=BlobStore(tmp_path / "archive" / "blob"),
                cursor_state=cursor_state,
            )
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
        (WatchSource("chatgpt", root, layout=export_drop_layout((".zip",))),), operation_id="identity-refusal"
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
    advances past it; record it on the ZIP path instead of the member and the
    subject no longer names the member."""
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
        with live_zip_capture(tmp_path) as (publisher, zip_inputs):
            extracted = processor._extract_source_only_zip_member_records(
                zip_path,
                blob_store=publisher,
                zip_inputs=zip_inputs,
                fallback_provider=Provider.CHATGPT,
                file_mtime="2026-09-28T00:00:00+00:00",
            )
            assert extracted is not None
            records, _bytes = extracted
        return {record.source_path for _raw_id, record in records}

    assert extract() == {f"{zip_path}:b.json"}
    debt = cursor.list_convergence_debt(stage="live_ingest_admission")
    # One record per refused member, the same shape a foreign-origin member
    # refusal takes (polylogue-ltj6c): its ordinal and name, a typed reason.
    assert [(row.subject_id, (row.last_error or "").split(":", 1)[0]) for row in debt] == [
        (f"{zip_path}:#0:a.json", "content_identity_refused")
    ]

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
        acquired_payloads(
            iter_source_acquisition_records(
                Source(name="chatgpt", path=source_root),
                blob_store=BlobStore(tmp_path / "archive" / "blob"),
                cursor_state=cursor_state,
            )
        )
    )

    blob_store = BlobStore(tmp_path / "archive" / "blob")
    acquired_ids = {json.loads(blob_store.read_all(str(record.blob_hash)))["id"] for record in records}
    assert acquired_ids == {"first", "second", "after"}
    # ``source_index`` is the canonical member coordinate of each split element.
    assert sorted(int(record.source_index or 0) for record in records) == [
        zip_member_source_index(entry_ordinal=0, split_index=index) for index in (0, 1, 3)
    ]
    failures = cursor_state.get("failed_files", [])
    assert any(
        failure["path"] == f"{zip_path}:conversations.json" and "object key" in failure["error"] for failure in failures
    )

    # Replay yields every acquired element beside the gap, then names the gap.
    from polylogue.sources.source_acquisition_components import (
        ReplayedZipRevision,
        ZipEntryReadContext,
        replay_zip_entry_acquisition_revisions,
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
        replayed: list[ReplayedZipRevision] = []
        with pytest.raises(content_identity.ContentIdentityRefusal):
            for unit in replay_zip_entry_acquisition_revisions(archive, context):
                replayed.append(unit)
    assert [unit.source_index for unit in replayed] == [0, 1, 3]


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


def test_a_refused_grouped_zip_member_leaves_no_queued_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: skip the refused grouped member without discarding its
    queued blob and the next flush publishes bytes nothing references."""
    from polylogue.sources.decoder_zip import process_zip
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 64)
    zip_path = tmp_path / "codex.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("rollout.jsonl", b'{"n": 1.' + b"2" * 200 + b"}\n")
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    list(
        process_zip(
            zip_path,
            provider_hint=Provider.CODEX,
            should_group=True,
            file_mtime=None,
            capture_raw=True,
            cursor_state={},
            blob_store=publisher,
        )
    )
    assert not publisher.has_pending
    assert not any(path.is_file() for path in (tmp_path / "blob").rglob("*"))


def _split_member_with_a_refused_element() -> bytes:
    first, second, after = (json.dumps({"id": name, "mapping": {}}).encode() for name in ("first", "second", "after"))
    refused = b'{"id": "refused", "mapping": {}, "' + b"k" * 200 + b'": 1}'
    return b"[" + b", ".join((first, second, refused, after)) + b"]"


def test_a_refused_split_element_is_a_baseline_fault_beside_its_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: swallow the split refusal on the revision replay and the
    baseline accepts the siblings with no typed gap for the refused element."""
    from polylogue.sources.live.production_baseline import capture_production_source_baseline
    from polylogue.sources.live.watcher import WatchSource

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 16)
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 64)
    root = tmp_path / "chatgpt"
    root.mkdir()
    zip_path = root / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("conversations.json", _split_member_with_a_refused_element())

    baseline = capture_production_source_baseline(
        (WatchSource("chatgpt", root, layout=export_drop_layout((".zip",))),), operation_id="split-refusal"
    )
    member = f"{zip_path}:conversations.json"
    faults = {row.path: row.reason for row in baseline.decisions if row.disposition == "fault"}
    assert faults[member].startswith("content_identity_refused:")
    assert sum(1 for row in baseline.accepted if row.path == member) == 3


def test_the_parse_route_captures_elements_after_a_refused_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: let the refusal leave the payload loop and the session
    after the refused element is never yielded."""
    from polylogue.sources.decoder_zip import process_zip

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 16)
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 64)
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("conversations.json", _split_member_with_a_refused_element())
    yielded = list(
        process_zip(
            zip_path,
            provider_hint=Provider.CHATGPT,
            should_group=False,
            file_mtime=None,
            capture_raw=True,
            cursor_state={},
            blob_store=BlobStore(tmp_path / "blob"),
        )
    )
    source_indexes = sorted(int(raw.source_index or 0) for raw, _session in yielded if raw is not None)
    assert source_indexes == [zip_member_source_index(entry_ordinal=0, split_index=index) for index in (0, 1, 3)]


def test_member_revision_hashes_through_the_identity_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The baseline revision needs no member-sized scratch copy.

    Anti-vacuity: spool the decompressed member to a temporary file before
    its identity pass and the patched ``TemporaryFile`` fails the replay.
    """
    import hashlib
    import tempfile
    import zipfile

    from polylogue.sources.source_acquisition_components import _stream_member_revision

    payload = b'{"mapping": {"a": [1, 2.5, "text"]}, "title": "t"}'
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("conversations.json", payload)

    def no_scratch(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("staged the member in scratch")

    monkeypatch.setattr(tempfile, "TemporaryFile", no_scratch)
    with zipfile.ZipFile(archive) as zf:
        revision = _stream_member_revision(zf, zf.getinfo("conversations.json"), None, None)
    assert revision.revision == hashlib.sha256(payload).hexdigest()
    assert revision.size_bytes == len(payload)
