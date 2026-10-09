"""Canonical raw captures remain private until creator-owned pickup settles."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
from collections.abc import Generator, Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.enums import Provider
from polylogue.core.json import JSONValue
from polylogue.pipeline.services.acquisition_records import make_raw_record
from polylogue.pipeline.services.acquisition_streams import _drain_batch
from polylogue.sources.cursor import _ParseContext
from polylogue.sources.emitter import _SessionEmitter
from polylogue.sources.parsers.base import RawSessionData
from polylogue.sources.staged_raw_payload import StagedRawPayload
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.connection_profile import scratch_connection_context


def test_creator_verifies_staged_copy_before_queue_and_retains_shared_outputs(tmp_path: Path) -> None:
    preparation = tmp_path / "preparation"
    stage = StagedRawPayload.from_value({"selected": "exact", "float": 1e2}, directory=preparation)
    expected = stage.path.read_bytes()
    first = RawSessionData(staged_payload=stage, source_path="neutral.jsonl", source_index=0)
    second = first.model_copy()
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blobs")
    record = make_raw_record(first, Provider.CHATGPT.value, blob_store=publisher)
    assert record.blob_hash == hashlib.sha256(expected).hexdigest()
    assert record.blob_size == len(expected)
    assert not publisher._store.blob_path(record.blob_hash).exists()
    assert not publisher.source_db_path.exists()
    assert len(publisher._pending) == 1
    assert not stage.path.exists() and preparation.exists()
    assert first.staged_payload is None and first.raw_bytes == b""
    assert make_raw_record(second, Provider.CHATGPT.value, blob_store=publisher).blob_hash == record.blob_hash
    assert len(publisher._pending) == 1
    assert "staged_payload" not in second.model_dump()
    assert str(preparation) not in repr(second)
    publisher.discard_prepared(publisher._pending[0][1])


def test_creator_refuses_changed_staged_inode_and_cleans_capture(tmp_path: Path) -> None:
    stage = StagedRawPayload.from_value({"selected": "exact"}, directory=tmp_path / "preparation")
    raw = RawSessionData(staged_payload=stage, source_path="neutral.jsonl")
    stage.path.unlink()
    stage.path.write_bytes(b'{"selected":"other"}')
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blobs")
    with pytest.raises(ValueError, match="identity changed"):
        make_raw_record(raw, Provider.CHATGPT.value, blob_store=publisher)
    assert not stage.path.exists()
    assert not publisher._pending and not publisher.source_db_path.exists()


def test_cancelled_creator_copy_and_failed_worker_page_discard_private_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = StagedRawPayload.from_value({"selected": "exact"}, directory=tmp_path / "preparation")
    raw = RawSessionData(staged_payload=stage, source_path="neutral.jsonl")
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blobs")

    def cancelled(*_args: object, **_kwargs: object) -> object:
        raise DaemonOperationCancelled()

    monkeypatch.setattr(BlobStore, "prepare_from_path", cancelled)
    with pytest.raises(DaemonOperationCancelled):
        make_raw_record(raw, Provider.CHATGPT.value, blob_store=publisher)
    assert not stage.path.exists() and not publisher._pending
    page_stage = StagedRawPayload.from_value({"selected": "page"}, directory=tmp_path / "preparation")

    def failed_page() -> Generator[RawSessionData, None, None]:
        yield RawSessionData(staged_payload=page_stage, source_path="neutral.jsonl")
        raise OSError("late read failure")

    with pytest.raises(OSError, match="late read"):
        _drain_batch(failed_page(), batch_size=2)
    assert not page_stage.path.exists()


def test_raw_representation_is_exclusive(tmp_path: Path) -> None:
    stage = StagedRawPayload.from_value({"selected": "exact"}, directory=tmp_path)
    try:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            RawSessionData(staged_payload=stage, raw_bytes=b"{}", source_path="neutral.json")
        with pytest.raises(ValidationError, match="mutually exclusive"):
            RawSessionData(staged_payload=stage, blob_hash=stage.seal.sha256, source_path="neutral.json")
    finally:
        stage.discard()


@pytest.mark.parametrize("user_data", [True, False])
def test_admission_scan_keeps_nested_wire_types_without_loading_unselected_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, user_data: bool
) -> None:
    from polylogue.schemas.observation_spill import SpilledKey, StreamedJSONDocument, _ScalarTokenStore
    from polylogue.sources.parsers.base_support import _unknown_wire_type

    record = {
        "type": "message",
        "arguments" if user_data else "content": [{"kind": "future_nested"}],
        "k" * 65536: "v" * 65536,
    }
    path = tmp_path / "neutral.json"
    path.write_text(json.dumps(record))
    original_read = _ScalarTokenStore.read

    def selected(self: _ScalarTokenStore, kind: str, ordinal: int) -> JSONValue:
        size = self.connection.execute(
            "SELECT decoded_bytes FROM json_scalar_tokens WHERE kind=? AND token=?", (kind, ordinal)
        ).fetchone()[0]
        assert size < 65536
        return original_read(self, kind, ordinal)

    def selected_key(self: SpilledKey) -> str:
        name = self.small_name
        assert name is not None
        return name

    with StreamedJSONDocument(path) as payload:
        monkeypatch.setattr(_ScalarTokenStore, "read", selected)
        monkeypatch.setattr(SpilledKey, "read", selected_key)
        assert _unknown_wire_type(payload) == (None if user_data else "future_nested")


@pytest.mark.asyncio
async def test_cancelled_source_read_settles_before_discarding_worker_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.config import Source
    from polylogue.pipeline.services import acquisition as acquisition_module
    from polylogue.pipeline.services.acquisition_streams import iter_source_raw_stream
    from polylogue.sources.retained_acquisition import SourceInputRecord

    stage = StagedRawPayload.from_value({"selected": "exact"}, directory=tmp_path / "preparation")
    started = asyncio.Event()
    released = threading.Event()
    settled = threading.Event()
    loop = asyncio.get_running_loop()

    def records(*_args: object, **_kwargs: object) -> Iterator[SourceInputRecord]:
        try:
            yield SourceInputRecord(
                '["physical-file-v1",0]', RawSessionData(staged_payload=stage, source_path="neutral.jsonl")
            )
            loop.call_soon_threadsafe(started.set)
            released.wait()
        finally:
            settled.set()

    monkeypatch.setattr(acquisition_module, "iter_source_acquisition_records", records)
    stream = iter_source_raw_stream(Source(name="neutral", path=tmp_path))
    pending = asyncio.create_task(anext(stream))
    try:
        await started.wait()
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done() and stage.path.exists() and not settled.is_set()
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert settled.is_set() and not stage.path.exists()
    finally:
        released.set()
        await stream.aclose()
        stage.discard()


def test_emitter_streams_unknown_giant_key_and_string_to_canonical_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3
    from contextlib import contextmanager
    from typing import Any

    from polylogue.core.json import dumps_bytes
    from polylogue.schemas import observation_spill
    from polylogue.schemas.observation_spill import SpilledKey, _ScalarTokenStore

    class BoundedInput(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and size >= 0, "whole source read"
            return super().read(size)

    document = {
        "id": "neutral-conversation",
        "mapping": {
            "node": {
                "message": {
                    "id": "neutral-message",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["selected exact text"]},
                }
            }
        },
        "k" * (4 * 1024 * 1024): "v" * (4 * 1024 * 1024),
    }
    expected = dumps_bytes(document)
    source = json.dumps(document).encode() + b"\n"
    original_scratch = scratch_connection_context

    @contextmanager
    def small_cells(**kwargs: Any) -> Generator[sqlite3.Connection, None, None]:
        with original_scratch(**kwargs) as connection:
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 32768)
            yield connection

    monkeypatch.setattr(observation_spill, "scratch_connection_context", small_cells)
    original_read = _ScalarTokenStore.read

    def selected(self: _ScalarTokenStore, kind: str, ordinal: int) -> JSONValue:
        row = self.connection.execute(
            "SELECT decoded_bytes FROM json_scalar_tokens WHERE kind=? AND token=?", (kind, ordinal)
        ).fetchone()
        assert row[0] < 4 * 1024 * 1024, "giant scalar materialization"
        return original_read(self, kind, ordinal)

    def no_original_key(self: SpilledKey) -> str:
        name = self.small_name
        assert name is not None, "giant original key reconstruction"
        return name

    monkeypatch.setattr(_ScalarTokenStore, "read", selected)
    monkeypatch.setattr(SpilledKey, "read", no_original_key)
    context = _ParseContext(
        Provider.CHATGPT, False, "neutral.jsonl", "neutral", None, True, {}, raw_directory=tmp_path / "preparation"
    )
    emitted = list(_SessionEmitter(context).emit(BoundedInput(source), "neutral.jsonl"))
    assert len(emitted) == 1
    raw, session = emitted[0]
    assert raw is not None and raw.staged_payload is not None and raw.raw_bytes == b""
    try:
        assert raw.staged_payload.seal.sha256 == hashlib.sha256(expected).hexdigest()
        assert raw.staged_payload.seal.size == len(expected)
        assert raw.staged_payload.path.read_bytes() == expected
        assert session.messages[0].text == "selected exact text"
        assert len(list((tmp_path / "preparation").iterdir())) == 1
    finally:
        raw.staged_payload.discard()
