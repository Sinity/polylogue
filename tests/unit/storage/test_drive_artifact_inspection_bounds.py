"""Complete candidacy preserves large Drive inputs without whole-document decoding."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.core.enums import ArtifactSupportStatus, Origin, Provider
from polylogue.sources import origin_specs
from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.runtime import RawSessionRecord

#: Comfortably past the 64 KB classification prefix and past the ceiling this
#: regression is about, while staying small enough to build in a test.
_OVERSIZED_DOCUMENT_BYTES = 9 * 1024 * 1024


@pytest.fixture
def blob_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[BlobStore]:
    root = tmp_path / "blobs"
    store = BlobStore(root)
    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: root)
    monkeypatch.setattr("polylogue.storage.blob_store.blob_store_root", lambda: root, raising=False)
    reset_blob_store()
    yield store
    reset_blob_store()


def _chunked_prompt_export(path: Path, *, min_bytes: int) -> None:
    """Write a synthetic AI Studio export of at least ``min_bytes``.

    The shape mirrors the real ``{runSettings, systemInstruction,
    chunkedPrompt}`` export; the content is generated filler, so no operator
    conversation reaches the repository.
    """
    filler = "synthetic drive turn body. " * 512
    chunks: list[dict[str, object]] = []
    written = 0
    while written < min_bytes:
        index = len(chunks)
        chunks.append(
            {
                "text": f"turn {index}: {filler}",
                "role": "user" if index % 2 == 0 else "model",
                "isThought": False,
            }
        )
        written += len(filler) + 64
    payload = {
        "runSettings": {"temperature": 1.0, "topP": 0.95, "model": "models/synthetic"},
        "systemInstruction": {},
        "chunkedPrompt": {"chunks": chunks, "pendingInputs": []},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert path.stat().st_size >= min_bytes


def _record(store: BlobStore, path: Path, *, source_path: str) -> RawSessionRecord:
    blob_hash, blob_size = store.write_from_path(path)
    return RawSessionRecord(
        raw_id=f"aistudio-drive:{path.name}",
        blob_hash=blob_hash,
        payload_provider=Provider.GEMINI,
        source_name=Provider.GEMINI.value,
        source_path=source_path,
        canonical_source_path=source_path,
        blob_size=blob_size,
        acquired_at="2026-09-05T00:00:00+00:00",
    )


def test_large_valid_drive_export_retains_complete_schema_evidence(
    blob_store: BlobStore,
    tmp_path: Path,
) -> None:
    export = tmp_path / "Synthetic_Conversation-0123456789abcdef.json"
    _chunked_prompt_export(export, min_bytes=_OVERSIZED_DOCUMENT_BYTES)

    observation = inspect_raw_artifact(_record(blob_store, export, source_path=f"/drive-cache/gemini/{export.name}"))

    assert observation.artifact_kind == "session_document"
    assert observation.support_status is ArtifactSupportStatus.SUPPORTED_PARSEABLE
    assert observation.parse_as_session
    assert observation.schema_eligible
    assert observation.decode_error is None


def test_large_document_candidacy_never_reads_the_whole_blob_into_memory(
    blob_store: BlobStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The complete projected reader must reach EOF without an eager fallback."""
    export = tmp_path / "Synthetic_Conversation-fedcba9876543210.json"
    _chunked_prompt_export(export, min_bytes=_OVERSIZED_DOCUMENT_BYTES)

    def refuse_eager_read(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("whole-document allocation")

    monkeypatch.setattr(blob_store, "read_all", refuse_eager_read)

    observation = inspect_raw_artifact(_record(blob_store, export, source_path=f"/drive-cache/gemini/{export.name}"))

    assert observation.artifact_kind == "session_document"
    assert observation.parse_as_session
    assert observation.decode_error is None


def _applet_access_log(path: Path) -> None:
    """Write an applet access log in AI Studio's own wire shape.

    The nested ``source.drive`` reference is load-bearing: it is the only
    field that puts an entry past ``is_scalarish``'s depth bound, and so the
    only reason the real document misses ``looks_metadataish_dict`` and needs
    the path rule. A fixture that flattens it classifies ``metadata_document``
    on the heuristic alone, with or without the rule.
    """
    path.write_text(
        json.dumps(
            {
                "applets": [
                    {
                        "lastAccessTime": "2026-09-05T00:00:00.000000Z",
                        "firstAccessTime": "2026-09-01T00:00:00.000000Z",
                        "source": {"drive": {"resourceId": "synthetic-resource", "revisionId": "synthetic-revision"}},
                        "name": "Synthetic Applet",
                        "description": "A synthetic applet entry standing in for the real access log.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_applet_access_log_is_a_declared_non_session_document(
    blob_store: BlobStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AI Studio's applet log shares the export folder and carries no turn."""
    log = tmp_path / "applet_access_history.json"
    _applet_access_log(log)
    record = _record(blob_store, log, source_path=f"/drive-cache/gemini/{log.name}")

    observation = inspect_raw_artifact(record)

    assert observation.artifact_kind == "metadata_document"
    assert observation.support_status is ArtifactSupportStatus.RECOGNIZED_UNPARSED

    monkeypatch.setattr(
        origin_specs,
        "ORIGIN_SPECS",
        tuple(
            replace(spec, artifact_rules=()) if spec.origin is Origin.AISTUDIO_DRIVE else spec
            for spec in origin_specs.ORIGIN_SPECS
        ),
    )

    assert inspect_raw_artifact(record).artifact_kind == "unknown"


def test_retained_artifact_inspection_propagates_mid_read_compute_cancellation(
    blob_store: BlobStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    from typing import Any

    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.core.compute_cancel import compute_cancel

    export = tmp_path / "Synthetic_Cancellation-0123456789abcdef.json"
    _chunked_prompt_export(export, min_bytes=2 * 1024 * 1024)
    record = _record(blob_store, export, source_path=f"/drive-cache/gemini/{export.name}")
    retained = blob_store.blob_path(record.blob_hash or record.raw_id)
    cancelled = threading.Event()
    original_open = Path.open
    handles: list[Any] = []
    reads = 0

    class CancelAfterRead:
        def __init__(self, source: Any) -> None:
            self.source = source

        def __getattr__(self, name: str) -> Any:
            return getattr(self.source, name)

        def __enter__(self) -> CancelAfterRead:
            return self

        def __exit__(self, *args: Any) -> Any:
            return self.source.__exit__(*args)

        def read(self, size: int = -1) -> bytes:
            nonlocal reads
            data: bytes = self.source.read(size)
            if len(data) >= 65536:
                reads += 1
                cancelled.set()
            return data

    def open_retained(path: Path, *args: Any, **kwargs: Any) -> Any:
        source = original_open(path, *args, **kwargs)
        if path == retained:
            handles.append(source)
            return CancelAfterRead(source)
        return source

    token = compute_cancel.set(cancelled)
    try:
        with monkeypatch.context() as controlled:
            controlled.setattr(Path, "open", open_retained)
            with pytest.raises(DaemonOperationCancelled):
                inspect_raw_artifact(record, blob_store=blob_store)
        assert reads > 0
        assert handles and all(handle.closed for handle in handles)
        cancelled.clear()
        observation = inspect_raw_artifact(record, blob_store=blob_store)
        assert observation.parse_as_session
        assert observation.decode_error is None
    finally:
        compute_cancel.reset(token)
