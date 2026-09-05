"""Bounded classification must not report a valid Drive export as undecodable.

polylogue-1wjiw. Classification reads a 64 KB prefix, then re-reads the whole
document when that prefix is not itself valid JSON. While the re-read ceiling
sat below the size of real AI Studio exports, the prefix's mid-value decode
error was re-raised and stored as ``decode_failed`` / ``ArtifactKind.UNKNOWN``,
leaving 15 live conversations with no declared parser route.

Anti-vacuity: restoring a ceiling below the document's size turns
``test_large_valid_drive_export_classifies_as_a_session_document`` red, and
dropping the applet path rule from the AI Studio ``OriginSpec`` turns
``test_applet_access_log_is_a_declared_non_session_document`` red.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.core.enums import ArtifactSupportStatus, Provider
from polylogue.storage.artifacts import inspection
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
        blob_size=blob_size,
        acquired_at="2026-09-05T00:00:00+00:00",
    )


def test_large_valid_drive_export_classifies_as_a_session_document(
    blob_store: BlobStore,
    tmp_path: Path,
) -> None:
    export = tmp_path / "Synthetic_Conversation-0123456789abcdef.json"
    _chunked_prompt_export(export, min_bytes=_OVERSIZED_DOCUMENT_BYTES)

    observation = inspect_raw_artifact(_record(blob_store, export, source_path=f"/drive-cache/gemini/{export.name}"))

    assert observation.artifact_kind == "session_document"
    assert observation.support_status is ArtifactSupportStatus.SUPPORTED_PARSEABLE
    assert observation.decode_error is None


def test_a_ceiling_below_the_document_reports_it_undecodable(
    blob_store: BlobStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The controlled mutation the fix removes: too low a ceiling loses the document."""
    export = tmp_path / "Synthetic_Conversation-fedcba9876543210.json"
    _chunked_prompt_export(export, min_bytes=_OVERSIZED_DOCUMENT_BYTES)
    monkeypatch.setattr(inspection, "_FULL_JSON_INSPECTION_MAX_BYTES", 8 * 1024 * 1024)

    observation = inspect_raw_artifact(_record(blob_store, export, source_path=f"/drive-cache/gemini/{export.name}"))

    assert observation.artifact_kind == "unknown"
    assert observation.support_status is ArtifactSupportStatus.DECODE_FAILED


def test_applet_access_log_is_a_declared_non_session_document(
    blob_store: BlobStore,
    tmp_path: Path,
) -> None:
    """AI Studio's applet log shares the export folder and carries no turn."""
    log = tmp_path / "applet_access_history.json"
    log.write_text(
        json.dumps({"applets": [{"id": "synthetic-applet", "lastAccessed": "2026-09-05T00:00:00Z"}]}),
        encoding="utf-8",
    )

    observation = inspect_raw_artifact(_record(blob_store, log, source_path=f"/drive-cache/gemini/{log.name}"))

    assert observation.artifact_kind == "metadata_document"
    assert observation.support_status is ArtifactSupportStatus.RECOGNIZED_UNPARSED
