"""Accepted retained frontiers reuse their owned bytes without another file."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.enums import Provider, ValidationMode
from polylogue.schemas import observation_spill
from polylogue.schemas.validator import validate_retained_document
from polylogue.sources.revision_backfill import RetainedPreparationRetryableError, _retained_validation_input
from polylogue.storage.blob_store import BlobStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.unit.schemas.test_retained_validation import _registry
from tests.unit.storage.test_raw_convergence import _admit, _codex_conversation_bytes, _derive


@pytest.mark.parametrize("text_size", [0, 1100000])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_accepted_prefix_validation_matches_materialized_input(tmp_path: Path, text_size: int, encoding: str) -> None:
    prefix = (json.dumps({"type": "message", "kind": 1, "text": "x" * text_size}) + "\n").encode(encoding)
    original = tmp_path / "retained.jsonl"
    original.write_bytes(prefix + '{"unfinished":'.encode(encoding))
    baseline = tmp_path / "reference.jsonl"
    baseline.write_bytes(prefix)
    registry = _registry(tmp_path, {"type": "integer"})

    def validate(path: Path, extent: int | None) -> object:
        with _retained_validation_input(path, extent) as (validation_path, accepted_prefix_size):
            assert validation_path == path
            return validate_retained_document(
                Provider.CLAUDE_CODE,
                validation_path,
                accepted_prefix_size=accepted_prefix_size,
                mode=ValidationMode.STRICT,
                raw_id="neutral-raw",
                revision_sha256="a" * 64,
                evidence_id="neutral-raw",
                source_path="retained.jsonl",
                jsonl=True,
                registry=registry,
                signature_directory=tmp_path,
            )

    assert validate(original, len(prefix)) == validate(baseline, None)
    assert not list(tmp_path.glob("validation-*.jsonl"))
    assert original.read_bytes() == prefix + '{"unfinished":'.encode(encoding)


def test_accepted_prefix_reader_rewinds_and_hides_the_tail(tmp_path: Path) -> None:
    path = tmp_path / "retained.jsonl"
    path.write_bytes(b"accepted-tail")
    with observation_spill._document_bytes(path, 8) as reader:
        assert reader.read(4) == b"acce"
        assert reader.seek(0) == 0
        assert reader.read() == b"accepted"
        assert reader.read(1) == b""
        assert reader.seek(-3, os.SEEK_END) == 5
        assert reader.read() == b"ted"
        with pytest.raises(ValueError):
            reader.seek(9)
    assert path.read_bytes() == b"accepted-tail"


@pytest.mark.parametrize("fault", ["negative", "short", "truncate", "replace", "cancel"])
def test_accepted_prefix_faults_are_retryable_and_close_the_physical_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    prefix = b'{"type":"message","kind":1}\n'
    path = tmp_path / "retained.jsonl"
    path.write_bytes(prefix + b"partial")
    extent = -1 if fault == "negative" else len(prefix)
    if fault == "short":
        path.write_bytes(prefix[:-1])
    registry = _registry(tmp_path, {"type": "integer"})
    handles: list[io.FileIO] = []
    original_read = observation_spill._AcceptedPrefixReader.read

    def read(reader: observation_spill._AcceptedPrefixReader, size: int = -1) -> bytes:
        if not handles:
            handles.append(reader.source)
            if fault == "truncate":
                path.write_bytes(prefix[:-1])
            elif fault == "replace":
                replacement = path.with_suffix(".replacement")
                replacement.write_bytes(prefix + b"partial")
                os.replace(replacement, path)
            elif fault == "cancel":
                raise DaemonOperationCancelled("neutral accepted-prefix cancellation")
        return original_read(reader, size)

    monkeypatch.setattr(observation_spill._AcceptedPrefixReader, "read", read)
    expected = DaemonOperationCancelled if fault == "cancel" else RetainedPreparationRetryableError
    with pytest.raises(expected), _retained_validation_input(path, extent) as (validation_path, accepted_prefix_size):
        validate_retained_document(
            Provider.CLAUDE_CODE,
            validation_path,
            accepted_prefix_size=accepted_prefix_size,
            mode=ValidationMode.STRICT,
            raw_id="neutral-raw",
            revision_sha256="a" * 64,
            evidence_id="neutral-raw",
            source_path="retained.jsonl",
            jsonl=True,
            registry=registry,
            signature_directory=tmp_path,
        )
    assert all(handle.closed for handle in handles)
    assert not list(tmp_path.glob("validation-*.jsonl"))


def test_zero_accepted_prefix_keeps_incomplete_original_and_clean_census(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    payload = b'{"type":"session_meta","payload":{"id":"unfinished'
    raw_id = _admit(tmp_path, (), provider=Provider.CODEX, path="session.jsonl", payload=payload)
    original = BlobStore(tmp_path / "blob").blob_path(hashlib.sha256(payload).hexdigest())
    with observation_spill.StreamedJSONDocument(original, jsonl=True, accepted_prefix_size=0) as records:
        assert records == []
    report = _derive(tmp_path)
    assert report.failed == 0, report.outcomes
    with closing(sqlite3.connect(tmp_path / "source.db")) as connection:
        assert connection.execute(
            "SELECT status,member_count FROM raw_membership_census WHERE raw_id=?", (raw_id,)
        ).fetchone() == ("non_session", 0)
        assert connection.execute("SELECT parse_error FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone() == (
            None,
        )
    assert original.read_bytes() == payload


def test_retained_owner_validates_a_giant_accepted_prefix_without_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap_archive_root(tmp_path)
    payload = _codex_conversation_bytes("bounded-prefix", "x" * 1100000) + b'{"unfinished":'
    raw_id = _admit(tmp_path, (), provider=Provider.CODEX, path="session.jsonl", payload=payload)
    original_mkstemp = tempfile.mkstemp

    def mkstemp(
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | os.PathLike[str] | None = None,
        text: bool = False,
    ) -> tuple[int, str]:
        assert prefix != "validation-", "accepted validation must not copy the prefix"
        return original_mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)

    monkeypatch.setattr(tempfile, "mkstemp", mkstemp)
    report = _derive(tmp_path)
    assert report.failed == 0, report.outcomes
    with closing(sqlite3.connect(tmp_path / "index.db")) as connection:
        assert connection.execute("SELECT native_id FROM sessions WHERE raw_id=?", (raw_id,)).fetchall() == [
            ("bounded-prefix",)
        ]
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)


def test_zero_accepted_prefix_honors_cancellation_before_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "unfinished.jsonl"
    path.write_bytes(b'{"unfinished":')

    def cancelled() -> None:
        raise DaemonOperationCancelled("neutral zero-prefix cancellation")

    monkeypatch.setattr(observation_spill, "check_compute_cancelled", cancelled)
    with pytest.raises(DaemonOperationCancelled), observation_spill._document_bytes(path, 0):
        pytest.fail("cancelled zero-prefix work cannot expose a reader")
    assert path.read_bytes() == b'{"unfinished":'
