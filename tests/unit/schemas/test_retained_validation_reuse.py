"""Owned validation results require fresh schema and exact input currency."""

from __future__ import annotations

import gzip
import json
import threading
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import IO

import pytest
from ijson.common import JSONError

from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import Provider, ValidationMode
from polylogue.core.json import JSONValue
from polylogue.schemas import observation_spill
from polylogue.schemas.packages import SchemaResolution
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.schemas.validator import RetainedValidationReuse, RetainedValidationVerdict, validate_retained_document


def _publish(root: Path, kind: str) -> Path:
    return SchemaRegistry(storage_root=root).write_schema_version(
        "claude-code",
        "v1",
        {"type": "object", "properties": {"kind": {"type": kind}}, "required": ["kind"]},
        element_kind="session_record_stream",
    )


def _assert_complete_verdict_equal(actual: RetainedValidationVerdict, expected: RetainedValidationVerdict) -> None:
    assert replace(actual, drift_observation=None) == replace(expected, drift_observation=None)
    if actual.drift_observation is None:
        assert expected.drift_observation is None
        return
    assert expected.drift_observation is not None
    left, right = actual.drift_observation, expected.drift_observation
    assert b"".join(left.unseen_key_signature.iter_utf8_chunks()) == b"".join(
        right.unseen_key_signature.iter_utf8_chunks()
    )
    assert replace(left, unseen_key_signature=right.unseen_key_signature) == right


def _validate(
    path: Path,
    registry: SchemaRegistry,
    reuse: RetainedValidationReuse,
    *,
    prefix: int | None = None,
    mode: ValidationMode = ValidationMode.STRICT,
    raw_id: str = "raw",
) -> RetainedValidationVerdict:
    return validate_retained_document(
        Provider.CLAUDE_CODE,
        path,
        mode=mode,
        raw_id=raw_id,
        revision_sha256="a" * 64,
        evidence_id=raw_id,
        source_path="records.jsonl",
        jsonl=True,
        accepted_prefix_size=prefix,
        schema_resolution=SchemaResolution(
            provider="claude-code",
            package_version="v1",
            element_kind="session_record_stream",
            exact_structure_id=None,
            bundle_scope=None,
            reason="package_default",
        ),
        schema_resolution_is_explicit=True,
        registry=registry,
        signature_directory=path.parent,
        reuse=reuse,
    )


@pytest.fixture
def validation_bodies(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = observation_spill.StreamedJSONDocument.__enter__

    def enter(document: observation_spill.StreamedJSONDocument) -> JSONValue:
        calls.append(1)
        return original(document)

    monkeypatch.setattr(observation_spill.StreamedJSONDocument, "__enter__", enter)
    return calls


@pytest.mark.parametrize("mode", [ValidationMode.ADVISORY, ValidationMode.STRICT])
def test_owned_reuse_preserves_complete_refusal_and_file_backed_drift(
    tmp_path: Path, validation_bodies: list[int], mode: ValidationMode
) -> None:
    _publish(tmp_path / "schemas", "integer")
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "message",
                "kind": "invalid",
                **{f"neutral_unused_field_{ordinal}": True for ordinal in range(400)},
            }
        )
        + "\n"
    )
    reuse = RetainedValidationReuse()
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    first = _validate(path, registry, reuse, mode=mode)
    second = _validate(path, registry, reuse, mode=mode)
    assert second is first
    assert first.invalid_count == 1 and first.error_count > 0
    assert first.strict_refusal is (mode is ValidationMode.STRICT)
    assert first.drift_observation is not None
    signature = first.drift_observation.unseen_key_signature
    assert signature.byte_count > 4096
    assert len(b"".join(signature.iter_utf8_chunks())) == signature.byte_count
    assert len(validation_bodies) == 1
    assert signature._path is not None
    signature._path.unlink()
    third = _validate(path, registry, reuse, mode=mode)
    assert third is not first
    fresh = validate_retained_document(
        Provider.CLAUDE_CODE,
        path,
        mode=mode,
        raw_id="raw",
        revision_sha256="a" * 64,
        evidence_id="raw",
        source_path="records.jsonl",
        jsonl=True,
        schema_resolution=first.schema_resolution,
        schema_resolution_is_explicit=True,
        registry=registry,
        signature_directory=path.parent,
    )
    _assert_complete_verdict_equal(third, fresh)
    assert len(validation_bodies) == 3


def test_owned_reuse_refreshes_same_version_bytes_and_local_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validation_bodies: list[int]
) -> None:
    bundled = tmp_path / "bundled"
    local = tmp_path / "local"
    monkeypatch.setattr("polylogue.schemas.runtime_registry.SCHEMA_DIR", bundled)
    schema_path = _publish(bundled, "integer")
    path = tmp_path / "records.jsonl"
    path.write_text('{"type":"message","kind":1}\n')
    registry = SchemaRegistry(storage_root=local)
    reuse = RetainedValidationReuse()

    def checked() -> RetainedValidationVerdict:
        result = _validate(path, registry, reuse)
        _assert_complete_verdict_equal(result, _validate(path, registry, RetainedValidationReuse()))
        assert _validate(path, registry, reuse) is result
        return result

    first = checked()
    assert _validate(path, registry, reuse) is first
    _publish(local, "string")
    assert checked().invalid_count == 1
    (local / "claude-code").rename(tmp_path / "retired-local")
    assert checked().invalid_count == 0
    # A same-version byte override, rather than schema publication's deliberate
    # integer/string union, must invalidate the previous exact snapshot.
    schema_path.write_bytes(
        gzip.compress(
            json.dumps(
                {
                    "type": "object",
                    "properties": {"kind": {"type": "string"}},
                    "required": ["kind"],
                }
            ).encode()
        )
    )
    assert checked().invalid_count == 1
    assert len(validation_bodies) == 8


@pytest.mark.parametrize("change", ["prefix", "replace", "bytes", "short", "malformed", "coordinate"])
def test_owned_reuse_checks_prefix_physical_input_and_coordinate(
    tmp_path: Path, validation_bodies: list[int], change: str
) -> None:
    _publish(tmp_path / "schemas", "integer")
    path = tmp_path / "records.jsonl"
    first_record = b'{"type":"message","kind":1}\n'
    payload = first_record + b'{"type":"message","kind":2}\n'
    path.write_bytes(payload)
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    reuse = RetainedValidationReuse()
    first = _validate(path, registry, reuse, prefix=len(first_record))
    assert first.sample_count == 1
    if change == "short":
        path.write_bytes(b"{")
        with pytest.raises(observation_spill.AcceptedPrefixReadError):
            _validate(path, registry, reuse, prefix=len(first_record))
        assert len(validation_bodies) == 1
        return
    if change == "malformed":
        path.write_bytes(b'{"type":"message","kind":}\n')
        with pytest.raises(JSONError):
            _validate(path, registry, reuse, prefix=path.stat().st_size)
        assert reuse._entry is None
        assert len(validation_bodies) == 2
        return
    prefix = len(first_record)
    if change == "prefix":
        prefix = len(payload)
    elif change == "replace":
        replacement = tmp_path / "replacement.jsonl"
        replacement.write_bytes(payload)
        replacement.replace(path)
    elif change == "bytes":
        path.write_bytes(payload.replace(b'"kind":1', b'"kind":3'))
    second = _validate(path, registry, reuse, prefix=prefix, raw_id="other" if change == "coordinate" else "raw")
    assert second is not first
    assert second.sample_count == (2 if change == "prefix" else 1)
    assert len(validation_bodies) == 2


def test_owned_reuse_zero_extent_and_cancellation_keep_physical_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validation_bodies: list[int]
) -> None:
    _publish(tmp_path / "schemas", "integer")
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"unfinished":')
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    reuse = RetainedValidationReuse()
    first = _validate(path, registry, reuse, prefix=0)
    assert _validate(path, registry, reuse, prefix=0) is first
    assert first.sample_count == 0 and not first.strict_refusal
    assert len(validation_bodies) == 1
    original_open = Path.open
    readers: list[IO[bytes]] = []

    def open_input(owner: Path, mode: str = "r", buffering: int = -1) -> IO[bytes]:
        assert mode == "rb"
        result = original_open(owner, "rb", buffering=buffering)
        if owner == path:
            readers.append(result)
        return result

    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("neutral owned cancellation")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_input)
        patch.setattr("polylogue.schemas.validator.check_compute_cancelled", cancel)
        with pytest.raises(RuntimeError, match="neutral owned cancellation"):
            _validate(path, registry, reuse, prefix=0)
    assert readers and all(reader.closed for reader in readers)
    assert _validate(path, registry, reuse, prefix=0) is first


def test_shared_provider_snapshot_allows_independent_validation_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _publish(tmp_path / "schemas", "integer")
    paths = (tmp_path / "first.jsonl", tmp_path / "second.jsonl")
    for path in paths:
        path.write_text('{"type":"message","kind":1}\n')
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    original = observation_spill.StreamedJSONDocument.__enter__
    lock = threading.Lock()
    both_started = threading.Event()
    active = 0
    peak = 0

    def body(document: observation_spill.StreamedJSONDocument) -> JSONValue:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                both_started.set()
        # An observation window makes the serialization regression terminate.
        # This is fixture synchronization, never a product input deadline.
        both_started.wait(5)
        try:
            return original(document)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(observation_spill.StreamedJSONDocument, "__enter__", body)
    adapter = BoundedComputeAdapter(max_workers=4)
    try:
        submitted = [
            adapter.submit(
                partial(_validate, path, registry, RetainedValidationReuse(), raw_id=f"raw-{ordinal}"),
                admission_class="incremental-background",
                estimated_bytes=path.stat().st_size,
            )
            for ordinal, path in enumerate(paths)
        ]
        results = [operation.future.result() for operation in submitted]
        assert all(result.invalid_count == 0 for result in results)
        assert peak == 2
    finally:
        assert adapter.close(join_timeout_s=10) == ()


def test_completed_worker_snapshot_requires_fresh_schema_before_rebind(
    tmp_path: Path, validation_bodies: list[int]
) -> None:
    root = tmp_path / "schemas"
    _publish(root, "integer")
    path = tmp_path / "records.jsonl"
    path.write_text('{"type":"message","kind":1}\n')
    registry = SchemaRegistry(storage_root=root)
    with registry.current_provider_snapshot(Provider.CLAUDE_CODE) as snapshot:
        reader = snapshot.reader()
    reuse = RetainedValidationReuse()
    worker = _validate(path, reader, reuse)
    assert not worker.strict_refusal
    _publish(root, "string")
    reader.clear_cache()
    with pytest.raises(ValueError, match="fixed schema snapshot"):
        reader.write_schema_version("claude-code", "v2", {"type": "object"})
    assert _validate(path, reader, reuse) is worker
    worker_outcome = reuse.outcome
    assert worker_outcome == "hit"
    rebound = _validate(path, SchemaRegistry(storage_root=root), reuse)
    assert rebound is not worker
    assert reuse.outcome == "schema_changed"
    fresh = _validate(path, SchemaRegistry(storage_root=root), RetainedValidationReuse())
    _assert_complete_verdict_equal(rebound, fresh)
    assert len(validation_bodies) == 3
