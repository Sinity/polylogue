"""Owned validation results require fresh schema and exact input currency."""

from __future__ import annotations

import gzip
import json
import threading
from dataclasses import replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import IO, BinaryIO

import pytest
from ijson.common import JSONError

from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import Provider, ValidationMode
from polylogue.core.json import JSONValue
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate, MemberAddressingMode
from polylogue.schemas import observation_spill
from polylogue.schemas.packages import SchemaResolution
from polylogue.schemas.runtime_registry import ProviderSchemaSnapshot, PublicSchemaDocument, SchemaRegistry
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
    source_path: str = "records.jsonl",
    coordinate: CapturedZipMemberCoordinate | None = None,
    resolution: SchemaResolution | None = None,
    explicit: bool = True,
    jsonl: bool = True,
    revision_sha256: str = "a" * 64,
) -> RetainedValidationVerdict:
    return validate_retained_document(
        Provider.CLAUDE_CODE,
        path,
        mode=mode,
        raw_id=raw_id,
        revision_sha256=revision_sha256,
        evidence_id=raw_id,
        source_path=source_path,
        jsonl=jsonl,
        accepted_prefix_size=prefix,
        captured_zip_coordinate=coordinate,
        schema_resolution=resolution
        or SchemaResolution(
            provider="claude-code",
            package_version="v1",
            element_kind="session_record_stream",
            exact_structure_id=None,
            bundle_scope=None,
            reason="package_default",
        ),
        schema_resolution_is_explicit=explicit,
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
    _assert_complete_verdict_equal(second, first)
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
        _assert_complete_verdict_equal(_validate(path, registry, reuse), result)
        return result

    first = checked()
    _assert_complete_verdict_equal(_validate(path, registry, reuse), first)
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
    assert len(validation_bodies) == (1 if change == "coordinate" else 2)
    assert second.raw_id == ("other" if change == "coordinate" else "raw")
    assert second.evidence_id == second.raw_id


def test_owned_reuse_zero_extent_and_cancellation_keep_physical_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validation_bodies: list[int]
) -> None:
    _publish(tmp_path / "schemas", "integer")
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"unfinished":')
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    reuse = RetainedValidationReuse()
    first = _validate(path, registry, reuse, prefix=0)
    _assert_complete_verdict_equal(_validate(path, registry, reuse, prefix=0), first)
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
    _assert_complete_verdict_equal(_validate(path, registry, reuse, prefix=0), first)


@pytest.mark.parametrize("fixed", [False, True])
def test_shared_provider_snapshot_allows_independent_validation_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed: bool
) -> None:
    _publish(tmp_path / "schemas", "integer")
    paths = (tmp_path / "first.jsonl", tmp_path / "second.jsonl")
    for path in paths:
        path.write_text('{"type":"message","kind":1}\n')
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    if fixed:
        with registry.current_provider_snapshot(Provider.CLAUDE_CODE) as snapshot:
            registry = snapshot.reader()
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
    _assert_complete_verdict_equal(_validate(path, reader, reuse), worker)
    worker_outcome = reuse.outcome
    assert worker_outcome == "hit"
    rebound = _validate(path, SchemaRegistry(storage_root=root), reuse)
    assert rebound is not worker
    assert reuse.outcome == "schema_changed"
    fresh = _validate(path, SchemaRegistry(storage_root=root), RetainedValidationReuse())
    _assert_complete_verdict_equal(rebound, fresh)
    assert len(validation_bodies) == 3


@pytest.mark.parametrize("change", ["none", "alias_bytes", "retired_input"])
def test_body_evidence_certifies_alias_and_projects_complete_coordinates(
    tmp_path: Path, validation_bodies: list[int], change: str
) -> None:
    _publish(tmp_path / "schemas", "integer")
    original = tmp_path / "original.jsonl"
    alias_directory = tmp_path / "canonical"
    alias_directory.mkdir()
    alias = alias_directory / "retained.jsonl"
    payload = (
        json.dumps(
            {
                "type": "message",
                "kind": "invalid",
                **{f"neutral_unused_field_{ordinal}": True for ordinal in range(400)},
            }
        )
        + "\n"
    ).encode()
    original.write_bytes(payload)
    alias.write_bytes(payload)
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    reuse = RetainedValidationReuse()
    first = _validate(original, registry, reuse, raw_id="first")
    assert first.drift_observation is not None
    signature = first.drift_observation.unseen_key_signature
    assert signature._path is not None and signature._path.parent == tmp_path
    if change == "alias_bytes":
        alias.write_bytes(payload.replace(b'"invalid"', b"123456789"))
    elif change == "retired_input":
        original.unlink()
    second = _validate(alias, registry, reuse, raw_id="second")
    assert second.raw_id == second.evidence_id == "second"
    assert second.drift_observation is not None and second.drift_observation.raw_id == "second"
    assert second.drift_observation.native_id_example == "records.jsonl"
    assert len(validation_bodies) == (1 if change == "none" else 2)
    assert reuse.outcome == ("hit" if change == "none" else "input_changed")
    if change == "none":
        assert second.drift_observation.unseen_key_signature is signature
    fresh = _validate(alias, registry, RetainedValidationReuse(), raw_id="second")
    _assert_complete_verdict_equal(second, fresh)
    assert len(b"".join(signature.iter_utf8_chunks())) == signature.byte_count
    reuse.clear()
    assert reuse._entry is None


@pytest.mark.parametrize("change", ["mode", "source", "zip", "resolution", "explicitness", "format"])
def test_body_selection_recipe_changes_match_complete_fresh_verdict(
    tmp_path: Path, validation_bodies: list[int], change: str
) -> None:
    _publish(tmp_path / "schemas", "integer")
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    registry.write_schema_version(
        "claude-code",
        "v2",
        {"type": "object", "properties": {"kind": {"type": "integer"}}},
        element_kind="session_record_stream",
    )
    path = tmp_path / "records.jsonl"
    path.write_text('{"type":"message","kind":"invalid"}\n')
    reuse = RetainedValidationReuse()
    _validate(path, registry, reuse)
    resolution = (
        SchemaResolution(
            provider="claude-code",
            package_version="v2",
            element_kind="session_record_stream",
            exact_structure_id=None,
            bundle_scope=None,
            reason="package_default",
        )
        if change == "resolution"
        else None
    )
    coordinate = (
        CapturedZipMemberCoordinate(
            "/neutral.zip",
            "/neutral.zip",
            "records.jsonl",
            0,
            0,
            MemberAddressingMode.WHOLE_MEMBER,
            "b" * 64,
            "c" * 64,
        )
        if change == "zip"
        else None
    )

    def changed(owner: RetainedValidationReuse) -> RetainedValidationVerdict:
        return _validate(
            path,
            registry,
            owner,
            mode=ValidationMode.ADVISORY if change == "mode" else ValidationMode.STRICT,
            source_path="other.jsonl" if change == "source" else "records.jsonl",
            coordinate=coordinate,
            resolution=resolution,
            explicit=change != "explicitness",
            jsonl=change != "format",
        )

    second = changed(reuse)
    assert reuse.outcome == "recipe_changed"
    assert len(validation_bodies) == 2
    _assert_complete_verdict_equal(second, changed(RetainedValidationReuse()))
    assert len(validation_bodies) == 3


@pytest.mark.parametrize("change", ["alias", "original", "backing", "cancel"])
def test_alias_certification_settles_readers_and_refuses_changed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validation_bodies: list[int], change: str
) -> None:
    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.schemas import validator

    _publish(tmp_path / "schemas", "integer")
    original = tmp_path / "original.jsonl"
    alias = tmp_path / "alias.jsonl"
    payload = (
        json.dumps(
            {
                "type": "message",
                "kind": "invalid",
                **{f"neutral_unused_field_{ordinal}": True for ordinal in range(400)},
            }
        )
        + "\n"
    ).encode()
    original.write_bytes(payload)
    alias.write_bytes(payload)
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    reuse = RetainedValidationReuse()
    first = _validate(original, registry, reuse)
    assert first.drift_observation is not None
    signature = first.drift_observation.unseen_key_signature._path
    assert signature is not None
    digest_original = validator._validation_input_digest
    open_original = Path.open
    readers: list[IO[bytes]] = []

    def open_input(owner: Path, mode: str = "r", buffering: int = -1) -> IO[bytes]:
        assert mode == "rb"
        reader = open_original(owner, "rb", buffering=buffering)
        if owner == alias:
            readers.append(reader)
        return reader

    def certify(source: BinaryIO, byte_length: int) -> str:
        result = digest_original(source, byte_length)
        if change == "cancel":
            raise DaemonOperationCancelled("neutral alias cancellation")
        if change == "backing":
            signature.unlink()
        else:
            with open_original(alias if change == "alias" else original, "wb") as writer:
                writer.write(payload + b" ")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_input)
        patch.setattr(validator, "_validation_input_digest", certify)
        expected = DaemonOperationCancelled if change == "cancel" else observation_spill.AcceptedPrefixReadError
        with pytest.raises(expected):
            _validate(alias, registry, reuse, raw_id="other")
    assert readers and all(reader.closed for reader in readers)
    assert reuse.outcome != "hit"
    assert len(validation_bodies) == 1


def test_work_observation_counts_leaf_bodies_even_after_a_reuse_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validation_bodies: list[int]
) -> None:
    from polylogue import logging as plog
    from polylogue.core import work_observation

    monkeypatch.setattr(work_observation, "ENABLED", True)
    _publish(tmp_path / "schemas", "integer")
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"type":"message","kind":1}\n')
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    reuse = RetainedValidationReuse()
    previous = plog.set_level("info")
    try:
        with plog.capture() as records:
            first = _validate(path, registry, reuse)
            second = _validate(path, registry, reuse, raw_id="second")
            # The real leaf executes again through a fresh owner. A prior
            # wrapper hit must not hide this duplicated complete body.
            duplicate = _validate(path, registry, RetainedValidationReuse())
        _assert_complete_verdict_equal(second, _validate(path, registry, reuse, raw_id="second"))
        assert first.sample_count == second.sample_count == duplicate.sample_count == 1
        bodies = [r for r in records if r["event"] == "work.body.ok"]
        consumed = [r for r in records if r["event"] == "work.artifact.consume"]
        assert len(validation_bodies) == len(bodies) == 2
        assert len({r["work_key"] for r in bodies}) == 1
        assert bodies[0]["work_key"] is not None
        assert bodies[0]["artifact_ref"] != bodies[1]["artifact_ref"]
        assert len(consumed) == 3
        assert [r["effect"] for r in consumed] == ["consumed", "reused", "consumed"]
        assert consumed[0]["artifact_ref"] == consumed[1]["artifact_ref"] == bodies[0]["artifact_ref"]
        assert consumed[2]["artifact_ref"] == bodies[1]["artifact_ref"]
    finally:
        plog.set_level(previous)
        reuse.clear()


@pytest.mark.parametrize("change", ["prefix", "schema", "input"])
def test_work_observation_distinguishes_changed_owned_validation_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from polylogue import logging as plog
    from polylogue.core import work_observation

    monkeypatch.setattr(work_observation, "ENABLED", True)
    schema_path = _publish(tmp_path / "schemas", "integer")
    first_record = b'{"type":"message","kind":1}\n'
    path = tmp_path / "records.jsonl"
    path.write_bytes(first_record + b'{"type":"message","kind":2}\n')
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    reuse = RetainedValidationReuse()
    previous = plog.set_level("info")
    try:
        with plog.capture() as records:
            _validate(path, registry, reuse, revision_sha256=sha256(path.read_bytes()).hexdigest())
            prefix = None
            if change == "prefix":
                prefix = len(first_record)
            elif change == "schema":
                schema_path.write_bytes(gzip.compress(json.dumps({"type": "object"}).encode()))
            else:
                path.write_bytes(first_record + b'{"type":"message","kind":3}\n')
            _validate(path, registry, reuse, prefix=prefix, revision_sha256=sha256(path.read_bytes()).hexdigest())
        bodies = [r for r in records if r["event"] == "work.body.ok"]
        assert len(bodies) == 2
        assert all(r["work_key"] is not None for r in bodies)
        assert bodies[0]["work_key"] != bodies[1]["work_key"]
    finally:
        plog.set_level(previous)
        reuse.clear()


def test_direct_validation_leaf_has_explicit_unknown_key_and_off_skips_key_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue import logging as plog
    from polylogue.core import work_observation
    from polylogue.schemas import validator
    from polylogue.schemas.retained_validation import validate_retained_document as leaf

    _publish(tmp_path / "schemas", "integer")
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"type":"message","kind":1}\n')
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    monkeypatch.setattr(work_observation, "ENABLED", True)
    previous = plog.set_level("info")
    try:
        with plog.capture() as records:
            result = leaf(
                Provider.CLAUDE_CODE,
                path,
                mode=ValidationMode.ADVISORY,
                raw_id="direct",
                revision_sha256="a" * 64,
                evidence_id="direct",
                source_path="records.jsonl",
                jsonl=True,
                registry=registry,
                signature_directory=tmp_path,
            )
        assert result.sample_count == 1
        terminal = next(r for r in records if r["event"] == "work.body.ok")
        assert terminal["work_key"] is None
        assert terminal["work_key_reason"] == "input_identity_unavailable"
        assert terminal["artifact_ref"] is not None
        monkeypatch.setattr(work_observation, "ENABLED", False)

        def forbidden(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("disabled diagnostics computed a validation key")

        monkeypatch.setattr(validator, "_validation_observation_key", forbidden)
        with plog.capture() as disabled:
            assert _validate(path, registry, RetainedValidationReuse()).sample_count == 1
        assert not any(str(r["event"]).startswith("work.") for r in disabled)
    finally:
        plog.set_level(previous)


@pytest.mark.parametrize("fixed", [False, True])
def test_retained_validation_reuses_only_the_exact_fixed_snapshot_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed: bool
) -> None:
    """Real validation shares fixed caches; dynamic declarations remain fresh."""
    root = tmp_path / "schemas"
    schema_path = _publish(root, "string")
    registry = SchemaRegistry(storage_root=root)
    with registry.current_provider_snapshot(Provider.CLAUDE_CODE) as snapshot:
        fixed_reader = snapshot.reader()
    active = fixed_reader if fixed else registry
    readers: list[SchemaRegistry] = []
    loads: list[SchemaRegistry] = []
    original_reader = ProviderSchemaSnapshot.reader
    original_load = SchemaRegistry._load_element_schema

    def reader(owner: ProviderSchemaSnapshot, existing: SchemaRegistry | None = None) -> SchemaRegistry:
        result = original_reader(owner, existing)
        readers.append(result)
        return result

    def load(
        owner: SchemaRegistry, provider_token: str, *, version: str, element_kind: str | None
    ) -> PublicSchemaDocument | None:
        loads.append(owner)
        return original_load(owner, provider_token, version=version, element_kind=element_kind)

    monkeypatch.setattr(ProviderSchemaSnapshot, "reader", reader)
    monkeypatch.setattr(SchemaRegistry, "_load_element_schema", load)
    path = tmp_path / "records.jsonl"
    path.write_text('{"type":"message","kind":"synthetic"}\n')
    first = _validate(path, active, RetainedValidationReuse())
    second = _validate(path, active, RetainedValidationReuse())
    _assert_complete_verdict_equal(second, first)
    assert first.sample_count == second.sample_count == 1
    assert first.invalid_count == 0
    assert len(readers) == 2
    assert len(loads) == (1 if fixed else 2)
    if fixed:
        assert readers[0] is readers[1] is fixed_reader
    else:
        assert readers[0] is not readers[1]
        assert all(item is not registry for item in readers)
    # Replacing the live declaration cannot alter a fixed captured reader, but
    # the dynamic public route must observe it on its next validation body.
    schema_path.write_bytes(
        gzip.compress(
            json.dumps({"type": "object", "properties": {"kind": {"type": "integer"}}, "required": ["kind"]}).encode()
        )
    )
    third = _validate(path, active, RetainedValidationReuse())
    assert third.sample_count == 1
    assert third.invalid_count == (0 if fixed else 1)
