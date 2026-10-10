"""Retained validation uses complete replayable records and compact results."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import sqlite3
import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

import pytest

from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate
from polylogue.schemas import observation_spill, retained_validation
from polylogue.schemas.drift_sentinel import DriftSignature
from polylogue.schemas.packages import SchemaResolution, SchemaResolutionReason
from polylogue.schemas.retained_validation import PrefixValidationState, _bounded_validator, _normalized
from polylogue.schemas.runtime_registry import SCHEMA_DIR, SchemaRegistry
from polylogue.schemas.validator import (
    RetainedValidationVerdict,
    SchemaValidator,
    _normalize_empty_arrays,
    validate_retained_document,
)
from polylogue.storage.sqlite.connection_profile import scratch_connection_context


def _signature_text(signature: DriftSignature) -> str:
    return b"".join(signature.iter_utf8_chunks()).decode("utf-8", "surrogatepass")


def _schema(kind: object) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"kind": kind, "type": {"type": "string"}},
        "required": ["kind"],
        "additionalProperties": True,
    }


def _resolution(version: str, *, explicit_reason: SchemaResolutionReason = "package_default") -> SchemaResolution:
    return SchemaResolution(
        provider="claude-code",
        package_version=version,
        element_kind="session_record_stream",
        exact_structure_id=None,
        bundle_scope=None,
        reason=explicit_reason,
    )


def _registry(tmp_path: Path, current: object, historical: object | None = None) -> SchemaRegistry:
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    if historical is not None:
        registry.write_schema_version("claude-code", "v1", _schema(historical), element_kind="session_record_stream")
    registry.write_schema_version("claude-code", "v2", _schema(current), element_kind="session_record_stream")
    return registry


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize("count", [3, 27])
def test_retained_validation_initializes_reducer_tables_once_per_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    registry = _registry(tmp_path, {"type": "integer"})
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [{"type": "message", "kind": number} for number in range(count)])
    declarations: list[str] = []
    original_enter = observation_spill.StreamedJSONDocument.__enter__

    def traced_enter(owner: observation_spill.StreamedJSONDocument) -> JSONValue:
        result = original_enter(owner)
        owner.connection.set_trace_callback(
            lambda statement: (
                declarations.append(statement) if statement.startswith("CREATE TABLE IF NOT EXISTS retained_") else None
            )
        )
        return result

    monkeypatch.setattr(observation_spill.StreamedJSONDocument, "__enter__", traced_enter)
    for revision in range(2):
        declarations.clear()
        verdict = validate_retained_document(
            Provider.CLAUDE_CODE,
            path,
            mode=ValidationMode.STRICT,
            raw_id=f"raw-{revision}",
            revision_sha256="a" * 64,
            evidence_id=f"raw-{revision}",
            source_path="records.jsonl",
            jsonl=True,
            schema_resolution=_resolution("v2"),
            schema_resolution_is_explicit=False,
            registry=registry,
            signature_directory=tmp_path,
        )
        assert verdict.sample_count == count
        assert verdict.status is ValidationStatus.PASSED
        assert verdict.invalid_count == verdict.error_count == verdict.drift_count == 0
        # Exact drift paths now own a separate chunk table. Each table must
        # still be initialized once for the whole document, never per sample.
        names = [statement.split()[5].split("(", 1)[0] for statement in declarations]
        assert sorted(names) == [
            "retained_drift",
            "retained_drift_chunks",
            "retained_eval_items",
            "retained_eval_props",
            "retained_scope",
            "retained_unique",
            "retained_unread",
        ]
        assert retained_validation._ACTIVE_VALIDATION_SCRATCH.get() is None


def test_normalized_collection_iteration_uses_original_spill_cursors(tmp_path: Path) -> None:
    path = tmp_path / "document.json"
    path.write_text(json.dumps({"a": 1, "b": [], "rows": [1, 2, 3]}), encoding="utf-8")
    schema = {
        "type": "object",
        "properties": {"b": {"type": "null"}, "rows": {"type": "array", "items": {"type": "integer"}}},
    }
    owner = observation_spill.StreamedJSONDocument(path)
    with owner as payload:
        normalized = _normalized(payload, schema, schema, owner.connection)
        assert isinstance(normalized, dict)
        statements: list[str] = []
        owner.connection.set_trace_callback(statements.append)
        pairs = list(normalized.items())
        assert [(key, list(value) if key == "rows" else value) for key, value in pairs] == [
            ("a", 1),
            ("b", None),
            ("rows", [1, 2, 3]),
        ]
        assert not any("AND key_bytes =" in statement for statement in statements)
        assert not any("AND ordinal =" in statement for statement in statements)
        statements.clear()
        assert list(normalized.values()) == [1, None, [1, 2, 3]]
        assert not any("AND key_bytes =" in statement for statement in statements)


def test_normalized_iteration_checks_cancellation_before_loading_next_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "array.json"
    path.write_text("[1,2,3]", encoding="utf-8")
    owner = observation_spill.StreamedJSONDocument(path)
    with owner as payload:
        normalized = _normalized(payload, {"type": "array"}, {}, owner.connection)
        assert isinstance(normalized, list)
        statements: list[str] = []
        owner.connection.set_trace_callback(statements.append)

        def cancelled() -> None:
            raise RuntimeError("cancelled")

        monkeypatch.setattr(retained_validation, "check_compute_cancelled", cancelled)
        with pytest.raises(RuntimeError, match="cancelled"):
            next(iter(normalized))
        assert statements == []


def _schema_shaped_witness(schema: object, root: object | None = None, depth: int = 0) -> object:
    """Build a small synthetic candidate; the independent oracle decides validity."""
    from jsonschema import Draft202012Validator

    root = schema if root is None else root
    if depth > 32 or schema is False:
        return None
    if schema is True or not isinstance(schema, dict):
        return {}
    validator = Draft202012Validator(root)
    if "const" in schema:
        return schema["const"]
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return schema["enum"][0]
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            base = {key: value for key, value in schema.items() if key not in {"anyOf", "oneOf"}}
            for branch in branches:
                candidate = _schema_shaped_witness(branch, root, depth + 1)
                if isinstance(candidate, dict):
                    base_candidate = _schema_shaped_witness(base, root, depth + 1)
                    if isinstance(base_candidate, dict):
                        candidate = {**base_candidate, **candidate}
                if validator.is_valid(candidate):
                    return candidate
    if "allOf" in schema:
        base = {key: value for key, value in schema.items() if key != "allOf"}
        candidate = _schema_shaped_witness(base, root, depth + 1)
        if not isinstance(candidate, dict):
            candidate = {}
        for branch in schema["allOf"]:
            value = _schema_shaped_witness(branch, root, depth + 1)
            if isinstance(value, dict):
                candidate.update(value)
        if validator.is_valid(candidate):
            return candidate
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if "object" in kinds or "properties" in schema or "required" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            return {}
        return {key: _schema_shaped_witness(properties.get(key, {}), root, depth + 1) for key in required}
    if "array" in kinds:
        items = schema.get("items", {})
        minimum = int(schema.get("minItems", 0))
        return [_schema_shaped_witness(items, root, depth + 1) for _ in range(minimum)]
    if "string" in kinds:
        return ""
    if "integer" in kinds or "number" in kinds:
        return 0
    if "boolean" in kinds:
        return True
    if "null" in kinds:
        return None
    return {}


def _late_invalid_variant(schema: object, witness: object) -> object | None:
    """Try a few schema-guided type changes and let Draft 2020-12 arbitrate."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema)
    if not isinstance(schema, dict) or not isinstance(witness, dict):
        return None
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for key, child_schema in properties.items():
            if key not in witness or not isinstance(child_schema, dict):
                continue
            kind = child_schema.get("type")
            kind_key = kind if isinstance(kind, str) else None
            alternatives: dict[str, tuple[object, ...]] = {
                "string": (0, None, []),
                "integer": ("invalid", None, []),
                "number": ("invalid", None, []),
                "boolean": (0, "invalid", None),
                "object": ("invalid", None, []),
                "array": ("invalid", None, {}),
            }
            replacements = (
                alternatives.get(kind_key, (None, "invalid", [])) if kind_key is not None else (None, "invalid", [])
            )
            for replacement in replacements:
                candidate = copy.deepcopy(witness)
                candidate[key] = replacement
                if not validator.is_valid(candidate):
                    return candidate
    root_candidate: object
    for root_candidate in ([], None, "invalid", 0, False):
        if not validator.is_valid(root_candidate):
            return root_candidate
    return None


class _RetainedValidationArguments(TypedDict):
    signature_directory: Path
    provider: str
    path: Path
    raw_id: str
    revision_sha256: str
    evidence_id: str
    jsonl: bool
    schema_resolution: SchemaResolution
    schema_resolution_is_explicit: bool
    registry: SchemaRegistry


def test_retained_strict_counts_late_failure_and_advisory_accepts(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    _write_jsonl(path, [{"type": "record", "kind": "first"}, {"type": "record", "kind": 17}])
    registry = _registry(tmp_path, {"type": "string"})
    args: _RetainedValidationArguments = {
        "provider": "claude-code",
        "path": path,
        "signature_directory": tmp_path,
        "raw_id": "raw-a",
        "revision_sha256": "a" * 64,
        "evidence_id": "raw-a",
        "jsonl": True,
        "schema_resolution": _resolution("v2"),
        "schema_resolution_is_explicit": True,
        "registry": registry,
    }

    strict = validate_retained_document(mode=ValidationMode.STRICT, **args)
    advisory = validate_retained_document(mode=ValidationMode.ADVISORY, **args)
    skipped = validate_retained_document(mode=ValidationMode.OFF, **args)

    assert (strict.status, strict.strict_refusal) == (ValidationStatus.FAILED, True)
    assert (strict.sample_count, strict.invalid_count, strict.error_count) == (2, 1, 1)
    assert strict.first_diagnostic is not None and "kind" in strict.first_diagnostic
    assert (advisory.status, advisory.strict_refusal) == (ValidationStatus.PASSED, False)
    assert (advisory.sample_count, advisory.invalid_count, advisory.error_count) == (2, 1, 1)
    assert skipped.status is ValidationStatus.SKIPPED
    assert (strict.raw_id, strict.revision_sha256, strict.evidence_id) == (
        "raw-a",
        "a" * 64,
        "raw-a",
    )


def test_retained_validation_reports_real_nested_schema_traversal_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.core import work_progress

    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(work_progress, "PROGRESS_INTERVAL_S", 0)
    monkeypatch.setattr(work_progress, "emit", lambda event, **fields: events.append((event, fields)))

    path = tmp_path / "raw.jsonl"
    record = {
        "type": "record",
        "kind": "session",
        "empty-containers": [{}, {}, {}, {}],
        **{f"field-{i}": "payload" * 12 for i in range(40)},
    }
    _write_jsonl(path, [record])
    schema = _schema({"type": "string"})
    schema["additionalProperties"] = {"type": "string"}
    schema["properties"] = {"empty-containers": {"type": "array", "items": {"type": "object"}}}
    registry = _registry(tmp_path, schema)

    verdict = validate_retained_document(
        "claude-code",
        path,
        mode=ValidationMode.ADVISORY,
        raw_id="raw-progress",
        revision_sha256="b" * 64,
        evidence_id="raw-progress",
        jsonl=True,
        schema_resolution=_resolution("v2"),
        schema_resolution_is_explicit=True,
        registry=registry,
        signature_directory=(path).parent,
    )

    assert verdict.sample_count == 1
    progress = [fields for event, fields in events if event == "daemon.work.progress"]
    assert len(progress) > 3
    assert len({fields["unit_id"] for fields in progress}) == 1
    assert len({fields["productive_id"] for fields in progress}) == 1

    def bytes_count(fields: dict[str, object]) -> int:
        value = fields["bytes"]
        assert isinstance(value, int)
        return value

    progress_bytes = [bytes_count(fields) for fields in progress]
    assert progress_bytes == sorted(progress_bytes)
    assert progress_bytes[-1] > 40 * len("payload" * 12)
    assert progress_bytes[-1] >= 2 * 4


def test_retained_validation_productive_identity_uses_source_recipe_not_attempt_path(tmp_path: Path) -> None:
    from polylogue.core.work_progress import stable_productive_identity
    from polylogue.schemas.retained_validation import _retained_validation_productive_identity

    def identity(*, path: Path, revision_sha256: str = "a" * 64, mode: ValidationMode = ValidationMode.ADVISORY) -> str:
        return _retained_validation_productive_identity(
            "claude-code",
            path,
            mode=mode,
            raw_id="raw-a",
            revision_sha256=revision_sha256,
            evidence_id="raw-a",
            source_path="/configured/source.jsonl",
            jsonl=True,
            schema_resolution=_resolution("v2"),
            schema_resolution_is_explicit=True,
            signature_directory=path.parent,
        )

    original = identity(path=tmp_path / "attempt-a.jsonl")
    assert identity(path=tmp_path / "attempt-b.jsonl") == original
    assert identity(path=tmp_path / "attempt-a.jsonl", revision_sha256="b" * 64) != original
    assert identity(path=tmp_path / "attempt-a.jsonl", mode=ValidationMode.STRICT) != original
    assert stable_productive_identity(("source-path", "/tmp/source-\udcff.jsonl"))


def test_zip_occurrence_validation_progress_reuses_only_the_same_captured_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import warnings
    import zipfile

    from devtools.fresh_build_bench.run import WorkProgressTail
    from polylogue.core import work_progress
    from polylogue.core.raw_coordinates import MemberAddressingMode
    from polylogue.sources.source_acquisition_components import (
        captured_zip_member_coordinate,
        zip_acquisition_fingerprint,
    )
    from polylogue.sources.source_staging import bind_source_input

    payload = json.dumps(
        {"id": "same-session", "mapping": {"root": {"message": {"author": {"role": "user"}}}}},
        separators=(",", ":"),
    ).encode()
    first_container = tmp_path / "first.zip"
    second_container = tmp_path / "second.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(first_container, "w") as archive:
            archive.writestr("same.json", payload)
            archive.writestr("same.json", payload)
    with zipfile.ZipFile(second_container, "w") as archive:
        archive.writestr("same.json", payload)

    decoder_fingerprint = zip_acquisition_fingerprint(Provider.CHATGPT)

    def captured_coordinate(container: Path, ordinal: int) -> CapturedZipMemberCoordinate:
        with zipfile.ZipFile(container) as archive:
            entry = archive.infolist()[ordinal]
            assert entry.filename == "same.json"
            assert archive.read(entry) == payload
        with bind_source_input(container) as binding:
            coordinate = captured_zip_member_coordinate(
                binding.captured_identity,
                entry_name="same.json",
                entry_ordinal=ordinal,
                split_index=0,
                addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
                container_blob_hash=hashlib.sha256(container.read_bytes()).hexdigest(),
                decoder_fingerprint=decoder_fingerprint,
            )
        assert coordinate is not None
        return coordinate

    coordinates = (
        captured_coordinate(first_container, 0),
        captured_coordinate(first_container, 1),
        captured_coordinate(second_container, 0),
    )
    emitted: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(work_progress, "PROGRESS_INTERVAL_S", 0)
    monkeypatch.setattr(work_progress, "emit", lambda event, **fields: emitted.append((event, fields)))
    events_path = tmp_path / "events.jsonl"
    tail = WorkProgressTail(events_path, state_root=tmp_path)
    try:
        observed: list[tuple[str, int]] = []
        for attempt, coordinate in enumerate((*coordinates[:1], *coordinates[:1], *coordinates[1:])):
            path = tmp_path / f"attempt-{attempt}.json"
            path.write_bytes(payload)
            validate_retained_document(
                Provider.CHATGPT,
                path,
                mode=ValidationMode.ADVISORY,
                raw_id="same-raw-id",
                revision_sha256=hashlib.sha256(payload).hexdigest(),
                evidence_id="same-raw-id",
                source_path=coordinate.declared_member,
                captured_zip_coordinate=coordinate,
                signature_directory=(path).parent,
            )
            productive_ids = [
                str(fields["productive_id"]) for event, fields in emitted if event == "daemon.work.progress"
            ]
            assert productive_ids
            with events_path.open("a", encoding="utf-8") as handle:
                for event, fields in emitted:
                    if event == "daemon.work.progress":
                        handle.write(json.dumps({"event": event, **fields}) + "\n")
            emitted.clear()
            observed.append((productive_ids[-1], tail.poll()))

        assert observed[0][0] == observed[1][0]
        assert observed[0][1] > 0
        assert observed[1][1] == observed[0][1]
        assert len({identity for identity, _count in observed}) == 3
        assert observed[2][1] > observed[1][1]
        assert observed[3][1] > observed[2][1]
    finally:
        tail.close()


def test_spilled_object_membership_checks_only_the_key_index(tmp_path: Path) -> None:
    from polylogue.schemas.observation_spill import SpilledObject, StreamedJSONDocument

    path = tmp_path / "membership.json"
    path.write_text(json.dumps({"present": None, "surrogate\ud800": "value"}), encoding="utf-8")
    document = StreamedJSONDocument(path)
    with document as payload:
        assert isinstance(payload, SpilledObject)
        statements: list[str] = []
        document.connection.set_trace_callback(statements.append)
        before = len(statements)

        assert "present" in payload
        assert "missing" not in payload
        assert "surrogate\ud800" in payload
        assert None not in payload
        assert payload.__contains__(1) is False

        membership_sql = statements[before:]
        assert len(membership_sql) == 3
        assert all("SELECT 1," in sql and "FROM json_object_members" in sql for sql in membership_sql)
        assert all("json_nodes" not in sql for sql in membership_sql)


def test_retained_schema_validation_membership_does_not_decode_spilled_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "retained-membership.jsonl"
    _write_jsonl(
        path,
        [
            {
                "nullable": None,
                "count": 4,
                "large": "x" * 1_000_000,
                "nested": {"enabled": True},
            }
        ],
    )
    schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "x-polylogue-sample-granularity": "document",
        "type": "object",
        "properties": {
            "nullable": {"type": "null"},
            "count": {"type": "integer"},
            "large": {"type": "string"},
            "nested": {"type": "object", "properties": {"enabled": {"type": "boolean"}}},
            "absent": {"type": "string"},
        },
    }
    registry = SchemaRegistry(storage_root=tmp_path / "membership-schemas")
    registry.write_schema_version("claude-code", "v2", schema, element_kind="session_record_stream")

    original_contains = observation_spill.SpilledObject.__contains__
    original_load_node = observation_spill._load_node
    inside_membership = False
    membership_calls = 0
    membership_node_loads = 0

    def contains(self: observation_spill.SpilledObject, key: object) -> bool:
        nonlocal inside_membership, membership_calls
        previous = inside_membership
        inside_membership = True
        membership_calls += 1
        try:
            return original_contains(self, key)
        finally:
            inside_membership = previous

    def load_node(connection: sqlite3.Connection, node_id: int) -> JSONValue:
        nonlocal membership_node_loads
        if inside_membership:
            membership_node_loads += 1
        return original_load_node(connection, node_id)

    monkeypatch.setattr(observation_spill.SpilledObject, "__contains__", contains)
    monkeypatch.setattr(observation_spill, "_load_node", load_node)

    verdict = validate_retained_document(
        "claude-code",
        path,
        mode=ValidationMode.ADVISORY,
        raw_id="raw-membership",
        revision_sha256="c" * 64,
        evidence_id="raw-membership",
        source_path=str(path),
        jsonl=True,
        schema_resolution=_resolution("v2"),
        schema_resolution_is_explicit=True,
        registry=registry,
        signature_directory=(path).parent,
    )

    assert verdict.status is ValidationStatus.PASSED
    assert verdict.invalid_count == 0
    assert verdict.sample_count == 1
    assert membership_calls > 0
    assert membership_node_loads == 0


def test_retained_historical_fallback_replays_every_jsonl_record(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    _write_jsonl(path, [{"type": "record", "kind": "text"}, {"type": "record", "kind": 17}])
    registry = _registry(
        tmp_path,
        {"type": "string"},
        {"anyOf": [{"type": "string"}, {"type": "integer"}]},
    )

    verdict = validate_retained_document(
        "claude-code",
        path,
        mode=ValidationMode.STRICT,
        raw_id="raw-b",
        revision_sha256="b" * 64,
        evidence_id="raw-b",
        jsonl=True,
        schema_resolution=_resolution("v2"),
        schema_resolution_is_explicit=False,
        registry=registry,
        signature_directory=(path).parent,
    )

    assert verdict.status is ValidationStatus.PASSED
    assert verdict.invalid_count == 0
    assert verdict.sample_count == 2
    assert verdict.schema_resolution is not None
    assert verdict.schema_resolution.package_version == "v1"


def test_prefix_validation_state_matches_each_current_and_historical_resolution(tmp_path: Path) -> None:
    registry = SchemaRegistry(storage_root=tmp_path / "codex-schemas")

    def codex_schema(accepted_text: list[str]) -> dict[str, object]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["type", "payload"],
            "properties": {
                "type": {"enum": ["session_meta", "response_item"]},
                "payload": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"text": {"enum": accepted_text}},
                                "required": ["text"],
                            },
                        }
                    },
                },
            },
            "allOf": [
                {
                    "if": {"properties": {"type": {"const": "response_item"}}},
                    "then": {"properties": {"payload": {"required": ["content"]}}},
                }
            ],
        }

    registry.write_schema_version("codex", "v1", codex_schema(["old", "new"]), element_kind="session_record_stream")
    registry.write_schema_version("codex", "v2", codex_schema(["old"]), element_kind="session_record_stream")
    registry.write_schema_version("codex", "v3", codex_schema(["current"]), element_kind="session_record_stream")
    records: list[JSONDocument] = [
        {"type": "session_meta", "payload": {"id": "prefix-state", "timestamp": "2026-01-01T00:00:00Z"}}
    ]
    for index, text in enumerate(("old", "old", "new", "bad")):
        records.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"m{index}",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    path = tmp_path / "prefix-state.jsonl"
    with PrefixValidationState(
        provider=Provider.CODEX,
        source_path=str(path),
        mode=ValidationMode.STRICT,
        registry=registry,
        scratch_directory=tmp_path,
        signature_directory=tmp_path,
    ) as state:
        for record_index, record in enumerate(records, start=1):
            state.observe(record)
            if record_index < 2:
                continue
            prefix_bytes = "".join(json.dumps(item) + "\n" for item in records[:record_index]).encode()
            path.write_bytes(prefix_bytes)
            raw_id = f"raw-prefix-{record_index}"
            digest = hashlib.sha256(prefix_bytes).hexdigest()
            streamed = state.verdict(raw_id=raw_id, revision_sha256=digest, evidence_id=raw_id)
            ordinary = validate_retained_document(
                Provider.CODEX,
                path,
                mode=ValidationMode.STRICT,
                raw_id=raw_id,
                revision_sha256=digest,
                evidence_id=raw_id,
                source_path=str(path),
                jsonl=True,
                registry=registry,
                signature_directory=(path).parent,
            )
            assert streamed == ordinary
            if record_index == 2:
                assert streamed.schema_resolution is not None
                assert streamed.schema_resolution.package_version == "v2"
            if record_index == 4:
                assert streamed.schema_resolution is not None
                assert streamed.schema_resolution.package_version == "v1"
            if record_index == 5:
                assert streamed.schema_resolution is not None
                assert (streamed.schema_resolution.package_version, streamed.invalid_count) == ("v3", 4)
                assert streamed.status is ValidationStatus.FAILED
        counts = {
            candidate.version: candidate.reducer.sample_count for candidate in state._reducers if candidate.reducer
        }
        assert counts == {"v3": 5, "v2": 4, "v1": 5}


def test_prefix_validation_state_preserves_sampler_witness_order_at_64_records(tmp_path: Path) -> None:
    from polylogue.schemas.generation.dynamic_keys import (
        legacy_structure_schema_digest,
        observed_structure_schema,
        structure_schema_digest,
    )

    registry = SchemaRegistry(storage_root=tmp_path / "witness-schemas")
    schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["type", "payload"],
        "properties": {"type": {"type": "string"}, "payload": {"type": "object"}},
    }
    registry.write_schema_version("codex", "v1", schema, element_kind="session_record_stream")

    def row_witnesses(row: object) -> set[str]:
        observed = observed_structure_schema(row)
        canonical = structure_schema_digest(observed)
        legacy = legacy_structure_schema_digest(observed)
        return {canonical, legacy}

    header: JSONDocument = {"type": "session_meta", "payload": {"id": "witness-session"}}
    first_message: JSONDocument = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": "m0",
            "role": "user",
            "content": [{"type": "input_text", "text": "same"}],
        },
    }
    catalog = registry.load_package_catalog("codex")
    assert catalog is not None
    catalog.packages[0].elements[0].exact_structure_ids = sorted(row_witnesses(header) | row_witnesses(first_message))
    registry.save_package_catalog(catalog)

    records: list[JSONDocument] = [header]
    for index in range(64):
        row = dict(first_message)
        message_payload = first_message["payload"]
        assert isinstance(message_payload, dict)
        row["payload"] = {**message_payload, "id": f"m{index}"}
        records.append(row)

    path = tmp_path / "witness-prefix.jsonl"
    with PrefixValidationState(
        provider=Provider.CODEX,
        source_path=str(path),
        mode=ValidationMode.ADVISORY,
        registry=registry,
        scratch_directory=tmp_path,
        signature_directory=tmp_path,
    ) as state:
        for record_index, record in enumerate(records, start=1):
            state.observe(record)
            if record_index not in {64, 65}:
                continue
            prefix_bytes = "".join(json.dumps(item) + "\n" for item in records[:record_index]).encode()
            path.write_bytes(prefix_bytes)
            raw_id = f"witness-raw-{record_index}"
            digest = hashlib.sha256(prefix_bytes).hexdigest()
            streamed = state.verdict(raw_id=raw_id, revision_sha256=digest, evidence_id=raw_id)
            ordinary = validate_retained_document(
                Provider.CODEX,
                path,
                mode=ValidationMode.ADVISORY,
                raw_id=raw_id,
                revision_sha256=digest,
                evidence_id=raw_id,
                source_path=str(path),
                jsonl=True,
                registry=registry,
                signature_directory=(path).parent,
            )
            assert streamed == ordinary
            assert streamed.schema_resolution is not None
            expected_first = header if record_index == 64 else first_message
            assert streamed.schema_resolution.exact_structure_id in row_witnesses(expected_first)


def test_retained_drift_reduction_is_order_independent(tmp_path: Path) -> None:
    registry = _registry(tmp_path, {"type": "string"})
    resolution = _resolution("v2", explicit_reason="exact_structure")

    def run(name: str, rows: list[dict[str, object]]) -> RetainedValidationVerdict:
        path = tmp_path / name
        _write_jsonl(path, rows)
        return validate_retained_document(
            "claude-code",
            path,
            mode=ValidationMode.ADVISORY,
            raw_id="raw-c",
            revision_sha256="c" * 64,
            evidence_id="raw-c",
            jsonl=True,
            schema_resolution=resolution,
            schema_resolution_is_explicit=True,
            registry=registry,
            signature_directory=(path).parent,
        )

    rows = [{"type": "record", "kind": 1, "alpha": 1}, {"type": "record", "kind": 2, "beta": 1}]
    forward = run("forward.jsonl", rows)
    reverse = run("reverse.jsonl", list(reversed(rows)))

    assert forward.drift_observation is not None
    assert reverse.drift_observation is not None
    assert forward.drift_observation.classification == "field_changed"
    assert reverse.drift_observation.classification == "field_changed"
    assert _signature_text(forward.drift_observation.unseen_key_signature) == "alpha"
    assert _signature_text(reverse.drift_observation.unseen_key_signature) == "alpha"


def test_retained_drift_classifies_default_and_known_unread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(tmp_path, {"type": "string"})
    path = tmp_path / "record.json"
    path.write_text(json.dumps({"type": "record", "kind": "value"}), encoding="utf-8")
    monkeypatch.setattr("polylogue.schemas.retained_validation.unread_field_names", lambda _provider: {"kind"})

    unseen = validate_retained_document(
        "claude-code",
        path,
        mode=ValidationMode.ADVISORY,
        raw_id="raw-d",
        revision_sha256="d" * 64,
        evidence_id="raw-d",
        schema_resolution=_resolution("v2"),
        schema_resolution_is_explicit=True,
        registry=registry,
        signature_directory=(path).parent,
    )
    unread = validate_retained_document(
        "claude-code",
        path,
        mode=ValidationMode.ADVISORY,
        raw_id="raw-e",
        revision_sha256="e" * 64,
        evidence_id="raw-e",
        schema_resolution=_resolution("v2", explicit_reason="exact_structure"),
        schema_resolution_is_explicit=True,
        registry=registry,
        signature_directory=(path).parent,
    )

    assert unseen.drift_observation is not None
    assert unseen.drift_observation.classification == "unseen_shape"
    assert unread.drift_observation is not None
    assert unread.drift_observation.classification == "known_field_unread"
    assert _signature_text(unread.drift_observation.unseen_key_signature) == "kind"


def test_retained_unread_detection_does_not_probe_every_declared_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path, {"type": "string"})
    path = tmp_path / "record.json"
    path.write_text(json.dumps({"type": "record", "kind": "value"}), encoding="utf-8")
    names = {f"absent_{number}" for number in range(1000)} | {"kind"}
    monkeypatch.setattr(retained_validation, "unread_field_names", lambda _provider: names)
    probes = 0
    original = observation_spill.SpilledObject._member

    def member(self: observation_spill.SpilledObject, key: str) -> int | None:
        nonlocal probes
        if key in names:
            probes += 1
        return original(self, key)

    monkeypatch.setattr(observation_spill.SpilledObject, "_member", member)
    verdict = validate_retained_document(
        Provider.CLAUDE_CODE,
        path,
        mode=ValidationMode.ADVISORY,
        raw_id="raw-unread",
        revision_sha256="a" * 64,
        evidence_id="raw-unread",
        schema_resolution=_resolution("v2", explicit_reason="exact_structure"),
        schema_resolution_is_explicit=True,
        registry=registry,
        signature_directory=tmp_path,
    )
    assert verdict.drift_observation is not None
    assert _signature_text(verdict.drift_observation.unseen_key_signature) == "kind"
    assert probes < 20


def test_spilled_unread_intersection_streams_long_names_and_checks_digest_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    present = "present_" + "x" * 160
    absent = "absent_" + "y" * 160
    oversized = "z" * 131072
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"kind": 1, present: 2, oversized: 3}), encoding="utf-8")
    owner = observation_spill.StreamedJSONDocument(path)
    with owner as sample:
        assert isinstance(sample, observation_spill.SpilledObject)
        # A digest match remains only an accelerator, never identity evidence.
        owner.connection.execute(
            "UPDATE json_key_meta SET digest=? WHERE short_chars=?",
            (hashlib.sha256(absent.encode()).digest(), len(oversized)),
        )

        def forbid_read(_self: observation_spill.SpilledKey) -> str:
            raise AssertionError("field intersection must not materialize an oversized key")

        monkeypatch.setattr(observation_spill.SpilledKey, "read", forbid_read)
        assert set(sample.matching_field_names({"kind", present, absent})) == {"kind", present}


def test_public_validator_shares_spill_safe_extended_keywords() -> None:
    schema = {
        "type": "object",
        "allOf": [{"properties": {"chosen": {"type": "integer"}}}],
        "properties": {
            "unique": {"type": "array", "uniqueItems": True},
            "items": {
                "type": "array",
                "prefixItems": [{"type": "integer"}],
                "unevaluatedItems": False,
            },
            "branch": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        },
        "unevaluatedProperties": False,
    }
    validator = SchemaValidator(schema, strict=False)

    assert validator.validate(
        {"chosen": 1, "unique": [True, 1], "items": [2], "branch": 3}, include_drift=False
    ).is_valid
    assert not validator.validate(
        {"chosen": 1, "unique": [1, 1.0], "items": [2, 3], "branch": 3}, include_drift=False
    ).is_valid


def test_public_validator_preserves_local_refs_anyof_and_oneof() -> None:
    validator = SchemaValidator(
        {
            "$defs": {"choice": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
            "type": "object",
            "properties": {
                "referenced": {"$ref": "#/$defs/choice"},
                "exclusive": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            },
            "required": ["referenced", "exclusive"],
        },
        strict=False,
    )

    assert validator.validate({"referenced": 2, "exclusive": "text"}, include_drift=False).is_valid
    assert not validator.validate({"referenced": True, "exclusive": "text"}, include_drift=False).is_valid


def test_retained_reduces_many_invalid_records_and_closes_spill_on_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path, {"type": "string"})
    path = tmp_path / "invalid.jsonl"
    _write_jsonl(path, [{"type": "record", "kind": index} for index in range(512)])
    resolution = _resolution("v2", explicit_reason="exact_structure")
    verdict = validate_retained_document(
        "claude_code",
        path,
        mode=ValidationMode.ADVISORY,
        raw_id="raw-many",
        revision_sha256="f" * 64,
        evidence_id="raw-many",
        jsonl=True,
        schema_resolution=resolution,
        schema_resolution_is_explicit=True,
        registry=registry,
        signature_directory=(path).parent,
    )
    assert verdict.sample_count == 512
    assert verdict.invalid_count == 512
    assert verdict.error_count == 512

    captured: dict[str, str] = {}
    original_enter = observation_spill.StreamedJSONDocument.__enter__

    def enter(document: Any) -> Any:
        payload = original_enter(document)
        captured["database"] = str(document.connection.execute("PRAGMA database_list").fetchone()[2])
        return payload

    monkeypatch.setattr(observation_spill.StreamedJSONDocument, "__enter__", enter)
    monkeypatch.setattr(
        "polylogue.schemas.retained_validation.check_compute_cancelled",
        lambda: (_ for _ in ()).throw(RuntimeError("cancelled")),
    )
    with pytest.raises(RuntimeError, match="cancelled"):
        validate_retained_document(
            "claude-code",
            path,
            mode=ValidationMode.ADVISORY,
            raw_id="raw-cancel",
            revision_sha256="e" * 64,
            evidence_id="raw-cancel",
            jsonl=True,
            schema_resolution=resolution,
            schema_resolution_is_explicit=True,
            registry=registry,
            signature_directory=(path).parent,
        )
    assert captured["database"]
    assert not Path(captured["database"]).exists()


def test_retained_invalid_record_peak_memory_does_not_track_error_count(tmp_path: Path) -> None:
    registry = _registry(tmp_path, {"type": "string"})
    resolution = _resolution("v2", explicit_reason="exact_structure")
    counts = (256, 2048)
    paths = {count: tmp_path / f"invalid-{count}.jsonl" for count in counts}
    for count, path in paths.items():
        with path.open("w", encoding="utf-8") as stream:
            for index in range(count):
                stream.write(json.dumps({"type": "record", "kind": index, f"added_{index}": True}) + "\n")

    def validate(path: Path, count: int) -> int:
        verdict = validate_retained_document(
            "claude-code",
            path,
            mode=ValidationMode.ADVISORY,
            raw_id=f"raw-{count}",
            revision_sha256=f"{count:064x}",
            evidence_id=f"raw-{count}",
            jsonl=True,
            schema_resolution=resolution,
            schema_resolution_is_explicit=True,
            registry=registry,
            signature_directory=(path).parent,
        )
        assert (verdict.sample_count, verdict.invalid_count, verdict.error_count, verdict.drift_count) == (
            count,
            count,
            count,
            count,
        )
        assert verdict.first_diagnostic is not None and "kind" in verdict.first_diagnostic
        assert verdict.drift_observation is not None
        assert verdict.drift_observation.classification == "field_changed"
        assert _signature_text(verdict.drift_observation.unseen_key_signature) == "added_0"
        return verdict.invalid_count

    validate(paths[256], 256)  # Warm package/schema caches before tracing the comparative runs.
    peaks: list[int] = []
    tracemalloc.start()
    try:
        for count in counts:
            gc.collect()
            tracemalloc.reset_peak()
            baseline, _ = tracemalloc.get_traced_memory()
            errors = validate(paths[count], count)
            _, peak = tracemalloc.get_traced_memory()
            peaks.append(peak - baseline)
            assert errors == count
        assert counts[1] == 8 * counts[0]
        assert peaks[1] < peaks[0] * 2, (
            f"peak traced bytes for {counts}: {peaks}; top allocations: "
            f"{tracemalloc.take_snapshot().statistics('lineno')[:12]}"
        )
    finally:
        tracemalloc.stop()


def test_committed_schema_files_match_draft202012_validity() -> None:
    """The streaming extensions preserve baseline validity for every committed package schema."""
    from jsonschema import Draft202012Validator

    registry = SchemaRegistry(storage_root=SCHEMA_DIR)
    cases: tuple[object, ...] = (
        None,
        True,
        0,
        "neutral",
        [],
        {},
        {"type": "message", "id": "neutral", "content": "text"},
        {"messages": []},
        {"type": "session", "messages": [{"role": "user", "content": "text"}]},
    )
    schema_count = 0
    schema_shaped_positive_count = 0
    late_invalid_count = 0
    generic_distribution: dict[str, list[int]] = {}
    with scratch_connection_context(
        prefix="polylogue-schema-package-parity-", filename="validation.sqlite"
    ) as connection:
        for provider in registry.list_committed_providers():
            for version in registry.list_committed_versions(provider):
                for schema_file in registry.list_committed_schema_files(provider, version):
                    schema = registry.load_committed_schema_file(provider, version, schema_file)
                    assert schema is not None, (provider, version, schema_file)
                    schema_count += 1
                    oracle = Draft202012Validator(schema)
                    generic_valid = sum(oracle.is_valid(_normalize_empty_arrays(case, schema)) for case in cases)
                    bucket = generic_distribution.setdefault(provider, [0, 0])
                    bucket[0 if generic_valid else 1] += 1
                    witness = _schema_shaped_witness(schema)
                    assert oracle.is_valid(witness), (provider, version, schema_file, "witness", witness)
                    schema_shaped_positive_count += 1
                    late_invalid = _late_invalid_variant(schema, witness)
                    if late_invalid is not None:
                        assert not oracle.is_valid(late_invalid), (
                            provider,
                            version,
                            schema_file,
                            "late invalid variant",
                            late_invalid,
                        )
                        late_invalid_count += 1
                    parity_cases = (*cases, witness, *((late_invalid,) if late_invalid is not None else ()))
                    for case in parity_cases:
                        expected = Draft202012Validator(schema).is_valid(_normalize_empty_arrays(case, schema))
                        actual = _bounded_validator(schema, connection).is_valid(
                            _normalized(case, schema, schema, connection)
                        )
                        assert actual == expected, (provider, version, schema_file, case)
    assert schema_count == 60
    assert schema_shaped_positive_count == schema_count, generic_distribution
    assert late_invalid_count == schema_count, f"only {late_invalid_count}/{schema_count} had invalid variants"


@pytest.mark.parametrize(
    "valid,reason,extra,unread,expected",
    [
        (False, "package_default", True, True, "field_changed"),
        (True, "package_default", True, True, "unseen_shape"),
        (True, "exact_structure", True, True, "new_field"),
        (True, "exact_structure", False, True, "known_field_unread"),
        (True, "exact_structure", False, False, None),
        (False, "exact_structure", False, True, "field_changed"),
    ],
)
def test_retained_drift_classification_preserves_combined_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid: bool,
    reason: SchemaResolutionReason,
    extra: bool,
    unread: bool,
    expected: str | None,
) -> None:
    """Mutation: unknown/default or unread fields outrank the stronger retained signal."""
    from polylogue.schemas.drift_sentinel import BENIGN_CLASSIFICATIONS, RISKY_CLASSIFICATIONS, is_risky

    registry = _registry(tmp_path, {"type": "string"})
    payload: dict[str, object] = {"type": "record", "kind": "value" if valid else 1}
    if extra:
        payload["added_0"] = 1
    path = tmp_path / "record.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "polylogue.schemas.retained_validation.unread_field_names", lambda _provider: {"kind"} if unread else set()
    )
    verdict = validate_retained_document(
        "claude-code",
        path,
        mode=ValidationMode.ADVISORY,
        raw_id="raw-combined",
        revision_sha256="f" * 64,
        evidence_id="raw-combined",
        schema_resolution=_resolution("v2", explicit_reason=reason),
        schema_resolution_is_explicit=True,
        registry=registry,
        signature_directory=(path).parent,
    )
    observation = verdict.drift_observation
    assert (None if observation is None else observation.classification) == expected
    if observation is not None:
        assert is_risky(observation.classification) is (expected != "new_field")
        assert observation.classification in (
            BENIGN_CLASSIFICATIONS if expected == "new_field" else RISKY_CLASSIFICATIONS
        )
    if expected in {"field_changed", "unseen_shape", "new_field"} and extra:
        assert observation is not None and _signature_text(observation.unseen_key_signature) == "added_0"


def test_retained_new_field_signature_is_sorted_and_repeated_records_do_not_duplicate_it(tmp_path: Path) -> None:
    """Mutation: arrival/key order or repeated field observations change the published signature."""
    registry = _registry(tmp_path, {"type": "string"})
    signatures = []
    rows = [
        {"type": "record", "kind": "value", "beta": 1, "alpha": 1},
        {"alpha": 1, "beta": 1, "kind": "value", "type": "record"},
    ]
    for index, records in enumerate(([rows[0]], [rows[1], rows[1], rows[0]])):
        path = tmp_path / f"records-{index}.jsonl"
        _write_jsonl(path, records)
        verdict = validate_retained_document(
            "claude-code",
            path,
            mode=ValidationMode.ADVISORY,
            raw_id="raw-signature",
            revision_sha256="a" * 64,
            evidence_id="raw-signature",
            jsonl=True,
            schema_resolution=_resolution("v2", explicit_reason="exact_structure"),
            schema_resolution_is_explicit=True,
            registry=registry,
            signature_directory=(path).parent,
        )
        assert verdict.drift_observation is not None
        assert verdict.drift_observation.classification == "new_field"
        signatures.append(_signature_text(verdict.drift_observation.unseen_key_signature))
    assert signatures == ["alpha,beta", "alpha,beta"]


def test_reused_registry_retained_current_historical_and_reload_match_fresh(tmp_path: Path) -> None:
    reader = _registry(tmp_path, {"type": "string"}, {"type": "integer"})
    writer = SchemaRegistry(storage_root=tmp_path / "schemas")
    path = tmp_path / "raw.jsonl"
    _write_jsonl(path, [{"type": "record", "kind": 17}])

    def verdict(registry: SchemaRegistry, *, explicit: bool = False) -> RetainedValidationVerdict:
        return validate_retained_document(
            "claude-code",
            path,
            mode=ValidationMode.STRICT,
            raw_id="raw-currency",
            revision_sha256="c" * 64,
            evidence_id="raw-currency",
            jsonl=True,
            schema_resolution=_resolution("v2"),
            schema_resolution_is_explicit=explicit,
            registry=registry,
            signature_directory=tmp_path,
        )

    historical = verdict(reader)
    assert historical.schema_resolution is not None
    assert historical.schema_resolution.package_version == "v1"
    writer.write_schema_version("claude-code", "v2", _schema({"type": "integer"}), element_kind="session_record_stream")
    current = verdict(reader)
    assert current == verdict(SchemaRegistry(storage_root=tmp_path / "schemas"))
    assert current.schema_resolution is not None
    assert current.schema_resolution.package_version == "v2"
    writer.write_schema_version("claude-code", "v2", _schema({"type": "string"}), element_kind="session_record_stream")
    writer.write_schema_version("claude-code", "v1", _schema({"type": "boolean"}), element_kind="session_record_stream")
    _write_jsonl(path, [{"type": "record", "kind": None}])
    rejected = verdict(reader, explicit=True)
    assert rejected == verdict(SchemaRegistry(storage_root=tmp_path / "schemas"), explicit=True)
    assert rejected.status is ValidationStatus.FAILED


def test_retained_signature_spills_exact_bytes_past_sqlite_cell_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete sorted signature survives the schema owner's closure and keeps its winner."""
    from collections.abc import Generator
    from contextlib import contextmanager

    original_scratch = scratch_connection_context

    @contextmanager
    def small_cells(**kwargs: Any) -> Generator[sqlite3.Connection, None, None]:
        with original_scratch(**kwargs) as connection:
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 32768)
            yield connection

    monkeypatch.setattr(observation_spill, "scratch_connection_context", small_cells)
    registry = _registry(tmp_path, {"type": "string"})
    names = [f"added_{index:05d}" for index in range(8000)]
    path = tmp_path / "wide.jsonl"
    _write_jsonl(path, [{"type": "record", "kind": "value", **dict.fromkeys(reversed(names), 1)}])
    verdict = validate_retained_document(
        "claude-code",
        path,
        mode=ValidationMode.ADVISORY,
        raw_id="wide",
        revision_sha256="a" * 64,
        evidence_id="wide",
        jsonl=True,
        schema_resolution=_resolution("v2", explicit_reason="exact_structure"),
        schema_resolution_is_explicit=True,
        registry=registry,
        signature_directory=tmp_path,
    )
    observation = verdict.drift_observation
    assert observation is not None
    expected = ",".join(names).encode()
    assert observation.unseen_key_signature.byte_count == len(expected) > 32768
    assert b"".join(observation.unseen_key_signature.iter_utf8_chunks()) == expected
    assert verdict.drift_count == len(names)
    from dataclasses import replace

    from polylogue.schemas.retained_validation import _stronger_drift

    earlier = replace(observation, unseen_key_signature=DriftSignature.from_text("aaa", directory=tmp_path))
    assert _stronger_drift(observation, earlier) is earlier
    assert _stronger_drift(earlier, observation) is earlier


@pytest.mark.parametrize("route", ["current", "historical", "all_reject", "explicit", "off"])
def test_retained_winner_acceptance_reuses_validation_but_preserves_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    records = [{"type": "message", "kind": number, "extra": number} for number in range(7)]
    path = tmp_path / "winner.jsonl"
    _write_jsonl(path, records)
    registry = _registry(
        tmp_path,
        {"type": "string"} if route in {"historical", "all_reject"} else {"type": "integer"},
        {"type": "string"} if route == "all_reject" else {"type": "integer"},
    )
    calls = {"acceptance": 0, "reduction": 0}
    original = retained_validation._bounded_validator

    class CountedValidator:
        def __init__(self, validator: Any) -> None:
            self.validator = validator
            self._scratch = validator._scratch

        def is_valid(self, value: object) -> bool:
            calls["acceptance"] += 1
            return bool(self.validator.is_valid(value))

        def iter_errors(self, value: object) -> Any:
            calls["reduction"] += 1
            return self.validator.iter_errors(value)

    def counted(schema: Any, connection: sqlite3.Connection) -> Any:
        return CountedValidator(original(schema, connection))

    monkeypatch.setattr(retained_validation, "_bounded_validator", counted)
    mode = ValidationMode.OFF if route == "off" else ValidationMode.STRICT
    verdict = validate_retained_document(
        Provider.CLAUDE_CODE,
        path,
        mode=mode,
        raw_id="winner",
        revision_sha256="d" * 64,
        evidence_id="winner",
        jsonl=True,
        registry=registry,
        schema_resolution=_resolution("v2"),
        schema_resolution_is_explicit=route == "explicit",
        signature_directory=tmp_path,
    )
    expected = {
        "current": {"acceptance": 7, "reduction": 0},
        "historical": {"acceptance": 8, "reduction": 0},
        "all_reject": {"acceptance": 2, "reduction": 7},
        "explicit": {"acceptance": 0, "reduction": 7},
        "off": {"acceptance": 0, "reduction": 0},
    }
    assert calls == expected[route]
    if route == "off":
        assert verdict.status is ValidationStatus.SKIPPED
        assert verdict.sample_count == 0
        return
    assert verdict.sample_count == 7
    assert verdict.drift_count == 7
    assert verdict.invalid_count == (7 if route == "all_reject" else 0)
    if route != "all_reject":
        assert verdict.error_count == 0
        assert verdict.first_diagnostic is None
    assert verdict.schema_resolution is not None
    version = "v1" if route == "historical" else "v2"
    assert verdict.schema_resolution.package_version == version
    reference = validate_retained_document(
        Provider.CLAUDE_CODE,
        path,
        mode=mode,
        raw_id="winner",
        revision_sha256="d" * 64,
        evidence_id="winner",
        jsonl=True,
        registry=registry,
        schema_resolution=_resolution(version),
        schema_resolution_is_explicit=True,
        signature_directory=tmp_path,
    )
    assert verdict == reference
