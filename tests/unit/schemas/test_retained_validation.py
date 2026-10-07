"""Retained validation uses complete replayable records and compact results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.enums import ValidationMode, ValidationStatus
from polylogue.schemas import observation_spill
from polylogue.schemas.packages import SchemaResolution
from polylogue.schemas.retained_validation import _bounded_validator, _normalized
from polylogue.schemas.runtime_registry import SCHEMA_DIR, SchemaRegistry
from polylogue.schemas.validator import SchemaValidator, _normalize_empty_arrays, validate_retained_document
from polylogue.storage.sqlite.connection_profile import scratch_connection_context


def _schema(kind: object) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"kind": kind, "type": {"type": "string"}},
        "required": ["kind"],
        "additionalProperties": True,
    }


def _resolution(version: str, *, explicit_reason: str = "package_default") -> SchemaResolution:
    return SchemaResolution(
        provider="claude-code",
        package_version=version,
        element_kind="session_record_stream",
        exact_structure_id=None,
        bundle_scope=None,
        reason=explicit_reason,  # type: ignore[arg-type]
    )


def _registry(tmp_path: Path, current: object, historical: object | None = None) -> SchemaRegistry:
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    if historical is not None:
        registry.write_schema_version("claude-code", "v1", _schema(historical), element_kind="session_record_stream")
    registry.write_schema_version("claude-code", "v2", _schema(current), element_kind="session_record_stream")
    return registry


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_retained_strict_counts_late_failure_and_advisory_accepts(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    _write_jsonl(path, [{"type": "record", "kind": "first"}, {"type": "record", "kind": 17}])
    registry = _registry(tmp_path, {"type": "string"})
    args = {
        "provider": "claude-code",
        "path": path,
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
    )

    assert verdict.status is ValidationStatus.PASSED
    assert verdict.invalid_count == 0
    assert verdict.sample_count == 2
    assert verdict.schema_resolution is not None
    assert verdict.schema_resolution.package_version == "v1"


def test_retained_drift_reduction_is_order_independent(tmp_path: Path) -> None:
    registry = _registry(tmp_path, {"type": "string"})
    resolution = _resolution("v2", explicit_reason="exact_structure")

    def run(name: str, rows: list[object]):
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
        )

    rows = [{"type": "record", "kind": 1, "alpha": 1}, {"type": "record", "kind": 2, "beta": 1}]
    forward = run("forward.jsonl", rows)
    reverse = run("reverse.jsonl", list(reversed(rows)))

    assert forward.drift_observation is not None
    assert reverse.drift_observation is not None
    assert forward.drift_observation.classification == "field_changed"
    assert reverse.drift_observation.classification == "field_changed"
    assert forward.drift_observation.unseen_key_signature == "alpha"
    assert reverse.drift_observation.unseen_key_signature == "alpha"


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
    )

    assert unseen.drift_observation is not None
    assert unseen.drift_observation.classification == "unseen_shape"
    assert unread.drift_observation is not None
    assert unread.drift_observation.classification == "known_field_unread"
    assert unread.drift_observation.unseen_key_signature == "kind"


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
        )
    assert captured["database"]
    assert not Path(captured["database"]).exists()


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
    with scratch_connection_context(
        prefix="polylogue-schema-package-parity-", filename="validation.sqlite"
    ) as connection:
        for provider in registry.list_committed_providers():
            for version in registry.list_committed_versions(provider):
                for schema_file in registry.list_committed_schema_files(provider, version):
                    schema = registry.load_committed_schema_file(provider, version, schema_file)
                    assert schema is not None, (provider, version, schema_file)
                    schema_count += 1
                    for case in cases:
                        expected = Draft202012Validator(schema).is_valid(_normalize_empty_arrays(case, schema))
                        actual = _bounded_validator(schema, connection).is_valid(
                            _normalized(case, schema, schema, connection)
                        )
                        assert actual == expected, (provider, version, schema_file, case)
    assert schema_count == 60
