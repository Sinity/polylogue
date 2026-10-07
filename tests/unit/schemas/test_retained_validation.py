"""Retained validation uses complete replayable records and compact results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.core.enums import ValidationMode, ValidationStatus
from polylogue.schemas.packages import SchemaResolution
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.schemas.validator import SchemaValidator, validate_retained_document


def _schema(kind: object) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"kind": kind},
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
    registry = _registry(tmp_path, "string")
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
    registry = _registry(tmp_path, "string", {"anyOf": [{"type": "string"}, {"type": "integer"}]})

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
    registry = _registry(tmp_path, "string")
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
    registry = _registry(tmp_path, "string")
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
