"""Historical package fallback is limited to rejected default validation."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.schemas.packages import SchemaResolution
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.validator import SchemaValidator, ValidationResult
from polylogue.storage.runtime import RawSessionRecord

_ELEMENT_KIND = "session_record_stream"


def _schema(message_type: str) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "type": {"const": "assistant"},
            "message": {
                "type": "object",
                "properties": {
                    "type": {"const": "message"},
                    "role": {"const": "assistant"},
                    "content": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"const": "tool_use"},
                                "name": {"const": "SendMessage"},
                                "input": {
                                    "type": "object",
                                    "properties": {"message": {"type": message_type}},
                                    "required": ["message"],
                                    "additionalProperties": True,
                                },
                            },
                            "required": ["type", "name", "input"],
                            "additionalProperties": True,
                        },
                    },
                },
                "required": ["type", "role", "content"],
                "additionalProperties": True,
            },
        },
        "required": ["type", "message"],
        "additionalProperties": True,
    }


def _payload(message: object) -> dict[str, object]:
    return {
        "type": "assistant",
        "message": {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": "SendMessage",
                    "input": {"message": message},
                }
            ],
        },
    }


@pytest.fixture
def schema_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SchemaRegistry:
    registry = SchemaRegistry(storage_root=tmp_path)
    registry.write_schema_version("claude-code", "v1", _schema("object"), element_kind=_ELEMENT_KIND)
    registry.write_schema_version("claude-code", "v2", _schema("string"), element_kind=_ELEMENT_KIND)
    monkeypatch.setattr("polylogue.schemas.validator.SchemaRegistry", partial(SchemaRegistry, storage_root=tmp_path))
    SchemaValidator._cache.clear()
    return registry


def test_rejected_default_uses_compatible_historical_package(schema_registry: SchemaRegistry) -> None:
    """The retained v1 object form must validate after v2 narrows the field."""
    payload = _payload({"type": "shutdown_request", "reason": "done"})
    validator = SchemaValidator.for_payload("claude-code", [payload])

    assert validator.schema["$id"] == "polylogue://schemas/claude-code/v1/session_record_stream"
    assert validator.validate(payload, include_drift=False).is_valid


def test_current_default_payload_stays_on_default_package(schema_registry: SchemaRegistry) -> None:
    payload = _payload("done")

    validator = SchemaValidator.for_payload("claude-code", [payload])

    assert validator.schema["$id"] == "polylogue://schemas/claude-code/v2/session_record_stream"
    assert validator.validate(payload, include_drift=False).is_valid


def test_invalid_payload_does_not_select_historical_package(schema_registry: SchemaRegistry) -> None:
    payload = _payload(7)

    validator = SchemaValidator.for_payload("claude-code", [payload])

    assert validator.schema["$id"] == "polylogue://schemas/claude-code/v2/session_record_stream"
    assert not validator.validate(payload, include_drift=False).is_valid


def test_explicit_package_choice_stays_authoritative(schema_registry: SchemaRegistry) -> None:
    payload = _payload({"type": "shutdown_request", "reason": "done"})
    validator = SchemaValidator.for_payload(
        "claude-code",
        [payload],
        schema_resolution=SchemaResolution(
            provider="claude-code",
            package_version="v2",
            element_kind=_ELEMENT_KIND,
            exact_structure_id=None,
            bundle_scope=None,
            reason="package_default",
        ),
    )

    assert validator.schema["$id"] == "polylogue://schemas/claude-code/v2/session_record_stream"
    assert not validator.validate(payload, include_drift=False).is_valid


def test_inferred_package_choice_can_use_historical_fallback(schema_registry: SchemaRegistry) -> None:
    payload = _payload({"type": "shutdown_request", "reason": "done"})
    payload_validation = SchemaValidator.validate_payload(
        "claude-code",
        [payload],
        schema_resolution=SchemaResolution(
            provider="claude-code",
            package_version="v2",
            element_kind=_ELEMENT_KIND,
            exact_structure_id=None,
            bundle_scope=None,
            reason="package_default",
        ),
        schema_resolution_is_explicit=False,
    )

    assert isinstance(payload_validation.validator, SchemaValidator)
    assert payload_validation.validator.schema["$id"] == "polylogue://schemas/claude-code/v1/session_record_stream"
    assert payload_validation.schema_resolution is not None
    assert payload_validation.schema_resolution.package_version == "v1"
    assert payload_validation.sample_results[0][1].is_valid


def test_strict_ingest_uses_accepted_historical_schema_resolution(
    schema_registry: SchemaRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict ingest carries a legacy stream's accepted schema into parsing."""
    from polylogue.pipeline.services import ingest_worker
    from polylogue.pipeline.services.ingest_worker import _IngestContext, _ParsePlan, ingest_record
    from polylogue.storage.blob_store import get_blob_store

    payload = _payload({"type": "shutdown_request", "reason": "done"})
    raw_content = json.dumps(payload).encode() + b"\n"
    blob_store = get_blob_store()
    raw_id, blob_size = blob_store.write_from_bytes(raw_content)
    raw_record = RawSessionRecord(
        raw_id=raw_id,
        source_name=Provider.CLAUDE_CODE,
        payload_provider=Provider.CLAUDE_CODE,
        source_path="/exports/legacy-claude-code.jsonl",
        blob_size=blob_size,
        acquired_at="2026-01-01T00:00:00Z",
    )
    observed: dict[str, object] = {}
    original_parse_plan_sessions = ingest_worker._parse_plan_sessions

    def capture_parse_plan_sessions(context: _IngestContext, plan: _ParsePlan) -> object:
        observed["schema_resolution"] = plan.schema_resolution
        return original_parse_plan_sessions(context, plan)

    monkeypatch.setattr("polylogue.pipeline.services.ingest_worker._SCHEMA_REGISTRY", schema_registry)
    monkeypatch.setattr(ingest_worker, "_parse_plan_sessions", capture_parse_plan_sessions)

    result = ingest_record(raw_record, str(tmp_path / "archive"), "strict")

    assert result.error is None
    assert result.sessions
    assert isinstance(observed["schema_resolution"], SchemaResolution)
    assert observed["schema_resolution"].package_version == "v1"


def test_payload_validation_reuses_selection_verdicts(
    schema_registry: SchemaRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ingest operation consumes its chosen schema's original verdicts."""
    calls = 0
    original_validate = SchemaValidator.validate

    def count_validate(
        self: SchemaValidator, payload: object, *, include_drift: bool | None = None
    ) -> ValidationResult:
        nonlocal calls
        calls += 1
        return original_validate(self, payload, include_drift=include_drift)

    monkeypatch.setattr(SchemaValidator, "validate", count_validate)
    payload = _payload("done")

    payload_validation = SchemaValidator.validate_payload("claude-code", [payload])

    assert payload_validation.sample_results
    assert calls == 1
