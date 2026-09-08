"""Historical package fallback is limited to rejected default validation."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

from polylogue.schemas.packages import SchemaResolution
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.validator import SchemaValidator

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
