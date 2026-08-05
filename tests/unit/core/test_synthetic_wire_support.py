"""Focused contracts for inferred-provider synthetic wire support."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.json import JSONValue
from polylogue.schemas import validator as validator_module
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus, wire_formats
from polylogue.schemas.synthetic.build_wire_formats import validate_wire_payload
from polylogue.schemas.synthetic.models import SchemaRecord
from polylogue.schemas.synthetic.runtime import SCHEMA_CONSTRUCT_HANDLERS
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.schemas.synthetic.wire_formats import UnsupportedSyntheticWireRouteError
from polylogue.schemas.validator import SchemaValidator
from polylogue.sources.source_parsing import iter_antigravity_language_server_sessions


def test_every_catalog_provider_has_an_explicit_route_and_receipt_counts() -> None:
    registry = SchemaRegistry()
    receipt = wire_formats.build_wire_support_receipt(registry=registry)

    assert set(receipt.catalog_providers) == set(registry.list_providers())
    assert not receipt.missing_routes
    assert receipt.supported_count == sum(
        route.status == "supported" for route in wire_formats.PROVIDER_WIRE_ROUTES.values()
    )
    assert receipt.unsupported_count == sum(
        route.status == "unsupported" for route in wire_formats.PROVIDER_WIRE_ROUTES.values()
    )
    assert all(entry.reason for entry in receipt.entries if entry.status == "unsupported")


def test_supported_routes_validate_selected_schema_and_parser_entry_point() -> None:
    receipt = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    supported = [entry for entry in receipt.entries if entry.status == "supported"]
    assert supported
    assert all(entry.schema_valid is True for entry in supported)
    assert all(entry.parsed_session_count > 0 for entry in supported)
    assert all(entry.parsed_message_count > 0 for entry in supported)
    assert all(entry.construct_coverage is not None and entry.construct_coverage.complete for entry in supported)
    assert receipt.complete


def test_support_receipt_is_deterministic() -> None:
    first = wire_formats.build_wire_support_receipt(registry=SchemaRegistry()).to_dict()
    second = wire_formats.build_wire_support_receipt(registry=SchemaRegistry()).to_dict()

    assert first == second


def test_codex_flat_and_envelope_records_cannot_be_mixed() -> None:
    mixed: JSONValue = [
        {"type": "session_meta", "payload": {"id": "mixed-session"}},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
    ]

    with pytest.raises(ValueError, match="cannot mix flat records"):
        validate_wire_payload("codex", mixed)


def test_catalog_unsupported_routes_fail_selection_with_typed_reason() -> None:
    registry = SchemaRegistry()
    catalog = set(registry.list_providers())
    unsupported = sorted(
        provider
        for provider, route in wire_formats.PROVIDER_WIRE_ROUTES.items()
        if provider in catalog and route.status == "unsupported"
    )

    assert unsupported
    for provider in unsupported:
        with pytest.raises(UnsupportedSyntheticWireRouteError) as exc_info:
            SyntheticCorpus.for_provider(provider)
        assert exc_info.value.provider == provider
        assert exc_info.value.reason == wire_formats.PROVIDER_WIRE_ROUTES[provider].reason


def test_supported_selection_uses_declared_route_format() -> None:
    registry = SchemaRegistry()
    for provider, route in wire_formats.PROVIDER_WIRE_ROUTES.items():
        if route.status != "supported":
            continue
        selection = select_synthetic_schema(provider, registry_factory=lambda: registry)
        assert selection.wire_format == route.wire_format


def test_antigravity_metadata_only_source_has_no_language_server_session(tmp_path: Path) -> None:
    metadata_path = tmp_path / "brain" / "work-session" / "plan.md.metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps({"artifactType": "plan", "summary": "metadata only"}), encoding="utf-8")

    sessions = list(iter_antigravity_language_server_sessions(Source(name="antigravity", path=tmp_path)))

    assert sessions == []


def test_construct_handler_removal_changes_coverage_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    before = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    monkeypatch.delitem(SCHEMA_CONSTRUCT_HANDLERS, "array")
    after = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    assert before.to_dict() != after.to_dict()
    assert not after.complete
    assert any(
        entry.construct_coverage is not None
        and any(keyword.startswith("type:array") for keyword in entry.construct_coverage.missing_keywords)
        for entry in after.entries
        if entry.status == "supported"
    )


def test_string_payload_cannot_satisfy_integer_coverage() -> None:
    coverage = wire_formats.construct_coverage(
        {"type": "integer"},
        ("not an integer",),
    )

    assert coverage.missing_keywords == ("type:integer",)
    assert not coverage.complete


def test_crossed_property_values_cannot_satisfy_each_others_types() -> None:
    schema: SchemaRecord = {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "label": {"type": "string"},
        },
        "required": ["count", "label"],
    }
    coverage = wire_formats.construct_coverage(schema, ({"count": "wrong", "label": 7},))

    assert any(keyword.startswith("type:integer@$.properties.count") for keyword in coverage.missing_keywords)
    assert any(keyword.startswith("type:string@$.properties.label") for keyword in coverage.missing_keywords)
    assert not coverage.complete


def test_frequency_optional_field_is_an_obligation_even_when_omitted() -> None:
    schema: SchemaRecord = {
        "type": "object",
        "properties": {
            "required_value": {"type": "string"},
            "optional_value": {"type": "integer", "x-polylogue-frequency": 0.0},
        },
        "required": ["required_value"],
    }

    coverage = wire_formats.construct_coverage(schema, ({"required_value": "present"},))

    optional_path = "type:integer@$.properties.optional_value"
    assert optional_path in coverage.schema_keywords
    assert optional_path in coverage.missing_keywords
    assert not coverage.complete


def test_coverage_witnesses_select_nested_array_union_branches() -> None:
    schema: SchemaRecord = {
        "type": "object",
        "properties": {
            "choice": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
            "items": {
                "type": "array",
                "items": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
            },
            "typed_choice": {"type": ["object", "string"]},
        },
    }
    corpus = SyntheticCorpus(schema, wire_formats.WireFormat(encoding="json"), "test")

    payloads = [json.loads(raw) for raw in wire_formats.generate_coverage_witnesses(corpus, seed=31)]

    assert {type(payload["choice"]) for payload in payloads} >= {int, str}
    assert {type(payload["items"][0]) for payload in payloads} >= {int, str}
    assert {type(payload["typed_choice"]) for payload in payloads} >= {dict, str}


def test_coverage_witnesses_keep_nested_union_choices_independent() -> None:
    schema: SchemaRecord = {
        "oneOf": [
            {"type": "string"},
            {"oneOf": [{"type": "integer"}, {"type": "boolean"}]},
        ]
    }
    corpus = SyntheticCorpus(schema, wire_formats.WireFormat(encoding="json"), "test")

    payloads = [json.loads(raw) for raw in wire_formats.generate_coverage_witnesses(corpus, seed=17, max_witnesses=8)]

    assert any(isinstance(payload, str) for payload in payloads)
    assert 0 in payloads
    assert True in payloads


def test_receipt_generation_and_validation_use_injected_registry_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SchemaRegistry()
    selected_schema = registry.get_element_schema("chatgpt", version="default", element_kind="session_document")
    assert selected_schema is not None
    divergent_schema = deepcopy(selected_schema)
    properties = divergent_schema.setdefault("properties", {})
    assert isinstance(properties, dict)
    properties["memory_scope"] = {"type": "integer", "x-polylogue-frequency": 1.0}
    properties["registry_only_marker"] = {
        "type": "integer",
        "x-polylogue-frequency": 0.0,
    }
    required = divergent_schema.setdefault("required", [])
    assert isinstance(required, list)
    required.append("memory_scope")
    original_loader = registry.get_element_schema

    def load_divergent_schema(
        provider: str,
        *,
        version: str = "default",
        element_kind: str | None = None,
    ) -> SchemaRecord | None:
        if provider == "chatgpt" and element_kind == "session_document":
            return divergent_schema
        return original_loader(provider, version=version, element_kind=element_kind)

    monkeypatch.setattr(registry, "get_element_schema", load_divergent_schema)
    captured_schemas: list[Mapping[str, object]] = []

    class CapturingValidator(SchemaValidator):
        def __init__(
            self,
            schema: Mapping[str, object],
            strict: bool = True,
            provider: Provider | None = None,
        ) -> None:
            captured_schemas.append(schema)
            super().__init__(schema, strict=strict, provider=provider)

    monkeypatch.setattr(validator_module, "SchemaValidator", CapturingValidator)
    receipt = wire_formats.build_wire_support_receipt(registry=registry)

    entry = next(item for item in receipt.entries if item.provider == "chatgpt")
    assert entry.schema_valid is True
    assert entry.healthy
    assert any(schema is divergent_schema for schema in captured_schemas)
    assert entry.construct_coverage is not None
    marker = "type:integer@$.properties.registry_only_marker"
    assert marker in entry.construct_coverage.schema_keywords
    assert marker in entry.construct_coverage.exercised_keywords


def test_claude_code_route_only_waives_unrepresentable_nested_content() -> None:
    receipt = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())
    entry = next(item for item in receipt.entries if item.provider == "claude-code")

    assert entry.construct_coverage is not None
    assert entry.construct_coverage.complete
    assert not any(
        "snapshot.properties.trackedFileBackups" in keyword
        for keyword in entry.construct_coverage.nonrepresentable_keywords
    )
    assert all(
        "$.properties.message.properties.content.anyOf[1].items[*].properties.content.anyOf[1]" in keyword
        for keyword in entry.construct_coverage.nonrepresentable_keywords
    )


def test_unmatched_union_does_not_count_as_exercised() -> None:
    schema: SchemaRecord = {
        "oneOf": [{"type": "integer"}, {"type": "string"}],
    }
    coverage = wire_formats.construct_coverage(schema, (True,))

    assert "oneOf" in coverage.missing_keywords
    assert not coverage.complete


def test_missing_required_and_additional_property_evidence_makes_receipt_incomplete() -> None:
    schema: SchemaRecord = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": False,
    }
    coverage = wire_formats.construct_coverage(schema, ({"unexpected": "text"},))
    entry = wire_formats.WireSupportEntry(
        provider="synthetic",
        status="supported",
        reason=None,
        package_version="test",
        element_kind="session_document",
        schema_valid=True,
        parsed_session_count=1,
        parsed_message_count=1,
        construct_coverage=coverage,
    )
    receipt = wire_formats.WireSupportReceipt(
        catalog_providers=("synthetic",),
        entries=(entry,),
        missing_routes=(),
    )

    assert "required" in coverage.missing_keywords
    assert "additionalProperties" in coverage.missing_keywords
    assert not receipt.complete


def test_removed_provider_route_changes_explicit_support_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    before = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    monkeypatch.delitem(wire_formats.PROVIDER_WIRE_ROUTES, "codex")
    after = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    assert before.to_dict() != after.to_dict()
    assert after.missing_routes == ("codex",)
    assert after.supported_count == before.supported_count - 1
    with pytest.raises(UnsupportedSyntheticWireRouteError, match="no explicit synthetic wire route"):
        SyntheticCorpus.for_provider("codex")


def test_codex_native_id_pinning_preserves_one_wire_shape() -> None:
    corpus = SyntheticCorpus.for_provider("codex")
    [raw] = corpus.generate(count=1, seed=73, messages_per_session=range(3, 4), session_native_ids=("pinned",))
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]

    assert all(record.get("type") != "message" for record in records)
    assert any(record.get("type") == "session_meta" for record in records)
    assert all(record.get("type") in {"session_meta", "response_item"} for record in records)
