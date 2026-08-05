"""Focused contracts for inferred-provider synthetic wire support."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.enums import Provider, Role
from polylogue.core.json import JSONValue
from polylogue.schemas import validator as validator_module
from polylogue.schemas.packages import SchemaResolution
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus, wire_formats
from polylogue.schemas.synthetic.build_wire_formats import validate_wire_payload
from polylogue.schemas.synthetic.models import SchemaRecord
from polylogue.schemas.synthetic.runtime import SCHEMA_CONSTRUCT_HANDLERS
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.schemas.synthetic.wire_formats import UnsupportedSyntheticWireRouteError
from polylogue.schemas.validator import SchemaValidator
from polylogue.sources import dispatch as dispatch_module
from polylogue.sources.parsers.base_models import ParsedMessage, ParsedSession
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
    assert all(
        any(witness.artifact_kind == "baseline" and witness.healthy for witness in entry.parser_witnesses)
        for entry in supported
    )
    assert all(all(witness.artifact_evidence for witness in entry.parser_witnesses) for entry in supported)
    assert receipt.complete


def test_parser_witness_loss_is_not_masked_by_aggregate_parsed_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    original_parse_payload = dispatch_module.parse_payload

    def drop_first_coverage_witness(
        provider: str,
        payload: object,
        fallback_id: str,
        _depth: int = 0,
        *,
        schema_resolution: SchemaResolution | None = None,
        source_path: str | None = None,
    ) -> list[ParsedSession]:
        if fallback_id.endswith(":1"):
            return []
        return original_parse_payload(
            provider,
            payload,
            fallback_id,
            _depth,
            schema_resolution=schema_resolution,
            source_path=source_path,
        )

    monkeypatch.setattr(dispatch_module, "parse_payload", drop_first_coverage_witness)
    receipt = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    entry = next(item for item in receipt.entries if item.provider == "chatgpt")
    dropped = next(witness for witness in entry.parser_witnesses if witness.index == 0)
    assert entry.parsed_session_count > 0
    assert dropped.parsed_session_count == 0
    assert dropped.parsed_message_count == 0
    assert not dropped.healthy
    assert not entry.healthy
    assert not receipt.complete


@pytest.mark.parametrize("returned_session", ["empty", "unrelated", "metadata"])
def test_parser_witness_requires_meaningful_evidence_from_its_own_artifact(
    monkeypatch: pytest.MonkeyPatch,
    returned_session: str,
) -> None:
    original_parse_payload = dispatch_module.parse_payload

    def return_non_evidence_for_first_coverage_witness(
        provider: str,
        payload: object,
        fallback_id: str,
        _depth: int = 0,
        *,
        schema_resolution: SchemaResolution | None = None,
        source_path: str | None = None,
    ) -> list[ParsedSession]:
        if provider == "chatgpt" and fallback_id.endswith(":1"):
            if returned_session == "empty":
                return [ParsedSession(source_name=Provider.CHATGPT, provider_session_id="empty", messages=[])]
            if returned_session == "metadata":
                payload_record = payload if isinstance(payload, Mapping) else {}
                metadata_id = str(payload_record.get("id", "metadata-session"))
                metadata_text = str(payload_record.get("title", "metadata title"))
                return [
                    ParsedSession(
                        source_name=Provider.CHATGPT,
                        provider_session_id=metadata_id,
                        messages=[
                            ParsedMessage(
                                provider_message_id=metadata_id,
                                role=Role.ASSISTANT,
                                text=metadata_text,
                            )
                        ],
                    )
                ]
            return [
                ParsedSession(
                    source_name=Provider.CHATGPT,
                    provider_session_id="unrelated",
                    messages=[
                        ParsedMessage(
                            provider_message_id="unrelated",
                            role=Role.ASSISTANT,
                            text="content from another artifact",
                        )
                    ],
                )
            ]
        return original_parse_payload(
            provider,
            payload,
            fallback_id,
            _depth,
            schema_resolution=schema_resolution,
            source_path=source_path,
        )

    monkeypatch.setattr(dispatch_module, "parse_payload", return_non_evidence_for_first_coverage_witness)
    receipt = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    entry = next(item for item in receipt.entries if item.provider == "chatgpt")
    witness = next(item for item in entry.parser_witnesses if item.artifact_kind == "coverage" and item.index == 0)
    assert witness.parsed_session_count == (0 if returned_session == "empty" else 1)
    assert witness.artifact_evidence == ()
    assert not witness.healthy
    assert not entry.healthy
    assert not receipt.complete


def test_baseline_parser_failure_reaches_support_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    original_parse_payload = dispatch_module.parse_payload

    def fail_baseline(
        provider: str,
        payload: object,
        fallback_id: str,
        _depth: int = 0,
        *,
        schema_resolution: SchemaResolution | None = None,
        source_path: str | None = None,
    ) -> list[ParsedSession]:
        if provider == "chatgpt" and fallback_id.endswith(":0"):
            raise RuntimeError("baseline parser failure")
        return original_parse_payload(
            provider,
            payload,
            fallback_id,
            _depth,
            schema_resolution=schema_resolution,
            source_path=source_path,
        )

    monkeypatch.setattr(dispatch_module, "parse_payload", fail_baseline)
    receipt = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    entry = next(item for item in receipt.entries if item.provider == "chatgpt")
    baseline = next(item for item in entry.parser_witnesses if item.artifact_kind == "baseline")
    assert baseline.validation_error == "RuntimeError: baseline parser failure"
    assert entry.validation_error is not None
    assert "baseline parser: RuntimeError: baseline parser failure" in entry.validation_error
    assert not baseline.healthy
    assert not entry.healthy
    assert not receipt.complete


def test_baseline_schema_failure_reaches_support_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    original_validate = validator_module.SchemaValidator.validate
    failed_baseline = False

    def fail_chatgpt_baseline(self: SchemaValidator, payload: JSONValue) -> validator_module.ValidationResult:
        nonlocal failed_baseline
        if not failed_baseline and isinstance(payload, dict) and isinstance(payload.get("mapping"), dict):
            failed_baseline = True
            return validator_module.ValidationResult(is_valid=False, errors=["baseline schema failure"])
        return original_validate(self, payload)

    monkeypatch.setattr(validator_module.SchemaValidator, "validate", fail_chatgpt_baseline)
    receipt = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    entry = next(item for item in receipt.entries if item.provider == "chatgpt")
    baseline = next(item for item in entry.parser_witnesses if item.artifact_kind == "baseline")
    assert failed_baseline
    assert entry.schema_valid is False
    assert baseline.validation_error == "baseline schema failure"
    assert entry.validation_error is not None
    assert "baseline schema failure" in entry.validation_error
    assert not entry.healthy
    assert not receipt.complete


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


def test_selection_reuses_resolved_default_package_for_schema_and_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SchemaRegistry()
    original_get_package = registry.get_package
    original_get_element_schema = registry.get_element_schema
    original_get_workload_profile = registry.get_workload_profile
    resolved_package = original_get_package("chatgpt", version="v1")
    assert resolved_package is not None
    requested_schema_versions: list[str] = []
    requested_profile_versions: list[str] = []

    def divergent_default_package(provider: str, version: str = "default") -> object:
        if provider == "chatgpt" and version == "default":
            return resolved_package
        return original_get_package(provider, version=version)

    def record_schema_version(
        provider: str,
        *,
        version: str = "default",
        element_kind: str | None = None,
    ) -> SchemaRecord | None:
        if provider == "chatgpt":
            requested_schema_versions.append(version)
            if version == "default":
                return original_get_element_schema(provider, version="v2", element_kind=element_kind)
        return original_get_element_schema(provider, version=version, element_kind=element_kind)

    def record_profile_version(provider: str, version: str = "default") -> SchemaRecord | None:
        if provider == "chatgpt":
            requested_profile_versions.append(version)
        return original_get_workload_profile(provider, version=version)

    monkeypatch.setattr(registry, "get_package", divergent_default_package)
    monkeypatch.setattr(registry, "get_element_schema", record_schema_version)
    monkeypatch.setattr(registry, "get_workload_profile", record_profile_version)

    selection = select_synthetic_schema("chatgpt", registry_factory=lambda: registry)

    assert selection.package_version == resolved_package.version
    assert SyntheticCorpus.from_selection(selection).package_version == resolved_package.version
    assert selection.schema == original_get_element_schema(
        "chatgpt",
        version=resolved_package.version,
        element_kind=resolved_package.default_element_kind,
    )
    assert requested_schema_versions == [resolved_package.version]
    assert requested_profile_versions == [resolved_package.version]


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
