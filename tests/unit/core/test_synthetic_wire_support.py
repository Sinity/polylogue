"""Focused contracts for inferred-provider synthetic wire support."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.json import JSONValue
from polylogue.scenarios import CorpusSpec
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus, wire_formats
from polylogue.schemas.synthetic.build_wire_formats import validate_wire_payload
from polylogue.schemas.synthetic.runtime import SCHEMA_CONSTRUCT_HANDLERS
from polylogue.sources import iter_source_sessions


def test_every_catalog_provider_has_an_explicit_route_and_receipt_counts() -> None:
    registry = SchemaRegistry()
    receipt = wire_formats.build_wire_support_receipt(registry=registry)

    assert set(receipt.catalog_providers) == set(registry.list_providers())
    assert not receipt.missing_routes
    assert receipt.supported_count == len(wire_formats.PROVIDER_WIRE_FORMATS)
    assert receipt.unsupported_count == len(wire_formats.PROVIDER_WIRE_ROUTES) - len(wire_formats.PROVIDER_WIRE_FORMATS)
    assert all(entry.reason for entry in receipt.entries if entry.status == "unsupported")


def test_supported_routes_validate_selected_schema_and_parser_entry_point() -> None:
    receipt = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    supported = [entry for entry in receipt.entries if entry.status == "supported"]
    assert supported
    assert all(entry.schema_valid is True for entry in supported)
    assert all(entry.parsed_session_count > 0 for entry in supported)
    assert all(entry.parsed_message_count > 0 for entry in supported)


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


def test_antigravity_metadata_only_payload_is_not_written_as_a_session_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.infra.source_builders import SyntheticAntigravityLanguageServerClient

    spec = CorpusSpec.for_provider(
        "antigravity",
        count=1,
        messages_min=3,
        messages_max=3,
        seed=41,
        session_native_ids=("synthetic-cascade",),
    )
    written = SyntheticCorpus.write_spec_artifacts(spec, tmp_path, prefix="metadata-only")

    # The selected package schema produces metadata-shaped JSON, but the
    # provider route writes the production conversation .pb boundary. A
    # metadata-only JSON artifact must never be presented as a session source.
    assert written.batch.artifacts
    assert all(path.suffix == ".pb" for path in written.files)
    assert not list(tmp_path.glob("*.json"))

    monkeypatch.setattr(
        "polylogue.sources.parsers.antigravity.AntigravityLanguageServerClient",
        SyntheticAntigravityLanguageServerClient,
    )
    sessions = list(iter_source_sessions(Source(name="antigravity", path=tmp_path)))
    assert len(sessions) == 1
    assert sessions[0].provider_session_id == "synthetic-cascade"


def test_construct_handler_removal_changes_coverage_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    before = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    monkeypatch.delitem(SCHEMA_CONSTRUCT_HANDLERS, "array")
    after = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    assert before.to_dict() != after.to_dict()
    assert any(
        entry.construct_coverage is not None and "type:array" in entry.construct_coverage.missing_keywords
        for entry in after.entries
        if entry.status == "supported"
    )


def test_removed_provider_route_changes_explicit_support_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    before = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    monkeypatch.delitem(wire_formats.PROVIDER_WIRE_ROUTES, "codex")
    after = wire_formats.build_wire_support_receipt(registry=SchemaRegistry())

    assert before.to_dict() != after.to_dict()
    assert after.missing_routes == ("codex",)
    assert after.supported_count == before.supported_count - 1


def test_codex_native_id_pinning_preserves_one_wire_shape() -> None:
    corpus = SyntheticCorpus.for_provider("codex")
    [raw] = corpus.generate(count=1, seed=73, messages_per_session=range(3, 4), session_native_ids=("pinned",))
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]

    assert all(record.get("type") != "message" for record in records)
    assert any(record.get("type") == "session_meta" for record in records)
    assert all(record.get("type") in {"session_meta", "response_item"} for record in records)
