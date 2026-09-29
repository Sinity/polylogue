"""Contracts for the declaration-driven source route differential."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.enums import Provider
from polylogue.sources.origin_specs import origin_specs
from tests.infra.source_differential import (
    SourceSpecimen,
    declared_adapters,
    project_sessions,
    run_differential,
)

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "claude-code" / "claude-normalization-main.jsonl"


def test_membership_is_derived_from_origin_specs() -> None:
    spec = next(item for item in origin_specs() if item.origin.value == "claude-code-session")
    adapters = declared_adapters(SourceSpecimen(Provider.CLAUDE_CODE, _FIXTURE.read_bytes()), (spec,))

    assert [adapter.kind for adapter in adapters] == ["eager", "streaming", "replay", "assembly"]
    assert all(adapter.provider is Provider.CLAUDE_CODE for adapter in adapters)
    assert all(adapter.evidence for adapter in adapters)


def test_all_declared_routes_receive_identical_input_and_converge() -> None:
    raw = _FIXTURE.read_bytes()
    report = run_differential(
        SourceSpecimen(
            provider=Provider.CLAUDE_CODE,
            raw_bytes=raw,
            filename=_FIXTURE.name,
            sidecars={"unused.sidecar": b"same bytes"},
        )
    )

    report.assert_complete()
    assert len(report.routes) == 4
    assert {result.input_hash for result in report.routes} == {hashlib.sha256(raw).hexdigest()}
    assert len({result.sidecar_hash for result in report.routes}) == 1
    assert len({result.semantic_hash for result in report.routes}) == 1
    assert report.canonical_hash


@pytest.mark.parametrize("failure", ["duplicate", "missing", "empty", "wrong-provider", "different-input"])
def test_report_rejects_incomplete_or_unmeasured_execution(failure: str) -> None:
    """A self-comparison of executed hashes would admit each missing-evidence case."""
    report = run_differential(SourceSpecimen(Provider.CLAUDE_CODE, _FIXTURE.read_bytes(), filename=_FIXTURE.name))
    report.assert_complete()
    first = report.routes[0]
    if failure == "duplicate":
        routes = (*report.routes, first)
    elif failure == "missing":
        routes = (first,)
    elif failure == "empty":
        routes = tuple(
            replace(route, sessions=(), semantic_hash=hashlib.sha256(b"[]").hexdigest()) for route in report.routes
        )
    elif failure == "wrong-provider":
        routes = (replace(first, adapter=replace(first.adapter, provider=Provider.DRIVE)), *report.routes[1:])
    else:
        routes = (replace(first, input_hash=hashlib.sha256(b"other").hexdigest()), *report.routes[1:])
    with pytest.raises(AssertionError):
        replace(report, routes=routes).assert_complete()


def test_projector_keeps_semantic_axes_and_only_drops_typed_transport_fields() -> None:
    raw = _FIXTURE.read_bytes()
    report = run_differential(SourceSpecimen(provider=Provider.CLAUDE_CODE, raw_bytes=raw, filename=_FIXTURE.name))
    session = report.routes[0].sessions[0]
    messages = cast(list[dict[str, object]], session["messages"])
    assert session["messages"]
    assert "session_events" in session
    assert "created_at" in session
    assert "active_leaf_message_provider_id" in session
    assert "parent_message_position" not in messages[0]


@pytest.mark.parametrize("missing", ["declarations", "routes"])
def test_report_requires_a_declared_and_executed_route_set(missing: str) -> None:
    report = run_differential(SourceSpecimen(Provider.CLAUDE_CODE, _FIXTURE.read_bytes(), filename=_FIXTURE.name))
    with pytest.raises(AssertionError):
        replace(report, **{missing: ()}).assert_complete()


def test_sidecar_enrichment_occurs_once_per_route() -> None:
    """Re-enriching source-walk output appends a second paste span and disagrees with direct parsing."""
    fixture_root = _FIXTURE.parents[1] / "source-differential"
    report = run_differential(
        SourceSpecimen(
            Provider.CLAUDE_CODE,
            (fixture_root / "sidecar-session.jsonl").read_bytes(),
            filename=".claude/projects/synthetic/session.jsonl",
            sidecars={".claude/history.jsonl": (fixture_root / "history.jsonl").read_bytes()},
        )
    )
    assert len(report.routes) == 4
    for route in report.routes:
        assert len(route.sessions) == 1
        messages = cast(list[dict[str, object]], route.sessions[0]["messages"])
        assert len(messages) == 1
        spans = cast(list[dict[str, object]], messages[0]["paste_spans"])
        assert len(spans) == 1
        assert spans[0]["source_marker"] == "1"


def test_idless_specimen_uses_the_production_file_fallback_on_every_route() -> None:
    """An independent eager fallback changes provider_session_id while replay uses the filename stem."""
    raw = (_FIXTURE.parents[1] / "source-differential" / "idless-session.jsonl").read_bytes()
    report = run_differential(SourceSpecimen(Provider.CLAUDE_CODE, raw, filename="fallback-session.jsonl"))
    assert len(report.routes) == 4
    assert {route.sessions[0]["provider_session_id"] for route in report.routes} == {"fallback-session"}


def test_drive_adapters_keep_the_selected_wire_and_exclude_nonexistent_assembly() -> None:
    """Choosing the origin's first wire either mislabels Drive or tries to run Gemini-only assembly."""
    raw = (_FIXTURE.parents[1] / "origin-capability" / "aistudio-drive-hinted.json").read_bytes()
    specimen = SourceSpecimen(Provider.DRIVE, raw, filename="drive.json")
    declarations = declared_adapters(specimen)
    assert [adapter.kind for adapter in declarations] == ["eager", "replay"]
    assert {adapter.provider for adapter in declarations} == {Provider.DRIVE}
    report = run_differential(specimen)
    assert report.declarations == declarations
    assert len(report.routes) == 2
    assert {route.adapter.provider for route in report.routes} == {Provider.DRIVE}


def test_json_atif_specimen_does_not_run_a_jsonl_stream_adapter() -> None:
    raw = (_FIXTURE.parents[1] / "source-differential" / "trajectory.json").read_bytes()
    specimen = SourceSpecimen(Provider.HERMES, raw, filename="trajectory.json")
    assert [adapter.kind for adapter in declared_adapters(specimen)] == ["eager", "replay"]
    report = run_differential(specimen)
    assert len(report.routes) == 2
    assert {route.sessions[0]["provider_session_id"] for route in report.routes} == {"observer:atif:synthetic-session"}


def test_projector_uses_parser_serialization_for_attachment_transport_fields() -> None:
    """The typed model excludes these fields before projection; no recursive filter is another authority."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession

    message = ParsedMessage(
        provider_message_id="message",
        role=Role.normalize("user"),
        parent_message_position=0,
        blocks=[
            ParsedContentBlock(type=BlockType.TEXT, text="body", metadata={"message_position": "semantic-metadata"})
        ],
    )
    attachment = ParsedAttachment(
        provider_attachment_id="attachment",
        message_provider_id="message",
        message_position=0,
        message_variant_index=1,
        inline_bytes=b"content",
        precomputed_blob=("a" * 64, 7),
        name="sample.txt",
    )
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE, provider_session_id="session", messages=[message], attachments=[attachment]
    )
    (projected,) = project_sessions([session])
    attachments = cast(list[dict[str, object]], projected["attachments"])
    assert len(attachments) == 1
    assert attachments[0]["name"] == "sample.txt"
    assert (
        not {"message_position", "message_variant_index", "owner_coordinate", "inline_bytes", "precomputed_blob"}
        & attachments[0].keys()
    )
    messages = cast(list[dict[str, object]], projected["messages"])
    assert len(messages) == 1
    assert "parent_message_position" not in messages[0]
    blocks = cast(list[dict[str, object]], messages[0]["blocks"])
    assert blocks[0]["metadata"] == {"message_position": "semantic-metadata"}
