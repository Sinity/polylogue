"""Contracts for the declaration-driven source route differential."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.enums import Provider
from polylogue.sources.origin_specs import origin_specs
from tests.infra.source_differential import (
    DifferentialReport,
    RouteResult,
    SourceSpecimen,
    declared_adapters,
    run_differential,
)

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "claude-code" / "claude-normalization-main.jsonl"


def test_membership_is_derived_from_origin_specs() -> None:
    spec = next(item for item in origin_specs() if item.origin.value == "claude-code-session")
    adapters = declared_adapters((spec,))

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
            fallback_id="claude-normalization-main",
        )
    )

    report.assert_complete()
    assert len(report.routes) == 4
    assert {result.input_hash for result in report.routes} == {hashlib.sha256(raw).hexdigest()}
    assert len({result.sidecar_hash for result in report.routes}) == 1
    assert len({result.semantic_hash for result in report.routes}) == 1
    assert report.canonical_hash


def test_report_rejects_duplicate_or_missing_execution() -> None:
    raw_hash = hashlib.sha256(b"raw").hexdigest()
    result = RouteResult(
        declared_adapters(origin_specs()[:1])[0],
        raw_hash,
        hashlib.sha256(b"sidecar").hexdigest(),
        (),
        hashlib.sha256(b"semantic").hexdigest(),
    )
    with pytest.raises(AssertionError, match="duplicate adapter execution"):
        DifferentialReport((result, result)).assert_complete()
    with pytest.raises(AssertionError, match="no declared adapters"):
        DifferentialReport(()).assert_complete()


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
