"""OriginSpec admission-kernel laws (polylogue-2qx.1.1)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from polylogue.core.enums import Origin, Provider
from polylogue.sources.dispatch import RECORD_DETECTOR_PROVIDER_ORDER
from polylogue.sources.origin_specs import (
    ORIGIN_SPEC_REGISTRY,
    ORIGIN_SPECS,
    OriginSpecRegistry,
    validate_dispatch_precedence,
)


def test_origin_spec_pilots_cover_stream_document_and_reserved_lifecycle() -> None:
    """Production dependency: source admission is one typed public-origin registry.

    Anti-vacuity mutation: removing a pilot's parser, fixture, coverage, or
    lifecycle binding makes registration reject its owning OriginSpec.
    """

    by_origin = {spec.origin: spec for spec in ORIGIN_SPECS}

    claude = by_origin[Origin.CLAUDE_CODE_SESSION]
    chatgpt = by_origin[Origin.CHATGPT_EXPORT]
    grok = by_origin[Origin.GROK_EXPORT]

    assert claude.stream_parser_path is not None
    assert claude.detector_tightness == 60
    assert chatgpt.detector_tightness == 70
    assert chatgpt.acquisition_modes == ("takeout-json", "bundle", "browser-capture")
    assert grok.lifecycle == "reserved"
    assert not grok.parser_paths
    assert ORIGIN_SPEC_REGISTRY.diagnostics() == ()


def test_origin_specs_are_parity_checked_against_current_dispatch_order() -> None:
    assert validate_dispatch_precedence(RECORD_DETECTOR_PROVIDER_ORDER) == ()


def test_origin_spec_reports_source_locatable_missing_dispatch_provider() -> None:
    diagnostics = validate_dispatch_precedence((Provider.CODEX,))

    assert [(item.code, item.origin) for item in diagnostics] == [
        ("missing_dispatch_provider", Origin.CHATGPT_EXPORT),
        ("missing_dispatch_provider", Origin.CLAUDE_CODE_SESSION),
    ]
    assert {item.owner_path for item in diagnostics} == {"polylogue/sources/origin_specs.py"}


def test_origin_spec_rejects_missing_fixture_and_noninjective_collision_without_policy() -> None:
    claude = next(spec for spec in ORIGIN_SPECS if spec.origin is Origin.CLAUDE_CODE_SESSION)
    registry = OriginSpecRegistry()

    with pytest.raises(ValueError, match="missing fixture"):
        registry.register(replace(claude, fixture_paths=()))
    with pytest.raises(ValueError, match="collision policy"):
        registry.register(replace(claude, provider_wires=(Provider.CLAUDE_CODE, Provider.DRIVE)))
