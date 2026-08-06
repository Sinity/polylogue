"""Cross-origin laws for the declarative parser capability matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.sources import origin_from_provider
from polylogue.sources.dispatch import (
    detect_provider_evidence,
    parse_payload,
    require_positive_conversational_evidence,
)
from polylogue.sources.origin_specs import ORIGIN_SPECS
from tests.infra.origin_capability_matrix import CapabilityEntry, load_fixture, load_manifest


def _supported_entries() -> tuple[CapabilityEntry, ...]:
    return tuple(entry for entry in load_manifest().entries if entry.unsupported is None)


@pytest.mark.parametrize(
    "entry",
    _supported_entries(),
    ids=lambda entry: entry.origin.value,
)
def test_each_positive_witness_has_one_cross_origin_runtime_claim(entry: CapabilityEntry) -> None:
    """A fixture's detector claim must agree with its declared public origin."""
    assert len(entry.parser_claims) == 1
    claim = entry.parser_claims[0]
    payload = load_fixture(entry)

    detected, evidence = detect_provider_evidence(payload, entry.fixture_path)

    assert detected is claim.provider
    assert evidence.strip()
    assert origin_from_provider(detected) is entry.origin
    assert any(
        spec.origin is entry.origin and spec.lifecycle == "executable" and claim.provider in spec.provider_wires
        for spec in ORIGIN_SPECS
    )

    sessions = parse_payload(claim.provider, payload, entry.fallback_id or entry.origin.value)
    assert require_positive_conversational_evidence(
        sessions,
        provider=claim.provider,
        source_path=entry.fixture_path,
    )


def test_supported_matrix_claims_are_one_to_one_with_parser_providers() -> None:
    """No parser provider is silently claimed by two positive fixtures."""
    entries = _supported_entries()
    providers = [entry.parser_claims[0].provider for entry in entries]
    assert len(providers) == len(set(providers))
    assert all(entry.fixture_path is not None for entry in entries)
    assert all(Path(entry.fixture_path or "").is_file() for entry in entries)
