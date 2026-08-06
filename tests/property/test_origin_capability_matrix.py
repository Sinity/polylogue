"""Cross-origin laws for the declarative parser capability matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.sources.dispatch import (
    detect_provider_evidence,
    parse_payload,
    require_positive_conversational_evidence,
)
from tests.infra.origin_capability_matrix import (
    CapabilityWitness,
    executable_provider_wires,
    load_manifest,
    load_witness_fixture,
)


def _supported_witnesses() -> tuple[CapabilityWitness, ...]:
    return tuple(
        witness for entry in load_manifest().entries if entry.unsupported is None for witness in entry.witnesses
    )


@pytest.mark.parametrize(
    "witness",
    _supported_witnesses(),
    ids=lambda witness: witness.fallback_id,
)
def test_each_positive_witness_has_one_cross_origin_runtime_claim(witness: CapabilityWitness) -> None:
    """A fixture's detector claim must agree with its declared public origin."""
    assert len(witness.parser_claims) == 1
    claim = witness.parser_claims[0]
    payload = load_witness_fixture(witness)

    detected, evidence = detect_provider_evidence(payload, witness.fixture_path)

    if witness.route == "detected":
        assert detected is claim.provider
        assert evidence.strip()
    else:
        assert detected is Provider.GEMINI
        assert evidence.startswith("drive.looks_like")
    assert claim.provider in executable_provider_wires()
    assert origin_from_provider(claim.provider) is witness.origin

    sessions = parse_payload(claim.provider, payload, witness.fallback_id)
    assert require_positive_conversational_evidence(
        sessions,
        provider=claim.provider,
        source_path=witness.fixture_path,
    )


def test_supported_matrix_claims_exhaust_executable_parser_wires() -> None:
    """Every executable OriginSpec provider wire has one positive witness."""
    witnesses = _supported_witnesses()
    providers = [witness.parser_claims[0].provider for witness in witnesses]
    assert len(providers) == len(set(providers))
    assert set(providers) == set(executable_provider_wires())
    assert all(Path(witness.fixture_path).is_file() for witness in witnesses)
