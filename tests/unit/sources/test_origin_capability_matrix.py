"""Production-route capability matrix for every public archive origin."""

from __future__ import annotations

import copy
import json

import pytest

from polylogue.core.enums import Origin
from polylogue.core.sources import origin_from_provider
from polylogue.sources.dispatch import (
    detect_provider,
    detect_provider_evidence,
    parse_payload,
    require_positive_conversational_evidence,
)
from tests.infra.origin_capability_matrix import (
    MANIFEST_PATH,
    load_fixture,
    load_manifest,
    load_manifest_payload,
)


def test_manifest_covers_every_origin_with_typed_support_state() -> None:
    manifest = load_manifest()
    assert {entry.origin for entry in manifest.entries} == set(Origin)

    supported = [entry for entry in manifest.entries if entry.unsupported is None]
    unsupported = [entry for entry in manifest.entries if entry.unsupported is not None]
    assert len(supported) == 11
    assert len(unsupported) == 1
    assert unsupported[0].origin is Origin.UNKNOWN_EXPORT
    assert unsupported[0].unsupported is not None
    assert unsupported[0].unsupported.status == "unsupported"
    assert unsupported[0].unsupported.reason == "compatibility-only"
    assert unsupported[0].unsupported.detail


def test_each_supported_origin_has_one_claim_and_reaches_production_detector_and_parser() -> None:
    manifest = load_manifest()

    for entry in manifest.entries:
        if entry.unsupported is not None:
            assert entry.parser_claims == ()
            continue

        assert len(entry.parser_claims) == 1
        claim = entry.parser_claims[0]
        payload = load_fixture(entry)
        detected, evidence = detect_provider_evidence(payload, entry.fixture_path)

        assert detected is claim.provider, entry.origin.value
        assert origin_from_provider(detected) is entry.origin
        assert evidence.strip(), entry.origin.value

        sessions = parse_payload(
            claim.provider,
            payload,
            entry.fallback_id or "origin-capability",
            source_path=entry.fixture_path,
        )
        accepted = require_positive_conversational_evidence(
            sessions,
            provider=claim.provider,
            source_path=entry.fixture_path,
        )
        assert accepted, entry.origin.value


def test_malformed_witnesses_are_rejected_by_detector_and_content_gate() -> None:
    manifest = load_manifest()

    for case in manifest.malformed:
        detect_provider(case.payload)
        sessions = parse_payload(case.provider, case.payload, f"malformed-{case.name}")
        assert (
            require_positive_conversational_evidence(
                sessions,
                provider=case.provider,
                source_path=None,
            )
            == []
        ), case.name


def test_collision_witnesses_follow_real_detector_precedence_and_still_parse() -> None:
    manifest = load_manifest()

    for case in manifest.collisions:
        detected, evidence = detect_provider_evidence(case.payload)
        assert detected is case.expected_provider, case.name
        assert evidence.strip(), case.name
        sessions = parse_payload(case.expected_provider, case.payload, case.fallback_id)
        assert sessions, case.name


@pytest.mark.parametrize("claim_count", [0, 2])
def test_zero_or_multiple_parser_claims_fail_manifest_validation(claim_count: int) -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    supported = next(item for item in payload["entries"] if item["status"] == "supported")
    claim = supported["parser_claims"][0]
    supported["parser_claims"] = [] if claim_count == 0 else [claim, copy.deepcopy(claim)]

    with pytest.raises(ValueError, match="exactly one parser claim"):
        load_manifest_payload(payload)


def test_unsupported_route_cannot_become_silent_green_support() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    unsupported = next(item for item in payload["entries"] if item["status"] == "unsupported")
    unsupported["status"] = "supported"

    with pytest.raises(ValueError, match="exactly one parser claim"):
        load_manifest_payload(payload)
