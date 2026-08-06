"""Production-route capability matrix for every public archive origin."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.enums import Origin, Provider
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.sources.dispatch import (
    detect_provider,
    detect_provider_evidence,
    parse_payload,
    require_positive_conversational_evidence,
)
from tests.infra.origin_capability_matrix import (
    MANIFEST_PATH,
    load_manifest,
    load_manifest_payload,
    load_witness_fixture,
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
    assert sum(len(entry.witnesses) for entry in supported) == 12


def test_each_supported_origin_has_one_claim_and_reaches_production_detector_and_parser() -> None:
    manifest = load_manifest()

    for entry in manifest.entries:
        if entry.unsupported is not None:
            assert entry.witnesses == ()
            continue

        for witness in entry.witnesses:
            assert len(witness.parser_claims) == 1
            claim = witness.parser_claims[0]
            payload = load_witness_fixture(witness)
            detected, evidence = detect_provider_evidence(payload, witness.fixture_path)

            if witness.route == "detected":
                assert detected is claim.provider, entry.origin.value
                assert evidence.strip(), entry.origin.value
            else:
                assert claim.provider is Provider.DRIVE
                assert detected is Provider.GEMINI, entry.origin.value
                assert evidence.startswith("drive.looks_like"), entry.origin.value
            assert origin_from_provider(claim.provider) is entry.origin

            sessions = parse_payload(
                claim.provider,
                payload,
                witness.fallback_id,
                source_path=witness.fixture_path,
            )
            accepted = require_positive_conversational_evidence(
                sessions,
                provider=claim.provider,
                source_path=witness.fixture_path,
            )
            assert accepted, entry.origin.value


@pytest.mark.asyncio
async def test_supported_witnesses_reach_the_production_archive_ingest_seam(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The matrix positives pass through configured-source archive ingestion."""
    manifest = load_manifest()
    supported = [entry for entry in manifest.entries if entry.unsupported is None]
    monkeypatch.setenv("POLYLOGUE_INGEST_PARSE_WORKERS", "1")
    sources = [
        Source(name=witness.parser_claims[0].provider.value, path=Path(witness.fixture_path))
        for entry in supported
        for witness in entry.witnesses
    ]

    result = await parse_sources_archive(
        workspace_env["archive_root"],
        sources,
        parse_workers=1,
    )

    assert result.parse_failures == 0
    assert result.counts["sessions"] >= len(sources)
    assert result.counts["messages"] >= len(sources)
    assert len(result.processed_ids) >= len(sources)


@pytest.mark.parametrize("family_name", ["empty", "partial", "malformed"])
def test_negative_witness_families_are_rejected_by_dispatch_and_content_gate(family_name: str) -> None:
    manifest = load_manifest()
    cases = getattr(manifest, family_name)

    for case in cases:
        detected = detect_provider(case.payload)
        sessions = parse_payload(case.provider, case.payload, f"malformed-{case.name}")
        assert (
            require_positive_conversational_evidence(
                sessions,
                provider=case.provider,
                source_path=None,
            )
            == []
        ), case.name
        if detected is not None and detected is not case.provider:
            cross_origin_sessions = parse_payload(detected, case.payload, f"cross-origin-{case.name}")
            assert (
                require_positive_conversational_evidence(
                    cross_origin_sessions,
                    provider=detected,
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
    claim = supported["witnesses"][0]["parser_claims"][0]
    supported["witnesses"][0]["parser_claims"] = [] if claim_count == 0 else [claim, copy.deepcopy(claim)]

    with pytest.raises(ValueError, match="exactly one parser claim"):
        load_manifest_payload(payload)


def test_unsupported_route_cannot_become_silent_green_support() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    unsupported = next(item for item in payload["entries"] if item["status"] == "unsupported")
    unsupported["status"] = "supported"

    with pytest.raises(ValueError, match="at least one witness"):
        load_manifest_payload(payload)


def test_parser_claim_cannot_cross_origin_spec_boundary() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    supported = next(item for item in payload["entries"] if item["status"] == "supported")
    supported["witnesses"][0]["parser_claims"] = [{"provider": "chatgpt"}]

    with pytest.raises(ValueError, match="not declared by OriginSpec|different public origin"):
        load_manifest_payload(payload)
