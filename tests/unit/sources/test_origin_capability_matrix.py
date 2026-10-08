"""Production-route capability matrix for every public archive origin."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from polylogue.config import Source
from polylogue.core.enums import Origin, Provider
from polylogue.core.json import JSONDocument
from polylogue.core.sources import origin_from_provider
from polylogue.operations.canonical_archive_ingest import ingest_one_shot_archive
from polylogue.sources.dispatch import (
    admit_parsed_sessions_for_publication,
    detect_provider,
    detect_provider_evidence,
    parse_payload,
)
from polylogue.sources.origin_specs import ORIGIN_SPECS
from polylogue.sources.parsers import codex
from polylogue.sources.parsers.antigravity import AntigravitySessionSummary
from polylogue.sources.parsers.base_models import ParsedSession
from tests.infra.origin_capability_matrix import (
    MANIFEST_PATH,
    load_manifest,
    load_manifest_payload,
    load_witness_fixture,
)


class _MatrixAntigravityLanguageServerClient:
    """Fake vendor server used to exercise the production adapter route."""

    def __init__(self, root: Path, payload: JSONDocument) -> None:
        self.root = root
        self.payload = payload

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def search_sessions(self, *, limit: int = 10000, query: str = "") -> list[AntigravitySessionSummary]:
        del limit, query
        summary = AntigravitySessionSummary.from_payload(self.payload)
        assert summary is not None
        return [summary]

    def export_markdown(self, cascade_id: str) -> str:
        assert cascade_id == self.payload["cascadeId"]
        markdown = self.payload["markdown"]
        assert isinstance(markdown, str)
        return markdown


def test_manifest_covers_every_origin_with_typed_support_state() -> None:
    manifest = load_manifest()
    assert {entry.origin for entry in manifest.entries} == set(Origin)

    supported = [entry for entry in manifest.entries if entry.unsupported is None]
    unsupported = [entry for entry in manifest.entries if entry.unsupported is not None]
    assert len(supported) == 11
    assert len(unsupported) == 2
    by_origin = {entry.origin: entry for entry in unsupported}
    assert set(by_origin) == {Origin.BEADS_ISSUE, Origin.UNKNOWN_EXPORT}
    for entry in unsupported:
        assert entry.unsupported is not None
        assert entry.unsupported.status == "unsupported"
        assert entry.unsupported.detail
    beads_receipt = by_origin[Origin.BEADS_ISSUE].unsupported
    unknown_receipt = by_origin[Origin.UNKNOWN_EXPORT].unsupported
    assert beads_receipt is not None
    assert unknown_receipt is not None
    assert beads_receipt.reason == "no-parser"
    assert unknown_receipt.reason == "compatibility-only"
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
            payload = cast(JSONDocument, load_witness_fixture(witness))
            if witness.route == "vendor":
                assert isinstance(payload, dict)
                assert payload.get("source") == "antigravity_language_server"
                assert isinstance(payload.get("cascadeId"), str)
                assert isinstance(payload.get("markdown"), str)
                continue

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
            accepted = admit_parsed_sessions_for_publication(
                sessions,
                provider=claim.provider,
                source_path=witness.fixture_path,
            )
            assert accepted, entry.origin.value


@pytest.mark.asyncio
async def test_supported_witnesses_reach_the_production_archive_ingest_seam(
    one_shot_workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The matrix positives pass through configured-source archive ingestion."""
    manifest = load_manifest()
    supported = [entry for entry in manifest.entries if entry.unsupported is None]
    monkeypatch.setenv("POLYLOGUE_INGEST_PARSE_WORKERS", "1")
    sources: list[Source] = []
    for entry in supported:
        for witness in entry.witnesses:
            if witness.route != "vendor":
                sources.append(Source(name=witness.parser_claims[0].provider.value, path=Path(witness.fixture_path)))
                continue
            payload = load_witness_fixture(witness)
            assert isinstance(payload, dict)
            source_root = tmp_path / witness.fallback_id
            conversations = source_root / "conversations"
            conversations.mkdir(parents=True)
            cascade_id = payload.get("cascadeId")
            assert isinstance(cascade_id, str)
            (conversations / f"{cascade_id}.pb").write_bytes(b"synthetic trajectory")
            monkeypatch.setattr(
                "polylogue.sources.parsers.antigravity.AntigravityLanguageServerClient",
                lambda root, payload=payload: _MatrixAntigravityLanguageServerClient(root, payload),
            )
            sources.append(Source(name=witness.parser_claims[0].provider.value, path=source_root))

    result = await ingest_one_shot_archive(
        one_shot_workspace_env["archive_root"],
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
            admit_parsed_sessions_for_publication(
                sessions,
                provider=case.provider,
                source_path=None,
            )
            == []
        ), case.name
        if detected is not None and detected is not case.provider:
            cross_origin_sessions = parse_payload(detected, case.payload, f"cross-origin-{case.name}")
            assert (
                admit_parsed_sessions_for_publication(
                    cross_origin_sessions,
                    provider=detected,
                    source_path=None,
                )
                == []
            ), case.name


def test_collision_witnesses_follow_real_detector_precedence_and_still_parse() -> None:
    manifest = load_manifest()

    for case in manifest.collisions:
        if case.name == "claude-code-before-codex":
            assert isinstance(case.payload, list)
            assert codex.looks_like(case.payload)
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


def _produced_topology_dimensions(sessions: list[ParsedSession]) -> set[str]:
    """Name each topology dimension the parser actually emitted evidence for."""
    produced: set[str] = set()
    for session in sessions:
        if session.parent_session_provider_id:
            produced.add("session_parent_target")
        if session.branch_point_provider_message_id:
            produced.add("inheritance_branch_point")
        if session.branch_type is not None:
            produced.add("message_branch_state")
        if any(event.payload.get("observation_kind") == "parent_dispatch" for event in session.session_events):
            produced.add("parent_dispatch")
        for message in session.messages:
            if message.parent_message_provider_id or message.parent_message_position is not None:
                produced.add("message_parent")
            # A linear transcript is normalized to variant 0 on the active
            # path; only an off-path message or a non-zero branch or variant
            # is branch evidence.
            if message.branch_index or message.variant_index or message.is_active_path is False:
                produced.add("message_branch_state")
    return produced


def test_no_origin_emits_topology_evidence_its_declaration_calls_structurally_absent() -> None:
    """Declarations are checked against what the parser produces, not against themselves.

    Anti-vacuity (polylogue-b5l6n): set claude-code-session's ``message_parent``
    back to ``structurally-absent`` while its parser still emits ``parentUuid``
    and this goes red; the census completeness test cannot see that direction.
    """
    manifest = load_manifest()
    declarations = {spec.origin: spec.topology_capabilities.as_dict() for spec in ORIGIN_SPECS}
    contradictions: list[str] = []
    checked = 0
    for entry in manifest.entries:
        declared = declarations[entry.origin]
        for witness in entry.witnesses:
            if witness.route == "vendor":
                continue
            claim = witness.parser_claims[0]
            sessions = parse_payload(
                claim.provider,
                cast(JSONDocument, load_witness_fixture(witness)),
                witness.fallback_id,
                source_path=witness.fixture_path,
            )
            checked += 1
            for dimension in sorted(_produced_topology_dimensions(sessions)):
                if declared[dimension].state == "structurally-absent":
                    contradictions.append(f"{entry.origin.value}.{dimension} ({witness.fixture_path})")
    assert checked
    assert contradictions == []


def test_complete_stream_detector_matches_existing_origin_witnesses() -> None:
    from io import BytesIO

    from polylogue.sources.dispatch import detect_provider_from_stream_evidence

    for entry in load_manifest().entries:
        if entry.unsupported is not None:
            continue
        for witness in entry.witnesses:
            if witness.route == "vendor":
                continue
            payload = load_witness_fixture(witness)
            expected, expected_evidence = detect_provider_evidence(payload)
            source = BytesIO(json.dumps(payload).encode())
            observed, observed_evidence = detect_provider_from_stream_evidence(source)
            assert observed is expected, entry.origin.value
            assert observed_evidence == expected_evidence, entry.origin.value
            assert source.tell() == 0
