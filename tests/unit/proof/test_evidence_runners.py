from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from polylogue.core.outcomes import OutcomeStatus
from polylogue.proof.catalog import build_verification_catalog
from polylogue.proof.models import EvidenceEnvelope, ProofObligation
from polylogue.proof.runners import (
    SemanticQueryObservation,
    run_cli_json_envelope_evidence,
    run_cli_visual_evidence,
    run_semantic_query_evidence,
)
from tests.infra.archive_scenarios import ArchiveScenario, ScenarioMessage, seed_workspace_scenarios
from tests.infra.query_cases import ArchiveQueryCase
from tests.infra.surfaces import ArchiveSurfaceSet, build_archive_surface_set


def _obligation(claim_id: str, *, subject_id: str | None = None) -> ProofObligation:
    catalog = build_verification_catalog()
    for obligation in catalog.obligations:
        if obligation.claim.id != claim_id:
            continue
        if subject_id is not None and obligation.subject.id != subject_id:
            continue
        return obligation
    raise AssertionError(f"missing obligation for claim={claim_id!r} subject={subject_id!r}")


def test_cli_help_runner_emits_ok_evidence() -> None:
    obligation = _obligation("cli.command.help", subject_id="polylogue doctor")

    envelope = run_cli_visual_evidence(obligation)

    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["subject_id"] == "polylogue doctor"
    assert envelope.evidence["runner_class"] == "cli_visual"
    assert envelope.evidence["help_exit_code"] == 0
    assert envelope.trust.runner_version == "proof-runners.v1"
    assert envelope.trust.origin == "proof-runner"
    assert envelope.trust.schema_version == 3
    assert envelope.trust.input_fingerprint is not None
    assert len(envelope.trust.input_fingerprint) == 64
    assert envelope.trust.environment_fingerprint is not None
    assert len(envelope.trust.environment_fingerprint) == 64
    assert envelope.counterexample is None


def test_cli_json_envelope_runner_emits_ok_evidence(cli_workspace: Mapping[str, Path]) -> None:
    obligation = _obligation("cli.command.json_envelope", subject_id="polylogue doctor --format json")

    envelope = run_cli_json_envelope_evidence(obligation)

    assert cli_workspace["db_path"].exists()
    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["runner_class"] == "cli_json"
    assert envelope.evidence["json_status"] == "ok"
    assert envelope.counterexample is None


@pytest.mark.asyncio()
async def test_semantic_query_law_runner_emits_ok_evidence(workspace_env: Mapping[str, Path]) -> None:
    scenarios = (
        ArchiveScenario(
            name="chatgpt-semantic",
            provider="chatgpt",
            messages=(ScenarioMessage(role="user", text="Hello"),),
        ),
        ArchiveScenario(
            name="codex-semantic",
            provider="codex",
            messages=(ScenarioMessage(role="user", text="Generate code"),),
        ),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, scenarios)
    surfaces = build_archive_surface_set(
        db_path=db_path,
        archive_root=workspace_env["archive_root"],
        scenarios=scenarios,
    )
    try:
        envelope = await _semantic_query_envelope(surfaces)
    finally:
        await surfaces.close()

    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["runner_class"] == "semantic_query"
    assert envelope.evidence["provider_ids"] == ["codex-semantic"]
    assert envelope.evidence["provider_count"] == 1
    assert envelope.counterexample is None


async def _semantic_query_envelope(surfaces: ArchiveSurfaceSet) -> EvidenceEnvelope:
    obligation = _obligation(
        "archive.query.provider_filter_consistency",
        subject_id="archive.query_law.provider_filter.codex",
    )
    repository_surface = next(surface for surface in surfaces.surfaces if surface.name == "repository")
    facade_surface = next(surface for surface in surfaces.surfaces if surface.name == "facade")
    provider_case = ArchiveQueryCase(name="provider-codex", provider="codex", expected_ids=())
    archive_facts = await repository_surface.archive_facts()
    observation = SemanticQueryObservation(
        provider="codex",
        all_ids=archive_facts.conversation_ids,
        provider_ids=await repository_surface.query_ids(provider_case),
        provider_count=await repository_surface.query_count(provider_case),
        equivalent_provider_ids=await facade_surface.query_ids(provider_case),
        surface_names=(repository_surface.name, facade_surface.name),
    )
    return run_semantic_query_evidence(obligation, observation)
