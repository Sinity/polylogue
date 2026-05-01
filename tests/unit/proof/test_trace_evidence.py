from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from polylogue.core.outcomes import OutcomeStatus
from polylogue.proof.catalog import build_verification_catalog
from polylogue.proof.models import EvidenceEnvelope, ProofObligation
from polylogue.proof.runners import (
    DiagnosticTraceMappingObservation,
    TraceEquivalenceObservation,
    run_diagnostic_trace_mapping_evidence,
    run_trace_equivalence_evidence,
)
from polylogue.proof.traces import (
    pipeline_probe_archive_subset_mapping,
    provider_filter_query_trace,
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


@pytest.mark.asyncio()
async def test_trace_equivalence_runner_compares_semantic_payloads(
    workspace_env: Mapping[str, Path],
) -> None:
    scenarios = (
        ArchiveScenario(
            name="chatgpt-trace",
            provider="chatgpt",
            messages=(ScenarioMessage(role="user", text="Hello"),),
        ),
        ArchiveScenario(
            name="codex-trace",
            provider="codex",
            messages=(ScenarioMessage(role="assistant", text="Generated code"),),
        ),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, scenarios)
    surfaces = build_archive_surface_set(
        db_path=db_path,
        archive_root=workspace_env["archive_root"],
        scenarios=scenarios,
    )
    try:
        envelope = await _trace_equivalence_envelope(surfaces)
    finally:
        await surfaces.close()

    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["runner_class"] == "trace_equivalence"
    assert envelope.evidence["surface_names"] == ["repository", "facade"]
    assert envelope.evidence["event_names"] == ["ReadArchive", "ApplyFilter", "ReturnRows"]
    assert isinstance(envelope.evidence["semantic_signature_hash"], str)
    assert len(envelope.evidence["semantic_signature_hash"]) == 64
    assert envelope.evidence["happens_before"] == [
        {
            "surface": "repository",
            "edges": [
                {
                    "before_event_id": "repository:read-archive",
                    "after_event_id": "repository:apply-provider-filter",
                    "relation": "total_order",
                },
                {
                    "before_event_id": "repository:apply-provider-filter",
                    "after_event_id": "repository:return-rows",
                    "relation": "total_order",
                },
            ],
        },
        {
            "surface": "facade",
            "edges": [
                {
                    "before_event_id": "facade:read-archive",
                    "after_event_id": "facade:apply-provider-filter",
                    "relation": "total_order",
                },
                {
                    "before_event_id": "facade:apply-provider-filter",
                    "after_event_id": "facade:return-rows",
                    "relation": "total_order",
                },
            ],
        },
    ]
    assert envelope.trust.runner_version == "proof-runners.v1"
    assert envelope.trust.origin == "proof-runner"
    assert envelope.trust.input_fingerprint is not None
    assert envelope.trust.environment_fingerprint is not None
    assert envelope.reproducer == ("pytest", "tests/unit/proof/test_trace_evidence.py")
    assert envelope.counterexample is None
    assert "raw_output" not in str(envelope.evidence["trace_payloads"])


def test_trace_equivalence_runner_rejects_semantic_drift() -> None:
    obligation = _obligation(
        "trace.operation.surface_equivalence",
        subject_id="trace.operation.provider_filter_query",
    )
    observation = TraceEquivalenceObservation(
        traces=(
            provider_filter_query_trace(
                surface="repository",
                provider="codex",
                ids=("codex-trace",),
                subject_id=obligation.subject.id,
                claim_id=obligation.claim.id,
            ),
            provider_filter_query_trace(
                surface="facade",
                provider="codex",
                ids=(),
                subject_id=obligation.subject.id,
                claim_id=obligation.claim.id,
            ),
        )
    )

    envelope = run_trace_equivalence_evidence(obligation, observation)

    assert envelope.status is OutcomeStatus.ERROR
    assert envelope.counterexample is not None


def test_pipeline_probe_diagnostic_maps_to_observable_trace_vocabulary() -> None:
    obligation = _obligation(
        "diagnostic.observable_trace_mapping",
        subject_id="diagnostic.observable.pipeline_probe_archive_subset",
    )
    mapping = pipeline_probe_archive_subset_mapping(
        subject_id=obligation.subject.id,
        claim_id=obligation.claim.id,
    )

    envelope = run_diagnostic_trace_mapping_evidence(
        obligation,
        DiagnosticTraceMappingObservation(mapping=mapping),
    )

    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["runner_class"] == "diagnostic_trace_mapping"
    assert envelope.evidence["diagnostic_name"] == "pipeline-probe.archive-subset.sample"
    assert envelope.evidence["event_name"] == "ReadArchive"
    assert envelope.evidence["mapped_subject_id"] == "diagnostic.observable.pipeline_probe_archive_subset"
    assert envelope.evidence["operation"] == "acquire-raw-conversations"
    assert envelope.evidence["artifact_node"] == "source_payload_stream"
    assert envelope.trust.runner_version == "proof-runners.v1"
    assert envelope.counterexample is None


async def _trace_equivalence_envelope(surfaces: ArchiveSurfaceSet) -> EvidenceEnvelope:
    obligation = _obligation(
        "trace.operation.surface_equivalence",
        subject_id="trace.operation.provider_filter_query",
    )
    repository_surface = next(surface for surface in surfaces.surfaces if surface.name == "repository")
    facade_surface = next(surface for surface in surfaces.surfaces if surface.name == "facade")
    provider_case = ArchiveQueryCase(name="provider-codex", provider="codex", expected_ids=())
    observation = TraceEquivalenceObservation(
        traces=(
            provider_filter_query_trace(
                surface=repository_surface.name,
                provider="codex",
                ids=await repository_surface.query_ids(provider_case),
                subject_id=obligation.subject.id,
                claim_id=obligation.claim.id,
            ),
            provider_filter_query_trace(
                surface=facade_surface.name,
                provider="codex",
                ids=await facade_surface.query_ids(provider_case),
                subject_id=obligation.subject.id,
                claim_id=obligation.claim.id,
            ),
        )
    )
    return run_trace_equivalence_evidence(obligation, observation)
