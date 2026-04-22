from __future__ import annotations

from datetime import datetime, timezone

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.proof.catalog import (
    VerificationCatalog,
    build_verification_catalog,
    catalog_quality_checks,
    compile_obligations,
)
from polylogue.proof.models import (
    Claim,
    EnvironmentContract,
    Kind,
    RunnerBinding,
    SubjectRef,
    TrustMetadata,
)
from polylogue.proof.subjects import SELECTED_SCHEMA_ANNOTATIONS, command_subjects


def test_default_catalog_compiles_first_vertical_slice() -> None:
    catalog = build_verification_catalog()

    assert {claim.id for claim in catalog.claims} == {
        "cli.command.help",
        "cli.command.no_traceback",
        "cli.command.plain_mode",
        "cli.command.json_envelope",
        "archive.query.provider_filter_consistency",
        "provider.capability.identity_bridge",
        "provider.capability.partial_coverage_declared",
        "schema.values.value_closure",
        "schema.foreign_key.resolves",
        "schema.mutual_exclusion.exclusive",
        "operation.spec.routing_metadata",
        "artifact.path.dependency_closure",
        "maintenance.repair.crash_consistency",
        "parser.quarantine.context_redaction",
        "error.machine_user_context",
        "trace.operation.surface_equivalence",
        "diagnostic.observable_trace_mapping",
        "generated.scenario.family_registered",
        "generated.scenario.local_deterministic",
        "generated.scenario.semantic_claim_mapping",
        "workflow.generated_surfaces_current",
        "workflow.pr_verification_recorded",
    }
    assert catalog.subjects_by_kind()["cli.command"] >= 1
    assert catalog.subjects_by_kind()["cli.json_command"] >= 1
    assert catalog.subjects_by_kind()["archive.query_law"] == 1
    assert catalog.subjects_by_kind()["provider.capability"] >= 2
    assert catalog.subjects_by_kind()["operation.spec"] >= 1
    assert catalog.subjects_by_kind()["artifact.path"] >= 1
    assert catalog.subjects_by_kind()["maintenance.target"] >= 1
    assert catalog.subjects_by_kind()["error.surface"] == 2
    assert catalog.subjects_by_kind()["trace.operation"] == 1
    assert catalog.subjects_by_kind()["diagnostic.observable"] == 1
    assert catalog.subjects_by_kind()["generated.scenario_family"] >= 5
    assert catalog.subjects_by_kind()["schema.annotation"] >= 1
    assert catalog.subjects_by_kind()["workflow.claim"] == 2
    assert {runner.claim_id for runner in catalog.runner_bindings} == {claim.id for claim in catalog.claims}
    assert {"smoke", "semantic", "structural", "workflow"}.issubset(
        {runner.evidence_class for runner in catalog.runner_bindings}
    )
    assert all(runner.environment.controlled_dimensions for runner in catalog.runner_bindings)
    assert all(check.status is OutcomeStatus.OK for check in catalog.quality_checks)


def test_visible_commands_are_not_omitted_from_command_claims() -> None:
    catalog = build_verification_catalog()
    command_ids = {subject.id for subject in command_subjects()}

    for claim_id in ("cli.command.help", "cli.command.no_traceback", "cli.command.plain_mode"):
        obligated_subject_ids = {
            obligation.subject.id for obligation in catalog.obligations if obligation.claim.id == claim_id
        }
        assert obligated_subject_ids == command_ids


def test_selected_schema_annotations_bind_schema_claims() -> None:
    catalog = build_verification_catalog()
    annotation_counts = catalog.schema_subjects_by_annotation()

    assert set(annotation_counts).issubset(set(SELECTED_SCHEMA_ANNOTATIONS))
    assert set(annotation_counts) == set(SELECTED_SCHEMA_ANNOTATIONS)
    assert catalog.obligations_by_claim()["schema.values.value_closure"] == annotation_counts["x-polylogue-values"]
    assert (
        catalog.obligations_by_claim()["schema.foreign_key.resolves"] == annotation_counts["x-polylogue-foreign-keys"]
    )
    assert (
        catalog.obligations_by_claim()["schema.mutual_exclusion.exclusive"]
        == annotation_counts["x-polylogue-mutually-exclusive"]
    )


def test_catalog_self_quality_exposes_catalog_contract_failures() -> None:
    subject = SubjectRef(kind="example", id="missing-source")
    stale_runner = RunnerBinding(
        id="runner:claim",
        claim_id="claim",
        runner="static",
        evidence_class="structural",
        cost_tier="static",
        freshness_policy="test",
        environment=EnvironmentContract(),
        trust=TrustMetadata(
            producer="tests",
            reviewed_at="2020-01-01T00:00:00+00:00",
            expires_at="2020-01-02T00:00:00+00:00",
        ),
    )
    claim = Claim(
        id="claim",
        description="missing breaker and bug class",
        subject_query=Kind("example"),
        evidence_schema={"type": "object"},
    )
    zero_subject_claim = Claim(
        id="zero",
        description="zero subject claim",
        subject_query=Kind("absent"),
        evidence_schema={"type": "object"},
        bug_classes=("bug",),
        tracked_exception="#999",
        runner_classes=("static",),
        observed_facts=("fact",),
        staleness_conditions=("condition",),
    )
    obligations = compile_obligations((subject,), (claim, zero_subject_claim), (stale_runner,))
    catalog = VerificationCatalog(
        subjects=(subject,),
        claims=(claim, zero_subject_claim),
        runner_bindings=(stale_runner,),
        obligations=obligations,
    )

    checks = {
        check.name: check for check in catalog_quality_checks(catalog, now=datetime(2026, 4, 22, tzinfo=timezone.utc))
    }

    assert checks["catalog.subject_source_spans"].status is OutcomeStatus.ERROR
    assert checks["catalog.runner_trust_metadata"].status is OutcomeStatus.ERROR
    assert checks["catalog.runner_environment_dimensions"].status is OutcomeStatus.ERROR
    assert checks["catalog.serious_claim_bug_classes"].status is OutcomeStatus.ERROR
    assert checks["catalog.serious_claim_breakers"].status is OutcomeStatus.ERROR
    assert checks["catalog.serious_claim_adequacy"].status is OutcomeStatus.ERROR
    assert checks["catalog.non_abstract_claim_subjects"].status is OutcomeStatus.ERROR
