"""Verification-catalog compiler for proof obligations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

from polylogue.core.json import JSONDocument
from polylogue.lib.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.proof.models import (
    And,
    AttrEq,
    BreakerMetadata,
    Claim,
    CostTier,
    EnvironmentContract,
    EvidenceClass,
    Kind,
    ProofObligation,
    RunnerBinding,
    SubjectRef,
    TrustMetadata,
)
from polylogue.proof.subjects import build_catalog_subjects
from polylogue.storage.backends.schema_ddl import SCHEMA_VERSION

_REVIEWED_AT = "2026-04-22T00:00:00+00:00"
_RUNNER_VERSION = "proof-catalog.v1"
_CONTROLLED_RUNNER_DIMENSIONS = (
    "timezone=UTC",
    "locale=C.UTF-8",
    "terminal_width=runner-default",
    "color_mode=runner-controlled",
    "random_seed=runner-supplied-or-not-used",
    "clock_policy=static-reviewed-at",
    "filesystem_ordering=sorted-subjects",
    "sqlite_features=not-live",
    "feature_flags=none",
)


@dataclass(frozen=True, slots=True)
class VerificationCatalog:
    """Compiled proof catalog with subjects, claims, runners, and obligations."""

    subjects: tuple[SubjectRef, ...]
    claims: tuple[Claim, ...]
    runner_bindings: tuple[RunnerBinding, ...]
    obligations: tuple[ProofObligation, ...]
    quality_checks: tuple[OutcomeCheck, ...] = field(default_factory=tuple)

    def obligations_by_claim(self) -> dict[str, int]:
        return dict(Counter(obligation.claim.id for obligation in self.obligations))

    def subjects_by_kind(self) -> dict[str, int]:
        return dict(Counter(subject.kind for subject in self.subjects))

    def schema_subjects_by_annotation(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for subject in self.subjects:
            annotation = subject.attrs.get("annotation")
            if subject.kind == "schema.annotation" and isinstance(annotation, str):
                counts[annotation] += 1
        return dict(counts)

    def to_payload(self) -> JSONDocument:
        return {
            "subjects": [subject.to_payload() for subject in self.subjects],
            "claims": [claim.to_payload() for claim in self.claims],
            "runner_bindings": [runner.to_payload() for runner in self.runner_bindings],
            "obligations": [obligation.to_payload() for obligation in self.obligations],
            "quality_checks": [check.to_dict() for check in self.quality_checks],
            "summary": {
                "subject_count": len(self.subjects),
                "claim_count": len(self.claims),
                "runner_binding_count": len(self.runner_bindings),
                "obligation_count": len(self.obligations),
                "subjects_by_kind": _counts_payload(self.subjects_by_kind()),
                "schema_subjects_by_annotation": _counts_payload(self.schema_subjects_by_annotation()),
                "obligations_by_claim": _counts_payload(self.obligations_by_claim()),
            },
        }


def build_verification_catalog(
    *,
    subjects: tuple[SubjectRef, ...] | None = None,
    claims: tuple[Claim, ...] | None = None,
    runner_bindings: tuple[RunnerBinding, ...] | None = None,
    now: datetime | None = None,
) -> VerificationCatalog:
    """Build the default verification catalog."""
    catalog_subjects = subjects or build_catalog_subjects()
    catalog_claims = claims or default_claims()
    catalog_runners = runner_bindings or default_runner_bindings(catalog_claims)
    obligations = compile_obligations(catalog_subjects, catalog_claims, catalog_runners)
    catalog = VerificationCatalog(
        subjects=catalog_subjects,
        claims=catalog_claims,
        runner_bindings=catalog_runners,
        obligations=obligations,
    )
    return VerificationCatalog(
        subjects=catalog.subjects,
        claims=catalog.claims,
        runner_bindings=catalog.runner_bindings,
        obligations=catalog.obligations,
        quality_checks=tuple(catalog_quality_checks(catalog, now=now)),
    )


def compile_obligations(
    subjects: Iterable[SubjectRef],
    claims: Iterable[Claim],
    runner_bindings: Iterable[RunnerBinding],
) -> tuple[ProofObligation, ...]:
    """Compile `(subject, claim, runner)` instances for matching claims."""
    subjects_tuple = tuple(subjects)
    claims_tuple = tuple(claims)
    runners_by_claim: dict[str, list[RunnerBinding]] = {}
    for runner in runner_bindings:
        runners_by_claim.setdefault(runner.claim_id, []).append(runner)

    obligations: list[ProofObligation] = []
    for claim in claims_tuple:
        for subject in subjects_tuple:
            if not claim.matches(subject):
                continue
            for runner in runners_by_claim.get(claim.id, []):
                obligations.append(
                    ProofObligation(
                        id=_obligation_id(subject, claim, runner),
                        subject=subject,
                        claim=claim,
                        runner=runner,
                    )
                )
    return tuple(sorted(obligations, key=lambda obligation: obligation.id))


def default_claims() -> tuple[Claim, ...]:
    """Return the first vertical-slice claim set."""
    command_query = Kind("cli.command")
    values_query = _schema_annotation_query("x-polylogue-values")
    foreign_key_query = _schema_annotation_query("x-polylogue-foreign-keys")
    provider_capability_query = Kind("provider.capability")
    artifact_path_query = And(
        (
            Kind("artifact.path"),
            AttrEq("has_durable_layer", True),
            AttrEq("has_non_core_layer", True),
        )
    )
    maintenance_target_query = Kind("maintenance.target")
    parser_quarantine_query = AttrEq("error_family", "parser-quarantine")
    error_surface_query = Kind("error.surface")
    trace_operation_query = Kind("trace.operation")
    observable_diagnostic_query = Kind("diagnostic.observable")
    generated_scenario_query = Kind("generated.scenario_family")
    implemented_generated_scenario_query = And((generated_scenario_query, AttrEq("status", "implemented")))
    return (
        Claim(
            id="cli.command.help",
            description="Every visible command exposes help without failing.",
            subject_query=command_query,
            evidence_schema=_evidence_schema("help_exit_code", "help_output"),
            bug_classes=("cli.help.regression", "command.inventory.omission"),
            runner_classes=("cli_visual",),
            observed_facts=("help_exit_code", "help_usage_banner", "command_path"),
            staleness_conditions=("Click command registration or help option handling changes.",),
            breaker=BreakerMetadata(
                description="A hidden or broken command makes the help runner fail for that command.",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
        Claim(
            id="cli.command.no_traceback",
            description="Visible command help output does not leak Python tracebacks.",
            subject_query=command_query,
            evidence_schema=_evidence_schema("stderr", "stdout"),
            bug_classes=("cli.traceback.leak", "operator-facing-error-regression"),
            runner_classes=("cli_visual",),
            observed_facts=("stdout", "stderr", "traceback_present", "exit_code"),
            staleness_conditions=("Click error handling, command callbacks, or exception formatting changes.",),
            breaker=BreakerMetadata(
                description="A command callback or Click wiring error leaks traceback text into evidence.",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
        Claim(
            id="cli.command.plain_mode",
            description="Visible commands preserve plain-mode operator output contracts.",
            subject_query=command_query,
            evidence_schema=_evidence_schema("plain_stdout", "rich_stdout"),
            bug_classes=("cli.plain-mode.regression", "terminal-rendering-regression"),
            runner_classes=("cli_visual",),
            observed_facts=("plain_stdout", "ansi_present", "command_path"),
            staleness_conditions=("Terminal renderer, rich/plain mode, or command output formatting changes.",),
            breaker=BreakerMetadata(
                description="A rich-only output path breaks the plain-mode runner comparison.",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
        Claim(
            id="cli.command.json_envelope",
            description="Selected JSON-capable commands emit a valid machine envelope.",
            subject_query=Kind("cli.json_command"),
            evidence_schema=_evidence_schema("json_status", "json_result_type", "parse_error"),
            bug_classes=("cli.json-envelope.regression", "machine-contract.invalid-json"),
            runner_classes=("cli_json",),
            observed_facts=("json_status", "json_result_type", "exit_code", "parse_error"),
            staleness_conditions=(
                "Machine-output envelope, selected JSON command list, or JSON serialization changes.",
            ),
            breaker=BreakerMetadata(
                description="A selected JSON command that emits invalid JSON or a missing success envelope breaks the claim.",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
        Claim(
            id="archive.query.provider_filter_consistency",
            description="Provider-filter query results preserve subset, count, and equivalent-construction laws.",
            subject_query=Kind("archive.query_law"),
            evidence_schema=_evidence_schema(
                "all_ids",
                "provider_ids",
                "provider_count",
                "equivalent_provider_ids",
            ),
            bug_classes=("query.provider-filter.drift", "archive-count.semantic-mismatch"),
            runner_classes=("semantic_query",),
            observed_facts=("all_ids", "provider_ids", "provider_count", "equivalent_provider_ids"),
            staleness_conditions=("Repository list/count behavior or equivalent filter construction changes.",),
            breaker=BreakerMetadata(
                description="A provider result outside all results, mismatched count, or divergent equivalent construction is a counterexample.",
                command=("pytest", "tests/unit/proof/test_evidence_runners.py"),
            ),
        ),
        Claim(
            id="provider.capability.identity_bridge",
            description="Provider capability metadata maps native identity facts onto canonical archive fields.",
            subject_query=provider_capability_query,
            evidence_schema=_evidence_schema(
                "native_identity_fields",
                "canonical_identity_fields",
                "identity_mappings",
            ),
            bug_classes=("provider.identity.loss", "normalization.native-fact-drop"),
            runner_classes=("provider_static",),
            observed_facts=("native_identity_fields", "canonical_identity_fields", "identity_mappings"),
            staleness_conditions=(
                "Provider parser identity, provider_meta projection, or conversation/message id semantics change.",
            ),
            breaker=BreakerMetadata(
                description="A provider without native/canonical identity mappings can silently drop provider-native facts.",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
        Claim(
            id="provider.capability.partial_coverage_declared",
            description="Provider capability metadata declares unsupported or partial reasoning, streaming, sidecar, and tool-use facets.",
            subject_query=provider_capability_query,
            evidence_schema=_evidence_schema(
                "reasoning_capability",
                "streaming_capability",
                "sidecar_spec",
                "coverage_facets",
                "partial_coverage",
            ),
            bug_classes=("provider.capability.implicit-gap", "provider-semantics.untracked-partial-support"),
            runner_classes=("provider_static",),
            observed_facts=(
                "reasoning_capability",
                "streaming_capability",
                "sidecar_spec",
                "coverage_facets",
                "partial_coverage",
            ),
            staleness_conditions=(
                "Provider parser support for reasoning, streaming, sidecars, or tool-use variants changes.",
            ),
            breaker=BreakerMetadata(
                description="A provider with absent or partial facets but no explicit gap record hides verification scope.",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
        Claim(
            id="schema.values.value_closure",
            description="Schema value annotations describe a closed, privacy-safe finite value set.",
            subject_query=values_query,
            evidence_schema=_evidence_schema("values", "observed_values"),
            bug_classes=("schema.value-domain.drift", "schema.privacy.enum-leak"),
            runner_classes=("schema_static",),
            observed_facts=("values", "observed_values", "schema_path"),
            staleness_conditions=(
                "Schema annotation grammar, synthetic corpus generation, or provider schema changes.",
            ),
            breaker=BreakerMetadata(
                description="A generated payload outside the annotated value set is a counterexample.",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
        Claim(
            id="schema.foreign_key.resolves",
            description="Schema foreign-key annotations resolve from source paths to target paths.",
            subject_query=foreign_key_query,
            evidence_schema=_evidence_schema("source_path", "target_path"),
            bug_classes=("schema.relationship.drift", "synthetic-corpus.integrity"),
            runner_classes=("schema_static",),
            observed_facts=("source_path", "target_path", "schema_element"),
            staleness_conditions=(
                "Schema relationship annotations or synthetic corpus relationship generation changes.",
            ),
            breaker=BreakerMetadata(
                description="A source path pointing at a missing target path breaks the relation claim.",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
        Claim(
            id="artifact.path.dependency_closure",
            description=(
                "Runtime artifact paths resolve declared dependencies and span durable, derived, index, or projection "
                "layers beyond conversation/message rows."
            ),
            subject_query=artifact_path_query,
            evidence_schema=_evidence_schema("path_name", "nodes", "layers", "missing_dependencies"),
            bug_classes=("artifact-graph.unresolved-dependency", "structural-proof.missing-derived-layer"),
            runner_classes=("artifact_path_static",),
            observed_facts=("path_name", "nodes", "layers", "missing_dependencies", "operation_targets"),
            staleness_conditions=("Artifact graph nodes, runtime paths, operation targets, or repair targets change.",),
            breaker=BreakerMetadata(
                description="A runtime path with unresolved dependencies or no derived/index/projection layer breaks routing.",
                command=("pytest", "tests/unit/proof/test_structural_error_evidence.py"),
            ),
        ),
        Claim(
            id="maintenance.repair.crash_consistency",
            description=(
                "Maintenance repair crash-consistency evidence states preview, execution, idempotence, and failure "
                "state effects."
            ),
            subject_query=maintenance_target_query,
            evidence_schema=_evidence_schema(
                "target",
                "preview_repaired_count",
                "repaired_count",
                "state_effect",
                "failure_state",
            ),
            bug_classes=("maintenance.failure-state.ambiguous", "destructive-repair.preview-mismatch"),
            runner_classes=("maintenance_repair_state",),
            observed_facts=(
                "target",
                "before_count",
                "after_dry_run_count",
                "after_count",
                "state_effect",
                "failure_state",
            ),
            staleness_conditions=("Repair result semantics, doctor preview behavior, or maintenance targets change.",),
            breaker=BreakerMetadata(
                description="A repair failure without an explicit unchanged/changed/rolled-back/partial state is ambiguous.",
                command=("pytest", "tests/unit/proof/test_structural_error_evidence.py"),
            ),
        ),
        Claim(
            id="parser.quarantine.context_redaction",
            description="Parser quarantine evidence keeps provider/source/path context while redacting payload fragments.",
            subject_query=parser_quarantine_query,
            evidence_schema=_evidence_schema("provider", "source_path", "parse_error", "payload_leak_detected"),
            bug_classes=("parser-quarantine.context-loss", "parser-quarantine.payload-leak"),
            runner_classes=("parser_quarantine_error",),
            observed_facts=("provider", "source_path", "raw_id", "parse_error", "payload_leak_detected"),
            staleness_conditions=(
                "Raw ingest envelopes, parser diagnostics, or validation quarantine persistence changes.",
            ),
            breaker=BreakerMetadata(
                description="A quarantine error without source context, or one that echoes private payload text, breaks the claim.",
                command=("pytest", "tests/unit/proof/test_structural_error_evidence.py"),
            ),
        ),
        Claim(
            id="error.machine_user_context",
            description="Error surfaces expose machine-readable context and matching user-facing context checks.",
            subject_query=error_surface_query,
            evidence_schema=_evidence_schema("machine_payload", "user_context_checks", "privacy_checks"),
            bug_classes=("error-envelope.context-loss", "operator-error.unactionable"),
            runner_classes=("error_context",),
            observed_facts=("machine_payload", "user_message", "user_context_checks", "privacy_checks"),
            staleness_conditions=("Machine error envelope, CLI error rendering, or diagnostic context keys change.",),
            breaker=BreakerMetadata(
                description="An error surface that only carries prose, or omits required context keys, is not actionable.",
                command=("pytest", "tests/unit/proof/test_structural_error_evidence.py"),
            ),
        ),
        Claim(
            id="trace.operation.surface_equivalence",
            description="Cross-surface operation traces expose equivalent semantic events and payloads.",
            subject_query=trace_operation_query,
            evidence_schema=_evidence_schema(
                "surface_names",
                "event_names",
                "semantic_signature_hash",
                "trace_payloads",
                "happens_before",
            ),
            bug_classes=("trace.surface-drift", "semantic-result.parity"),
            runner_classes=("trace_equivalence",),
            observed_facts=("surface_names", "event_names", "semantic_signature_hash", "happens_before"),
            staleness_conditions=("Surface query adapters, observable event payloads, or trace vocabulary changes.",),
            breaker=BreakerMetadata(
                description="Equivalent operations with different semantic event signatures expose cross-surface drift.",
                command=("pytest", "tests/unit/proof/test_trace_evidence.py"),
            ),
        ),
        Claim(
            id="diagnostic.observable_trace_mapping",
            description="Existing diagnostics map to observable trace nouns, operations, and artifact nodes.",
            subject_query=observable_diagnostic_query,
            evidence_schema=_evidence_schema(
                "diagnostic_name",
                "event_name",
                "mapped_subject_id",
                "operation",
                "artifact_node",
            ),
            bug_classes=("diagnostic.vocabulary-drift", "probe-proof-unroutable"),
            runner_classes=("diagnostic_trace_mapping",),
            observed_facts=("diagnostic_name", "event_name", "payload_contract", "artifact_node"),
            staleness_conditions=("Pipeline probe diagnostics or observable trace vocabulary changes.",),
            breaker=BreakerMetadata(
                description="A probe diagnostic without a proof-vocabulary mapping cannot route into trace evidence.",
                command=("pytest", "tests/unit/proof/test_trace_evidence.py"),
            ),
        ),
        Claim(
            id="generated.scenario.family_registered",
            description="Generated-world and workload families are registered as proof subjects or explicit migration tasks.",
            subject_query=generated_scenario_query,
            evidence_schema=_evidence_schema("name", "status", "generated_world", "workload_family"),
            bug_classes=("scenario-family.omission", "live-checks.overused"),
            runner_classes=("generated_scenario_static",),
            observed_facts=("name", "status", "generated_world", "workload_family", "reproducer"),
            staleness_conditions=("Scenario inventory, pipeline probes, or validation-lane families change.",),
            breaker=BreakerMetadata(
                description="A missing generated-world family leaves live/archive checks carrying routine confidence.",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
        Claim(
            id="generated.scenario.local_deterministic",
            description="Implemented generated scenario families are deterministic, local, and free of live archive dependencies.",
            subject_query=implemented_generated_scenario_query,
            evidence_schema=_evidence_schema("local_deterministic", "live_archive_dependency", "reproducer"),
            bug_classes=("scenario-family.live-dependency", "scenario-family.nondeterministic-local-fixture"),
            runner_classes=("generated_scenario_static",),
            observed_facts=("local_deterministic", "live_archive_dependency", "reproducer", "status"),
            staleness_conditions=("Generated scenario implementation status or reproducer commands change.",),
            breaker=BreakerMetadata(
                description="An implemented generated scenario with live archive dependency is not a routine proof subject.",
                command=("pytest", "tests/unit/proof/test_generated_scenario_obligations.py"),
            ),
        ),
        Claim(
            id="generated.scenario.semantic_claim_mapping",
            description="Generated scenario families map to semantic claim families instead of only process-output checks.",
            subject_query=generated_scenario_query,
            evidence_schema=_evidence_schema("semantic_claims", "implemented_claim_families", "mapped_claim_families"),
            bug_classes=("scenario-family.vacuous-process-check", "semantic-proof.unmapped-generated-world"),
            runner_classes=("generated_scenario_static",),
            observed_facts=("semantic_claims", "claim_states", "mapping_notes"),
            staleness_conditions=("Semantic claim vocabulary, generated scenarios, or proof-runner mapping changes.",),
            breaker=BreakerMetadata(
                description="A generated scenario without semantic claim mapping can go green without proving meaning.",
                command=("pytest", "tests/unit/proof/test_generated_scenario_obligations.py"),
            ),
        ),
        *_coverage_manifest_claims(),
        *_architecture_control_claims(),
        *_schema_roundtrip_claims(),
        *_effect_implication_claims(),
        *_product_surface_claims(),
    )


def _coverage_manifest_claims() -> tuple[Claim, ...]:
    return (
        Claim(
            id="assurance.coverage.manifest_structured",
            description="Coverage manifests are compiled into proof subjects with item and known-gap counts.",
            subject_query=Kind("assurance.coverage_manifest"),
            evidence_schema=_evidence_schema(
                "manifest_id",
                "assurance_domain",
                "sections",
                "item_count",
                "coverage_gap_count",
            ),
            oracle="construction_sanity",
            oracle_kind="construction_sanity",
            assertion_source="proof_catalog",
            observation_source="same_source_manifest",
            independence_level="cross_checked",
            assurance_domain="spec_completeness",
            bug_classes=("assurance.manifest.omission", "coverage-map.unroutable"),
            runner_classes=("coverage_manifest_static",),
            observed_facts=("manifest_id", "assurance_domain", "sections", "coverage_gap_count"),
            staleness_conditions=("Coverage manifests, assurance domains, or proof subject discovery changes.",),
            breaker=BreakerMetadata(
                description="A coverage manifest that is not represented in the proof catalog stops proof-pack routing.",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
        Claim(
            id="assurance.coverage.item_declared",
            description="Manifest coverage items carry domain, status, path, and automated-gate metadata.",
            subject_query=Kind("assurance.coverage_item"),
            evidence_schema=_evidence_schema(
                "manifest_id",
                "assurance_domain",
                "section",
                "name",
                "status",
                "has_automated_gate",
            ),
            oracle="construction_sanity",
            oracle_kind="construction_sanity",
            assertion_source="proof_catalog",
            observation_source="same_source_manifest",
            independence_level="cross_checked",
            assurance_domain="spec_completeness",
            bug_classes=("coverage-item.missing-domain", "coverage-item.unroutable-path"),
            runner_classes=("coverage_manifest_static",),
            observed_facts=("manifest_id", "section", "name", "status", "has_automated_gate"),
            staleness_conditions=("Coverage item schema, manifest sections, or source-path routing changes.",),
            breaker=BreakerMetadata(
                description="An item without routable coverage metadata cannot contribute to affected proof packs.",
                command=("devtools", "verify-manifests"),
            ),
        ),
        Claim(
            id="assurance.coverage.gap_has_closure_path",
            description="Known coverage gaps are first-class proof subjects with owners and next-evidence paths.",
            subject_query=Kind("assurance.coverage_gap"),
            evidence_schema=_evidence_schema(
                "manifest_id", "assurance_domain", "axis", "gap", "owner", "next_evidence"
            ),
            oracle="construction_sanity",
            oracle_kind="construction_sanity",
            assertion_source="proof_catalog",
            observation_source="same_source_manifest",
            independence_level="cross_checked",
            assurance_domain="spec_completeness",
            bug_classes=("coverage-gap.owner-missing", "proof-pack.known-gap-omission"),
            runner_classes=("coverage_manifest_static",),
            observed_facts=("manifest_id", "axis", "gap", "owner", "next_evidence"),
            staleness_conditions=("Coverage gap manifests or proof-pack known-gap rendering changes.",),
            breaker=BreakerMetadata(
                description="A known coverage gap without an owner and next evidence path cannot drive closure.",
                command=("devtools", "verify-manifests"),
            ),
        ),
    )


def _architecture_control_claims() -> tuple[Claim, ...]:
    return (
        Claim(
            id="architecture.topology.projection_enforced",
            description="Realized package topology is checked against the declared topology projection.",
            subject_query=Kind("architecture.topology"),
            evidence_schema=_evidence_schema("control_path", "command", "runner"),
            oracle="drift_check",
            oracle_kind="drift_check",
            assertion_source="proof_catalog",
            observation_source="repo_static_analysis",
            independence_level="cross_checked",
            assurance_domain="architecture_discipline",
            bug_classes=("topology.pending-cell", "architecture.placement-drift"),
            runner_classes=("architecture_static",),
            observed_facts=("control_path", "command", "runner"),
            staleness_conditions=("Topology projection, placement rules, or realized package tree changes.",),
            breaker=BreakerMetadata(
                description="A missing, orphan, conflict, or pending topology cell breaks structural confidence.",
                command=("devtools", "verify-topology"),
            ),
        ),
        Claim(
            id="architecture.layering.import_rules_enforced",
            description="Package imports obey the declared layering rules.",
            subject_query=Kind("architecture.layering"),
            evidence_schema=_evidence_schema("control_path", "command", "runner"),
            oracle="drift_check",
            oracle_kind="drift_check",
            assertion_source="proof_catalog",
            observation_source="repo_static_analysis",
            independence_level="cross_checked",
            assurance_domain="architecture_discipline",
            bug_classes=("layering.import-violation", "architecture.boundary-drift"),
            runner_classes=("architecture_static",),
            observed_facts=("control_path", "command", "runner"),
            staleness_conditions=("Layering rules, imports, or package placement changes.",),
            breaker=BreakerMetadata(
                description="A cross-layer import violation breaks the architecture discipline gate.",
                command=("devtools", "verify-layering"),
            ),
        ),
        Claim(
            id="architecture.file_budget.loc_enforced",
            description="Declared LOC budgets prevent oversized mixed-responsibility files from silently growing.",
            subject_query=Kind("architecture.file_budget"),
            evidence_schema=_evidence_schema("control_path", "command", "runner"),
            oracle="drift_check",
            oracle_kind="drift_check",
            assertion_source="proof_catalog",
            observation_source="repo_static_analysis",
            independence_level="cross_checked",
            assurance_domain="architecture_discipline",
            bug_classes=("file-budget.exception-stale", "test-owner.mixed-responsibility"),
            runner_classes=("architecture_static",),
            observed_facts=("control_path", "command", "runner"),
            staleness_conditions=("File-size budgets or large test/production files change.",),
            breaker=BreakerMetadata(
                description="A stale exemption or over-budget file breaks the file-budget control.",
                command=("devtools", "verify-file-budgets"),
            ),
        ),
        Claim(
            id="architecture.manifest.consistency_enforced",
            description="Architecture and verification manifests remain internally consistent.",
            subject_query=Kind("architecture.manifest"),
            evidence_schema=_evidence_schema("control_path", "command", "runner"),
            oracle="drift_check",
            oracle_kind="drift_check",
            assertion_source="proof_catalog",
            observation_source="repo_static_analysis",
            independence_level="cross_checked",
            assurance_domain="architecture_discipline",
            bug_classes=("manifest.schema-drift", "verification-map.inconsistent"),
            runner_classes=("architecture_static",),
            observed_facts=("control_path", "command", "runner"),
            staleness_conditions=("docs/plans manifests or manifest lint rules change.",),
            breaker=BreakerMetadata(
                description="A malformed or inconsistent manifest breaks structural proof routing.",
                command=("devtools", "verify-manifests"),
            ),
        ),
        Claim(
            id="architecture.witness.lifecycle_enforced",
            description="Committed witnesses remain classified, exercised, and free of stale xfail markers.",
            subject_query=Kind("architecture.witness"),
            evidence_schema=_evidence_schema("control_path", "command", "runner"),
            oracle="drift_check",
            oracle_kind="drift_check",
            assertion_source="proof_catalog",
            observation_source="repo_static_analysis",
            independence_level="cross_checked",
            assurance_domain="test_quality",
            bug_classes=("witness.stale", "witness.unexercised"),
            runner_classes=("architecture_static",),
            observed_facts=("control_path", "command", "runner"),
            staleness_conditions=("Committed witness metadata, witness tests, or lifecycle rules change.",),
            breaker=BreakerMetadata(
                description="A stale, unclassified, or unexercised witness breaks the lifecycle control.",
                command=("devtools", "verify-witness-lifecycle"),
            ),
        ),
    )


def _schema_roundtrip_claims() -> tuple[Claim, ...]:
    return (
        Claim(
            id="schema.roundtrip.inference_validation",
            description="Committed provider schema packages pass inference-to-validation roundtrip checks.",
            subject_query=Kind("schema.roundtrip"),
            evidence_schema=_evidence_schema("provider_count", "package_count", "element_count", "failures"),
            oracle="proof",
            oracle_kind="proof",
            assertion_source="proof_catalog",
            observation_source="unit_test",
            independence_level="cross_checked",
            assurance_domain="schema_correctness",
            bug_classes=("schema-roundtrip.validation-gap", "schema-package.unreadable"),
            runner_classes=("schema_roundtrip_static",),
            observed_facts=("provider_count", "package_count", "element_count", "failures"),
            staleness_conditions=("Provider schema packages, schema registry, or validation rules change.",),
            breaker=BreakerMetadata(
                description="A committed schema package that cannot be reloaded and validated breaks the roundtrip gate.",
                command=("devtools", "verify-schema-roundtrip", "--all"),
            ),
        ),
    )


def _effect_implication_claims() -> tuple[Claim, ...]:
    """Mint one claim per effect implication.

    Each claim matches operation.spec.effect subjects that carry the
    corresponding implication attr. The oracle and severity derive from
    the implication, not the operation — a single claim covers every
    operation that declares the same effect.
    """
    from polylogue.proof.sources.effect_compiler import (
        IMPLICATION_ORACLE,
        EffectClaimCompiler,
        EffectImplication,
    )

    _implication_descriptions: dict[EffectImplication, str] = {
        "deterministic": "Operation produces the same result for the same input every time.",
        "no_side_effect": "Operation does not mutate any external state.",
        "snapshot_consistent": "Operation reads from a consistent snapshot, not partial writes.",
        "preview": "Operation supports dry-run preview before committing changes.",
        "idempotent": "Running the operation twice produces the same result as once.",
        "rollback_safe": "Failed operations leave no partial state behind.",
        "atomic": "Operation completes entirely or not at all.",
        "path_sanitized": "File paths are validated and do not escape expected directories.",
        "atomic_rename": "Files are written via atomic rename, never partial writes.",
        "parent_exists": "All parent directories exist before file writes.",
        "timeout_bounded": "Network operations have explicit timeout bounds.",
        "retry_bounded": "Network retries are bounded in count and backoff.",
        "sampling_bounded": "Live archive sampling has an explicit row or time budget.",
        "privacy_safe_evidence": "Evidence from live archives is privacy-redacted.",
        "explicit_dry_run_evidence": "Destructive operations produce dry-run evidence before execution.",
        "confirmed_before_execute": "Destructive operations require explicit confirmation.",
    }

    _severe_implications: frozenset[EffectImplication] = frozenset(
        {
            "preview",
            "idempotent",
            "rollback_safe",
            "atomic",
            "path_sanitized",
            "atomic_rename",
            "parent_exists",
            "privacy_safe_evidence",
            "explicit_dry_run_evidence",
            "confirmed_before_execute",
        }
    )

    # Implications whose parent effect (Destructive) is not yet declared
    # by any operation. These claims stay abstract until an operation
    # opts into Destructive.
    _abstract_implications: frozenset[EffectImplication] = frozenset(
        {
            "explicit_dry_run_evidence",
            "confirmed_before_execute",
        }
    )

    effect_query = Kind("operation.spec.effect")
    claims: list[Claim] = []

    for implication in EffectClaimCompiler.implication_names():
        oracle = IMPLICATION_ORACLE.get(implication, "construction_sanity")
        is_severe = implication in _severe_implications
        claims.append(
            Claim(
                id=f"operation.effect.{implication}",
                description=_implication_descriptions.get(
                    implication,
                    f"Effect implication {implication} holds for every operation that declares the parent effect.",
                ),
                subject_query=And((effect_query, AttrEq("implication", implication))),
                evidence_schema=_evidence_schema("operation_name", "effect", "implication"),
                oracle=oracle,
                assurance_domain="operational_resilience",
                bug_classes=("operation.effect.implication_violation",),
                runner_classes=("effect_implication_static",),
                observed_facts=("operation_name", "effect", "implication"),
                staleness_conditions=("Operation effect declarations or implication vocabulary changes.",),
                severity="serious" if is_severe else "info",
                tracked_exception="abstract effect implication requires a concrete runner" if is_severe else None,
                abstract=(implication in _abstract_implications),
            )
        )

    return tuple(claims)


def _product_surface_claims() -> tuple[Claim, ...]:
    """Mint claims for registered product surfaces."""
    return (
        Claim(
            id="product.surface.registered",
            description="Every registered product type has a corresponding proof subject.",
            subject_query=Kind("product.surface"),
            evidence_schema=_evidence_schema("name", "display_name", "json_key"),
            oracle="construction_sanity",
            assurance_domain="surface_parity",
            bug_classes=("product.registry.omission",),
            runner_classes=("product_surface_static",),
            observed_facts=("name", "display_name", "json_key"),
            staleness_conditions=("Product registry entries are added or removed.",),
            severity="info",
        ),
    )


def default_runner_bindings(claims: Iterable[Claim]) -> tuple[RunnerBinding, ...]:
    """Bind every default claim to its first static runner contract."""
    bindings: list[RunnerBinding] = []
    for claim in claims:
        if claim.id in {"cli.command.help", "cli.command.no_traceback"}:
            bindings.append(
                _runner_binding(
                    claim,
                    runner="cli-help-contract",
                    evidence_class="smoke",
                    required_commands=("polylogue",),
                )
            )
        elif claim.id == "cli.command.plain_mode":
            bindings.append(
                _runner_binding(
                    claim,
                    runner="cli-plain-contract",
                    evidence_class="structural",
                    required_commands=("polylogue",),
                )
            )
        elif claim.id == "cli.command.json_envelope":
            bindings.append(
                _runner_binding(
                    claim,
                    runner="cli-json-envelope-contract",
                    evidence_class="structural",
                    cost_tier="unit",
                    required_commands=("polylogue",),
                )
            )
        elif claim.id.startswith("archive.query."):
            bindings.append(
                _runner_binding(
                    claim,
                    runner="semantic-query-law-contract",
                    evidence_class="semantic",
                    cost_tier="unit",
                )
            )
        elif claim.id.startswith("provider.capability."):
            bindings.append(
                _runner_binding(claim, runner="provider-capability-static-contract", evidence_class="structural")
            )
        elif claim.id.startswith("schema.roundtrip."):
            bindings.append(
                _runner_binding(
                    claim,
                    runner="schema-roundtrip-static-contract",
                    evidence_class="structural",
                    required_commands=("devtools",),
                )
            )
        elif claim.id.startswith("schema."):
            bindings.append(
                _runner_binding(claim, runner="schema-annotation-static-contract", evidence_class="structural")
            )
        elif claim.id.startswith("architecture."):
            bindings.append(_runner_binding(claim, runner="architecture-static-contract", evidence_class="structural"))
        elif claim.id.startswith("artifact.path."):
            bindings.append(_runner_binding(claim, runner="artifact-path-static-contract", evidence_class="structural"))
        elif claim.id.startswith("maintenance.repair."):
            bindings.append(
                _runner_binding(
                    claim,
                    runner="maintenance-repair-state-contract",
                    evidence_class="structural",
                    cost_tier="unit",
                )
            )
        elif claim.id.startswith("parser.quarantine."):
            bindings.append(
                _runner_binding(
                    claim,
                    runner="parser-quarantine-error-contract",
                    evidence_class="structural",
                    cost_tier="unit",
                )
            )
        elif claim.id.startswith("error."):
            bindings.append(_runner_binding(claim, runner="error-context-contract", evidence_class="structural"))
        elif claim.id.startswith("trace.operation."):
            bindings.append(
                _runner_binding(
                    claim,
                    runner="trace-equivalence-contract",
                    evidence_class="trace",
                    cost_tier="unit",
                )
            )
        elif claim.id.startswith("diagnostic."):
            bindings.append(
                _runner_binding(
                    claim,
                    runner="diagnostic-trace-mapping-contract",
                    evidence_class="trace",
                )
            )
        elif claim.id.startswith("generated.scenario.semantic_"):
            bindings.append(
                _runner_binding(
                    claim,
                    runner="generated-scenario-static-contract",
                    evidence_class="semantic",
                )
            )
        elif claim.id.startswith("generated.scenario."):
            bindings.append(
                _runner_binding(
                    claim,
                    runner="generated-scenario-static-contract",
                    evidence_class="structural",
                )
            )
        elif claim.id.startswith("assurance.coverage."):
            bindings.append(
                _runner_binding(
                    claim,
                    runner="coverage-manifest-static-contract",
                    evidence_class="structural",
                    required_commands=("devtools",),
                )
            )
        elif claim.id.startswith("operation.effect."):
            bindings.append(
                _runner_binding(claim, runner="effect-implication-static-contract", evidence_class="structural")
            )
        elif claim.id.startswith("product.surface."):
            bindings.append(
                _runner_binding(claim, runner="product-surface-static-contract", evidence_class="structural")
            )
    return tuple(bindings)


def catalog_quality_checks(catalog: VerificationCatalog, *, now: datetime | None = None) -> tuple[OutcomeCheck, ...]:
    """Run self-quality checks over the compiled catalog."""
    now_value = now or datetime.now(tz=timezone.utc)
    obligations_by_claim = catalog.obligations_by_claim()
    checks = [
        _missing_source_span_check(catalog.subjects),
        _stale_trust_metadata_check(catalog.runner_bindings, now=now_value),
        _missing_runner_environment_dimensions_check(catalog.runner_bindings),
        _missing_serious_bug_classes_check(catalog.claims),
        _missing_serious_breakers_check(catalog.claims),
        _missing_serious_claim_adequacy_check(catalog.claims),
        _serious_claim_oracle_independence_check(catalog.claims),
        _zero_subject_claims_check(catalog.claims, obligations_by_claim),
    ]
    return tuple(checks)


def _schema_annotation_query(annotation: str) -> And:
    return And((Kind("schema.annotation"), AttrEq("annotation", annotation)))


def _counts_payload(counts: Mapping[str, int]) -> JSONDocument:
    payload: JSONDocument = {}
    for key, value in counts.items():
        payload[key] = value
    return payload


def _evidence_schema(*required: str) -> JSONDocument:
    return {
        "type": "object",
        "required": list(required),
        "additionalProperties": True,
    }


def _runner_binding(
    claim: Claim,
    *,
    runner: str,
    evidence_class: EvidenceClass,
    cost_tier: CostTier = "static",
    required_commands: tuple[str, ...] = (),
) -> RunnerBinding:
    environment = EnvironmentContract(
        required_commands=required_commands,
        controlled_dimensions=_CONTROLLED_RUNNER_DIMENSIONS,
        uncontrolled_dimensions=(),
        network="none",
        live_archive=False,
        notes=("No live archive dependency; evidence is generated from repo-local metadata or seeded fixtures.",),
    )
    return RunnerBinding(
        id=f"{runner}:{claim.id}",
        claim_id=claim.id,
        runner=runner,
        evidence_class=evidence_class,
        cost_tier=cost_tier,
        freshness_policy="Refresh when the subject compiler, claim metadata, or runner contract changes.",
        environment=environment,
        trust=TrustMetadata(
            producer="polylogue.proof.catalog",
            reviewed_at=_REVIEWED_AT,
            level="authored",
            privacy="repo-local metadata only; no archive payloads",
            code_revision="catalog-reviewed",
            dirty_state=False,
            schema_version=SCHEMA_VERSION,
            environment_fingerprint="catalog-runner-environment-v1",
            runner_version=_RUNNER_VERSION,
            freshness="static review refreshed with generated catalog",
            origin="authored-catalog",
        ),
    )


def _obligation_id(subject: SubjectRef, claim: Claim, runner: RunnerBinding) -> str:
    return f"{claim.id}|{runner.id}|{subject.id}"


def _missing_source_span_check(subjects: tuple[SubjectRef, ...]) -> OutcomeCheck:
    missing = [subject.id for subject in subjects if subject.source_span is None or not subject.source_span.present]
    return _check(
        "catalog.subject_source_spans",
        missing,
        ok_summary="all subjects carry source spans",
        error_summary="subjects missing source spans",
    )


def _stale_trust_metadata_check(runners: tuple[RunnerBinding, ...], *, now: datetime) -> OutcomeCheck:
    stale: list[str] = []
    for runner in runners:
        missing_fields = _missing_runner_trust_fields(runner.trust)
        if missing_fields or runner.trust.expires_before(now):
            stale.append(f"{runner.id}: {', '.join(missing_fields) or 'expired'}")
    return _check(
        "catalog.runner_trust_metadata",
        stale,
        ok_summary="runner trust metadata is present and fresh",
        error_summary="runner trust metadata is stale or incomplete",
    )


def _missing_runner_trust_fields(trust: TrustMetadata) -> list[str]:
    missing: list[str] = []
    for field_name, value in (
        ("producer", trust.producer),
        ("reviewed_at", trust.reviewed_at),
        ("code_revision", trust.code_revision),
        ("dirty_state", trust.dirty_state),
        ("schema_version", trust.schema_version),
        ("environment_fingerprint", trust.environment_fingerprint),
        ("runner_version", trust.runner_version),
        ("freshness", trust.freshness),
        ("origin", trust.origin),
    ):
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field_name)
    return missing


def _missing_runner_environment_dimensions_check(runners: tuple[RunnerBinding, ...]) -> OutcomeCheck:
    missing = [
        runner.id
        for runner in runners
        if not runner.environment.controlled_dimensions and not runner.environment.uncontrolled_dimensions
    ]
    uncontrolled = [
        f"{runner.id}: {', '.join(runner.environment.uncontrolled_dimensions)}"
        for runner in runners
        if runner.environment.uncontrolled_dimensions
    ]
    if missing:
        return OutcomeCheck(
            name="catalog.runner_environment_dimensions",
            status=OutcomeStatus.ERROR,
            summary=f"runner environment dimensions missing: {len(missing)}",
            count=len(missing),
            details=missing[:50],
            breakdown={"missing": len(missing), "uncontrolled": len(uncontrolled)},
        )
    return OutcomeCheck(
        name="catalog.runner_environment_dimensions",
        status=OutcomeStatus.OK,
        summary=f"runner environment dimensions declared; uncontrolled dimensions: {len(uncontrolled)}",
        count=len(uncontrolled),
        details=uncontrolled[:50],
        breakdown={"missing": 0, "uncontrolled": len(uncontrolled)},
    )


def _missing_serious_bug_classes_check(claims: tuple[Claim, ...]) -> OutcomeCheck:
    missing = [claim.id for claim in claims if claim.severity == "serious" and not claim.bug_classes]
    return _check(
        "catalog.serious_claim_bug_classes",
        missing,
        ok_summary="serious claims expose bug classes",
        error_summary="serious claims missing bug classes",
    )


def _missing_serious_breakers_check(claims: tuple[Claim, ...]) -> OutcomeCheck:
    missing = [
        claim.id
        for claim in claims
        if claim.severity == "serious" and claim.breaker is None and claim.tracked_exception is None
    ]
    return _check(
        "catalog.serious_claim_breakers",
        missing,
        ok_summary="serious claims expose breakers or tracked exceptions",
        error_summary="serious claims missing breakers or tracked exceptions",
    )


def _missing_serious_claim_adequacy_check(claims: tuple[Claim, ...]) -> OutcomeCheck:
    missing: list[str] = []
    for claim in claims:
        if claim.severity != "serious":
            continue
        missing_fields = [
            field_name
            for field_name, values in (
                ("runner_classes", claim.runner_classes),
                ("observed_facts", claim.observed_facts),
                ("staleness_conditions", claim.staleness_conditions),
            )
            if not values
        ]
        if missing_fields:
            missing.append(f"{claim.id}: {', '.join(missing_fields)}")
    return _check(
        "catalog.serious_claim_adequacy",
        missing,
        ok_summary="serious claims declare runner classes, observed facts, and staleness conditions",
        error_summary="serious claims missing adequacy metadata",
    )


def _serious_claim_oracle_independence_check(claims: tuple[Claim, ...]) -> OutcomeCheck:
    weak_levels = {"same_source", "self_attesting", "ceremonial"}
    failures = [
        f"{claim.id}: {claim.independence_level}"
        for claim in claims
        if claim.severity == "serious"
        and claim.tracked_exception is None
        and (claim.oracle == "ceremonial" or claim.independence_level in weak_levels)
    ]
    return _check(
        "catalog.serious_claim_oracle_independence",
        failures,
        ok_summary="serious claims avoid ceremonial or self-attesting evidence",
        error_summary="serious claims rely on weak oracle independence",
    )


def _zero_subject_claims_check(claims: tuple[Claim, ...], obligations_by_claim: Mapping[str, int]) -> OutcomeCheck:
    missing = [claim.id for claim in claims if not claim.abstract and obligations_by_claim.get(claim.id, 0) == 0]
    return _check(
        "catalog.non_abstract_claim_subjects",
        missing,
        ok_summary="non-abstract claims bind at least one subject",
        error_summary="non-abstract claims bind zero subjects",
    )


def _check(name: str, failures: list[str], *, ok_summary: str, error_summary: str) -> OutcomeCheck:
    if failures:
        return OutcomeCheck(
            name=name,
            status=OutcomeStatus.ERROR,
            summary=f"{error_summary}: {len(failures)}",
            count=len(failures),
            details=failures[:50],
        )
    return OutcomeCheck(name=name, status=OutcomeStatus.OK, summary=ok_summary)


__all__ = [
    "VerificationCatalog",
    "build_verification_catalog",
    "catalog_quality_checks",
    "compile_obligations",
    "default_claims",
    "default_runner_bindings",
]
