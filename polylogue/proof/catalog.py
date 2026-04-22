"""Verification-catalog compiler for proof obligations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

from polylogue.lib.json import JSONDocument
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
    """Return the first vertical-slice claim set for issue #192."""
    command_query = Kind("cli.command")
    values_query = _schema_annotation_query("x-polylogue-values")
    foreign_key_query = _schema_annotation_query("x-polylogue-foreign-keys")
    mutual_exclusion_query = _schema_annotation_query("x-polylogue-mutually-exclusive")
    provider_capability_query = Kind("provider.capability")
    operation_query = Kind("operation.spec")
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
                issue="#333",
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
                issue="#333",
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
                issue="#333",
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
                issue="#333",
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
                issue="#333",
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
                issue="#332",
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
                issue="#332",
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
                issue="#332",
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
                issue="#332",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
        Claim(
            id="schema.mutual_exclusion.exclusive",
            description="Schema mutual-exclusion annotations prevent co-populating exclusive fields.",
            subject_query=mutual_exclusion_query,
            evidence_schema=_evidence_schema("parent", "fields"),
            bug_classes=("schema.mutual-exclusion.drift", "synthetic-corpus.invalid-combination"),
            runner_classes=("schema_static",),
            observed_facts=("parent", "fields", "co_populated_fields"),
            staleness_conditions=("Schema mutual-exclusion annotations or generated payload construction changes.",),
            breaker=BreakerMetadata(
                description="A generated record containing two fields from the same exclusion group is a counterexample.",
                issue="#332",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
        Claim(
            id="operation.spec.routing_metadata",
            description="Declared operation specs expose stable metadata for affected-obligation routing.",
            subject_query=operation_query,
            evidence_schema=_evidence_schema("name", "kind", "surfaces"),
            bug_classes=("operation.routing.metadata-missing", "agent-verification.unroutable-operation"),
            runner_classes=("operation_static",),
            observed_facts=("operation_name", "operation_kind", "surfaces", "path_targets", "code_refs"),
            staleness_conditions=("Operation specs, scenario projections, or artifact graph routing changes.",),
            breaker=BreakerMetadata(
                description="An operation without stable routing metadata cannot be mapped to focused proof checks.",
                issue="#334",
                command=("devtools", "affected-obligations", "--path", "polylogue/operations/specs.py"),
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
                issue="#340",
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
                issue="#340",
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
                issue="#340",
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
                issue="#340",
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
                issue="#341",
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
                issue="#341",
                command=("pytest", "tests/unit/proof/test_trace_evidence.py"),
            ),
        ),
        Claim(
            id="workflow.generated_surfaces_current",
            description="Generated documentation and agent surfaces are current after their sources change.",
            subject_query=AttrEq("claim_family", "generated-surfaces"),
            evidence_schema=_evidence_schema("required_command", "source_changes"),
            bug_classes=("generated-surface.drift", "agent-context.stale-generated-doc"),
            runner_classes=("workflow_static",),
            observed_facts=("required_command", "source_changes"),
            staleness_conditions=("Generated-surface sources or renderers change.",),
            breaker=BreakerMetadata(
                description="Generated docs or AGENTS surfaces drift when render-all is not refreshed.",
                issue="#334",
                command=("devtools", "render-all", "--check"),
            ),
        ),
        Claim(
            id="workflow.pr_verification_recorded",
            description="Durable PR bodies record actual verification commands and issue linkage.",
            subject_query=AttrEq("claim_family", "pr-body"),
            evidence_schema=_evidence_schema("required_sections", "required_linking"),
            bug_classes=("workflow.verification-record.omitted", "workflow.issue-link.omitted"),
            runner_classes=("workflow_static",),
            observed_facts=("required_sections", "required_linking"),
            staleness_conditions=("PR template, contributing workflow, or issue-first policy changes.",),
            breaker=BreakerMetadata(
                description="A non-trivial PR without a verification record or issue reference loses proof provenance.",
                issue="#334",
                command=("devtools", "affected-obligations", "--path", "CONTRIBUTING.md"),
            ),
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
        elif claim.id.startswith("schema."):
            bindings.append(
                _runner_binding(claim, runner="schema-annotation-static-contract", evidence_class="structural")
            )
        elif claim.id.startswith("operation.spec."):
            bindings.append(
                _runner_binding(claim, runner="operation-spec-static-contract", evidence_class="structural")
            )
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
        elif claim.id.startswith("workflow."):
            bindings.append(_runner_binding(claim, runner="workflow-static-contract", evidence_class="workflow"))
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
