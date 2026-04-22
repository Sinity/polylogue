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
    EnvironmentContract,
    Kind,
    ProofObligation,
    RunnerBinding,
    SubjectRef,
    TrustMetadata,
)
from polylogue.proof.subjects import build_catalog_subjects

_REVIEWED_AT = "2026-04-22T00:00:00+00:00"


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
    return (
        Claim(
            id="cli.command.help",
            description="Every visible command exposes help without failing.",
            subject_query=command_query,
            evidence_schema=_evidence_schema("help_exit_code", "help_output"),
            bug_classes=("cli.help.regression", "command.inventory.omission"),
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
            breaker=BreakerMetadata(
                description="A rich-only output path breaks the plain-mode runner comparison.",
                issue="#333",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
        Claim(
            id="schema.values.value_closure",
            description="Schema value annotations describe a closed, privacy-safe finite value set.",
            subject_query=values_query,
            evidence_schema=_evidence_schema("values", "observed_values"),
            bug_classes=("schema.value-domain.drift", "schema.privacy.enum-leak"),
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
            breaker=BreakerMetadata(
                description="A generated record containing two fields from the same exclusion group is a counterexample.",
                issue="#332",
                command=("devtools", "render-verification-catalog", "--check"),
            ),
        ),
    )


def default_runner_bindings(claims: Iterable[Claim]) -> tuple[RunnerBinding, ...]:
    """Bind every default claim to its first static runner contract."""
    bindings: list[RunnerBinding] = []
    for claim in claims:
        if claim.id.startswith("cli.command."):
            bindings.append(_runner_binding(claim, runner="cli-help-contract", required_commands=("polylogue",)))
        elif claim.id.startswith("schema."):
            bindings.append(_runner_binding(claim, runner="schema-annotation-static-contract"))
    return tuple(bindings)


def catalog_quality_checks(catalog: VerificationCatalog, *, now: datetime | None = None) -> tuple[OutcomeCheck, ...]:
    """Run self-quality checks over the compiled catalog."""
    now_value = now or datetime.now(tz=timezone.utc)
    obligations_by_claim = catalog.obligations_by_claim()
    checks = [
        _missing_source_span_check(catalog.subjects),
        _stale_trust_metadata_check(catalog.runner_bindings, now=now_value),
        _missing_serious_bug_classes_check(catalog.claims),
        _missing_serious_breakers_check(catalog.claims),
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
    required_commands: tuple[str, ...] = (),
) -> RunnerBinding:
    return RunnerBinding(
        id=f"{runner}:{claim.id}",
        claim_id=claim.id,
        runner=runner,
        cost_tier="static",
        freshness_policy="Refresh when the subject compiler, claim metadata, or runner contract changes.",
        environment=EnvironmentContract(
            required_commands=required_commands,
            network="none",
            live_archive=False,
            notes=("No live archive dependency in the #192 catalog slice.",),
        ),
        trust=TrustMetadata(
            producer="polylogue.proof.catalog",
            reviewed_at=_REVIEWED_AT,
            level="authored",
            privacy="repo-local metadata only; no archive payloads",
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
    stale = [
        runner.id
        for runner in runners
        if not runner.trust.producer.strip() or not runner.trust.reviewed_at.strip() or runner.trust.expires_before(now)
    ]
    return _check(
        "catalog.runner_trust_metadata",
        stale,
        ok_summary="runner trust metadata is present and fresh",
        error_summary="runner trust metadata is stale or incomplete",
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
