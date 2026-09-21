"""Reconcile one provider-schema generation pass against its denominator.

A generation pass produces one receipt per subject it reached. That is not an
account of the run: the subjects it never reached leave no trace at all, and a
zero-diff receipt looks identical whether the route read the whole corpus or
never opened a file.

This module turns a set of run receipts into an account where every declared
subject carries exactly one outcome:

``generated``
    The persisted commit route ran non-vacuously and at least one package
    version is new or changed.
``zero_diff``
    The route ran non-vacuously and every version is structurally unchanged.
    Accepted only with route evidence and a reconciled denominator -- without
    both, a zero diff is indistinguishable from the generator not running.
``proven_non_applicable``
    The declared frontier admits no eligible material for the subject and says
    why. The proof is the recorded, checked member list, not an empty scan.
``declared_non_applicable``
    The subject is outside the inference denominator by declaration.
``failed``
    The route ran and refused, errored, narrowed, or produced an unexplained
    zero sample count.
``not_run``
    No receipt exists for a subject the denominator requires. Recorded, never
    omitted.

The denominator comes from :mod:`polylogue.schemas.provider_denominator`, which
reads neither the packages nor the receipts, so a subject can be missing from
every artifact of the run and still be counted here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONDocument
from polylogue.schemas.provider_denominator import ProviderDenominator, derive_provider_denominator
from polylogue.schemas.source_frontier import FrontierCheck, SchemaFrontier

SubjectOutcome: TypeAlias = Literal[
    "generated",
    "zero_diff",
    "proven_non_applicable",
    "declared_non_applicable",
    "failed",
    "not_run",
]

#: Outcomes AC4 requires to be empty before a pass is acceptable.
BLOCKING_OUTCOMES: frozenset[str] = frozenset({"failed", "not_run"})

RECONCILIATION_SCHEMA = "polylogue.provider-schema-reconciliation.v1"


@dataclass(frozen=True, slots=True)
class SubjectDenominatorCounts:
    """What the frontier admitted for a subject, and what the run consumed."""

    baseline_members: int
    baseline_bytes: int
    members_missing_since_baseline: int
    members_added_since_baseline: int
    live_members: int
    candidates_inventoried: int | None
    candidates_included: int | None
    candidate_terminal_outcomes: Mapping[str, int]

    @property
    def conserves(self) -> bool:
        """Whether the run inventoried exactly the live admitted member set.

        ``candidates_inventoried`` is the production route's own count of
        physical candidates; ``live_members`` is the frontier's count of
        admitted members after applying the enumerated drift. Equality is the
        conservation claim: nothing was sampled away and nothing was read from
        outside the declared denominator.
        """

        if self.candidates_inventoried is None:
            return False
        if self.candidates_inventoried != self.live_members:
            return False
        return sum(self.candidate_terminal_outcomes.values()) == self.candidates_inventoried

    def to_payload(self) -> JSONDocument:
        return {
            "baseline_members": self.baseline_members,
            "baseline_bytes": self.baseline_bytes,
            "members_missing_since_baseline": self.members_missing_since_baseline,
            "members_added_since_baseline": self.members_added_since_baseline,
            "live_members": self.live_members,
            "candidates_inventoried": self.candidates_inventoried,
            "candidates_included": self.candidates_included,
            "candidate_terminal_outcomes": dict(sorted(self.candidate_terminal_outcomes.items())),
            "conserves": self.conserves,
        }


@dataclass(frozen=True, slots=True)
class SubjectReconciliation:
    """Exactly one recorded outcome for one declared subject."""

    subject: str
    outcome: SubjectOutcome
    reason: str
    counts: SubjectDenominatorCounts | None = None
    sample_count: int | None = None
    versions: tuple[JSONDocument, ...] = ()
    input_manifest_digest: str | None = None
    narrowed_paths: int = 0
    added_paths: int = 0

    def to_payload(self) -> JSONDocument:
        return {
            "subject": self.subject,
            "outcome": self.outcome,
            "reason": self.reason,
            "sample_count": self.sample_count,
            "input_manifest_digest": self.input_manifest_digest,
            "narrowed_paths": self.narrowed_paths,
            "added_paths": self.added_paths,
            "versions": list(self.versions),
            "denominator": self.counts.to_payload() if self.counts is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ProviderMatrix:
    """The aggregate account, bound to everything that decided it."""

    subjects: tuple[SubjectReconciliation, ...]
    denominator: ProviderDenominator
    baseline_digest: str | None
    frontier_declaration_digest: str
    code_revision: str
    generator_semantics: JSONDocument
    inference_configuration: JSONDocument
    blockers: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.blockers

    def counts_by_outcome(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for item in self.subjects:
            tally[item.outcome] = tally.get(item.outcome, 0) + 1
        return dict(sorted(tally.items()))

    def _payload_without_digest(self) -> JSONDocument:
        return {
            "schema": RECONCILIATION_SCHEMA,
            "baseline_digest": self.baseline_digest,
            "frontier_declaration_digest": self.frontier_declaration_digest,
            "provider_declaration_digest": self.denominator.declaration_digest,
            "code_revision": self.code_revision,
            "generator_semantics": self.generator_semantics,
            "inference_configuration": self.inference_configuration,
            "outcome_counts": self.counts_by_outcome(),
            "subjects": [item.to_payload() for item in self.subjects],
            "denominator": self.denominator.to_payload(),
            "blockers": list(self.blockers),
        }

    @property
    def matrix_digest(self) -> str:
        return hash_payload(self._payload_without_digest())

    def to_payload(self) -> JSONDocument:
        return {**self._payload_without_digest(), "matrix_digest": self.matrix_digest}

    def format_text(self) -> str:
        lines = [
            f"provider-matrix: {'OK' if self.ok else 'BLOCKED'}",
            f"  baseline_digest={self.baseline_digest}",
            f"  provider_declaration_digest={self.denominator.declaration_digest}",
            f"  code_revision={self.code_revision}",
        ]
        for item in self.subjects:
            counts = item.counts
            suffix = ""
            if counts is not None:
                suffix = (
                    f" denominator={counts.candidates_inventoried}/{counts.live_members}"
                    f" conserves={'yes' if counts.conserves else 'NO'}"
                )
            lines.append(f"  {item.subject}: {item.outcome} samples={item.sample_count}{suffix}")
            lines.append(f"      {item.reason}")
        for blocker in self.blockers:
            lines.append(f"  BLOCKER: {blocker}")
        return "\n".join(lines)


def _subject_counts(
    subject: str,
    frontier: SchemaFrontier,
    check: FrontierCheck,
    source: Mapping[str, object] | None,
) -> SubjectDenominatorCounts:
    baselines = [item for item in frontier.baselines if item.subject == subject]
    missing = sum(1 for finding in check.findings if finding.subject == subject and finding.kind == "member_missing")
    added = sum(1 for finding in check.findings if finding.subject == subject and finding.kind == "member_added")
    baseline_members = sum(item.member_count for item in baselines)
    terminal_raw = (source or {}).get("source_candidate_terminal_outcomes")
    terminal = {
        str(key): int(value)
        for key, value in (terminal_raw.items() if isinstance(terminal_raw, Mapping) else ())
        if isinstance(value, int)
    }
    inventoried = (source or {}).get("source_candidate_count")
    included = (source or {}).get("source_included_candidate_count")
    return SubjectDenominatorCounts(
        baseline_members=baseline_members,
        baseline_bytes=sum(item.byte_count for item in baselines),
        members_missing_since_baseline=missing,
        members_added_since_baseline=added,
        live_members=baseline_members - missing + added,
        candidates_inventoried=inventoried if isinstance(inventoried, int) else None,
        candidates_included=included if isinstance(included, int) else None,
        candidate_terminal_outcomes=terminal,
    )


def _zero_material_reason(subject: str, frontier: SchemaFrontier) -> str | None:
    declared = frontier.subject(subject)
    if declared is None:
        return None
    reasons = [root.zero_material_reason for root in declared.roots if root.zero_material_reason]
    if len(reasons) != len(declared.roots):
        return None
    return "; ".join(reasons)


def _reconcile_subject(
    subject: str,
    disposition: str,
    declared_reason: str | None,
    frontier: SchemaFrontier,
    check: FrontierCheck,
    receipt: Mapping[str, object] | None,
) -> SubjectReconciliation:
    if disposition == "declared_non_applicable":
        return SubjectReconciliation(
            subject=subject,
            outcome="declared_non_applicable",
            reason=declared_reason or "declared outside the schema-inference denominator",
        )
    if disposition == "no_executable_route":
        return SubjectReconciliation(
            subject=subject,
            outcome="proven_non_applicable",
            reason=(
                f"no executable acquisition route: {declared_reason}"
                if declared_reason
                else "no executable acquisition route admits material for this subject"
            ),
        )

    counts = _subject_counts(subject, frontier, check, None)
    if receipt is None:
        zero_reason = _zero_material_reason(subject, frontier)
        if zero_reason is not None and counts.live_members == 0:
            return SubjectReconciliation(
                subject=subject,
                outcome="proven_non_applicable",
                reason=(f"the checked frontier admits 0 members across every declared root: {zero_reason}"),
                counts=counts,
                sample_count=0,
            )
        return SubjectReconciliation(
            subject=subject,
            outcome="not_run",
            reason="the denominator requires this subject and the pass produced no receipt for it",
            counts=counts,
        )

    result = receipt.get("result")
    exit_code = receipt.get("exit_code")
    if isinstance(result, Mapping) and result.get("terminal") == "zero_eligible_material":
        zero_reason = _zero_material_reason(subject, frontier)
        if zero_reason is None or counts.live_members != 0:
            return SubjectReconciliation(
                subject=subject,
                outcome="failed",
                reason=(
                    "the route declared zero eligible material, but the checked frontier admits "
                    f"{counts.live_members} member(s) for this subject"
                ),
                counts=counts,
                sample_count=0,
            )
        return SubjectReconciliation(
            subject=subject,
            outcome="proven_non_applicable",
            reason=f"the checked frontier admits 0 members across every declared root: {zero_reason}",
            counts=counts,
            sample_count=0,
        )
    if not isinstance(result, Mapping):
        return SubjectReconciliation(
            subject=subject,
            outcome="failed",
            reason=f"the commit route produced no machine result (exit={exit_code})",
            counts=counts,
        )

    phase = result.get("phase_receipt")
    source = phase.get("source") if isinstance(phase, Mapping) else None
    source_map = source if isinstance(source, Mapping) else None
    counts = _subject_counts(subject, frontier, check, source_map)
    versions = tuple(item for item in result.get("versions", ()) if isinstance(item, Mapping))
    sample_count = result.get("sample_count")
    sample_count = sample_count if isinstance(sample_count, int) else None
    manifest_digest = source_map.get("source_input_manifest_digest") if source_map else None
    narrowed = sum(len(item.get("narrowed_paths", ())) for item in versions)
    added = sum(len(item.get("added_paths", ())) for item in versions)

    if exit_code != 0 or not result.get("success"):
        return SubjectReconciliation(
            subject=subject,
            outcome="failed",
            reason=f"the commit route exited {exit_code}: {result.get('error') or 'generation failed'}",
            counts=counts,
            sample_count=sample_count,
            versions=versions,
            input_manifest_digest=manifest_digest if isinstance(manifest_digest, str) else None,
            narrowed_paths=narrowed,
            added_paths=added,
        )
    if narrowed:
        return SubjectReconciliation(
            subject=subject,
            outcome="failed",
            reason=f"{narrowed} previously committed leaf type(s) narrowed without operator adjudication",
            counts=counts,
            sample_count=sample_count,
            versions=versions,
            input_manifest_digest=manifest_digest if isinstance(manifest_digest, str) else None,
            narrowed_paths=narrowed,
            added_paths=added,
        )

    included = counts.candidates_included or 0
    if included == 0 or not sample_count:
        zero_reason = _zero_material_reason(subject, frontier)
        if zero_reason is not None and counts.live_members == 0:
            outcome, reason = (
                "proven_non_applicable",
                f"the checked frontier admits 0 members across every declared root: {zero_reason}",
            )
        else:
            outcome, reason = (
                "failed",
                (
                    f"unexplained zero sample count: {counts.live_members} member(s) are admitted by the "
                    f"declared frontier and {included} candidate(s) were included"
                ),
            )
        return SubjectReconciliation(
            subject=subject,
            outcome=outcome,
            reason=reason,
            counts=counts,
            sample_count=sample_count,
            versions=versions,
            input_manifest_digest=manifest_digest if isinstance(manifest_digest, str) else None,
        )

    changed = [item for item in versions if item.get("status") in {"new", "changed"}]
    route = (
        f"route read {included} of {counts.candidates_inventoried} inventoried candidate(s) "
        f"over {counts.live_members} admitted member(s) and produced {sample_count} sample(s)"
    )
    if changed:
        outcome = "generated"
        reason = f"{len(changed)} package version(s) new or changed; {route}"
    elif not counts.conserves:
        outcome = "failed"
        reason = (
            "zero-diff is not acceptable without a reconciled denominator: the route inventoried "
            f"{counts.candidates_inventoried} candidate(s) against {counts.live_members} admitted member(s)"
        )
    else:
        outcome = "zero_diff"
        reason = f"every committed version is structurally unchanged; {route}"
    return SubjectReconciliation(
        subject=subject,
        outcome=outcome,
        reason=reason,
        counts=counts,
        sample_count=sample_count,
        versions=versions,
        input_manifest_digest=manifest_digest if isinstance(manifest_digest, str) else None,
        narrowed_paths=narrowed,
        added_paths=added,
    )


def _generator_semantics(receipts: Mapping[str, Mapping[str, object]]) -> JSONDocument:
    """Collect the generator semantics revision every receipt reports.

    A pass that mixed two generator revisions is not one measurement, so the
    fingerprints are collected as sets and a disagreement stays visible in the
    bound metadata instead of being averaged away.
    """

    collected: dict[str, set[str]] = {}
    for receipt in receipts.values():
        result = receipt.get("result")
        phase = result.get("phase_receipt") if isinstance(result, Mapping) else None
        source = phase.get("source") if isinstance(phase, Mapping) else None
        recipe = source.get("source_recipe") if isinstance(source, Mapping) else None
        if not isinstance(recipe, Mapping):
            continue
        for key in ("implementation_fingerprint", "structure_fingerprint", "statistics_fingerprint"):
            value = recipe.get(key)
            if isinstance(value, str):
                collected.setdefault(key, set()).add(value)
    return {key: sorted(values) for key, values in sorted(collected.items())}


def reconcile_provider_matrix(
    *,
    frontier: SchemaFrontier,
    check: FrontierCheck,
    receipts: Mapping[str, Mapping[str, object]],
    code_revision: str,
    inference_configuration: JSONDocument,
    denominator: ProviderDenominator | None = None,
) -> ProviderMatrix:
    """Account for every declared subject exactly once."""

    resolved = denominator if denominator is not None else derive_provider_denominator()
    subjects = tuple(
        _reconcile_subject(
            item.subject,
            item.disposition,
            item.reason,
            frontier,
            check,
            receipts.get(item.subject),
        )
        for item in resolved.subjects
    )

    blockers: list[str] = list(resolved.findings)
    if not check.ok:
        blockers.append(f"the declared frontier does not match the live roots: {len(check.errors)} error(s)")
    for finding in check.errors:
        blockers.append(f"frontier {finding.kind}: {finding.subject} -- {finding.detail}")
    for item in subjects:
        if item.outcome in BLOCKING_OUTCOMES:
            blockers.append(f"{item.subject}: {item.outcome} -- {item.reason}")
    for token, receipt in receipts.items():
        if resolved.subject(token) is None:
            blockers.append(f"{token}: a receipt exists for a subject the denominator does not declare")
            continue
        argv = receipt.get("argv")
        flags = {item for item in argv if isinstance(item, str)} if isinstance(argv, Sequence) else set()
        missing = sorted({"--full-corpus", "--frontier"} - flags)
        if missing:
            blockers.append(
                f"{token}: the recorded invocation did not declare {', '.join(missing)}, so the pass was not "
                "a complete run bound to the declared frontier"
            )
    semantics = _generator_semantics(receipts)
    for key, values in semantics.items():
        if len(values) > 1:
            blockers.append(f"the pass mixed {len(values)} generator revisions at {key}")

    return ProviderMatrix(
        subjects=subjects,
        denominator=resolved,
        baseline_digest=check.baseline_digest,
        frontier_declaration_digest=check.declaration_digest,
        code_revision=code_revision,
        generator_semantics=semantics,
        inference_configuration=inference_configuration,
        blockers=tuple(blockers),
    )


def load_receipts(payloads: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    """Index run receipts by subject, refusing a duplicate."""

    receipts: dict[str, Mapping[str, object]] = {}
    for payload in payloads:
        subject = payload.get("subject")
        if not isinstance(subject, str) or not subject:
            raise ValueError("every run receipt must name its subject")
        if subject in receipts:
            raise ValueError(f"two run receipts claim subject {subject}")
        receipts[subject] = payload
    return receipts


__all__ = [
    "BLOCKING_OUTCOMES",
    "RECONCILIATION_SCHEMA",
    "ProviderMatrix",
    "SubjectDenominatorCounts",
    "SubjectOutcome",
    "SubjectReconciliation",
    "load_receipts",
    "reconcile_provider_matrix",
]
