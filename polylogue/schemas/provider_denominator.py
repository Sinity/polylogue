"""Derive the provider denominator a schema run must reconcile against.

A coverage claim is only as good as its denominator, and the recurring defect
in this area is a denominator derived from the thing it measures: a routing
check that iterates its own routed set, a coverage report whose subject list is
read back out of its own output. Such a denominator cannot fail, because a
subject the run never reached is also a subject the denominator never knew
about.

So this module derives the denominator from two declarations that exist for
different reasons and are maintained for different consumers:

* :data:`polylogue.core.schema_subjects.SCHEMA_SUBJECTS` -- the canonical
  schema-subject kernel, which names every structural wire subject, its
  package disposition and its public origins;
* :func:`polylogue.sources.origin_specs.origin_specs` -- the executable origin
  registry that ingest detection and parsing run on, which exists whether or
  not a single schema package was ever generated.

Neither is written by schema generation, and the two must agree: every origin a
subject declares has to resolve to a registered ``OriginSpec``, and every
executable origin has to be claimed by exactly one subject. A disagreement is a
finding, not a quietly smaller denominator.

**What this module must never read** is equally load-bearing: the committed
package tree, the schema registry, the declared source frontier, and any run
receipt are all products of the run being measured.
:func:`derive_provider_denominator` therefore imports none of them, and
``tests/unit/schemas/test_provider_denominator.py`` asserts that the module's
transitive import closure stays clear of them -- so "it cannot see itself" is a
checkable property rather than a comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONDocument
from polylogue.core.schema_subjects import SCHEMA_SUBJECTS, SchemaSubjectSpec
from polylogue.sources.origin_specs import origin_specs

#: How a subject enters the denominator.
#:
#: ``generation_required``
#:     The subject has an executable acquisition route, so the only honest
#:     outcomes are a non-vacuous generation or a complete, zero-eligible
#:     denominator proven by the run.
#: ``declared_non_applicable``
#:     The subject is declared outside the inference denominator: this
#:     repository authors its wire format, so there is nothing to infer and no
#:     material to count.
#: ``no_executable_route``
#:     Every origin the subject declares is reserved or compatibility-only, so
#:     no acquisition route can produce eligible material at all.
SubjectDisposition: TypeAlias = Literal[
    "generation_required",
    "declared_non_applicable",
    "no_executable_route",
]

_EXECUTABLE: Final[str] = "executable"


@dataclass(frozen=True, slots=True, order=True)
class DenominatorSubject:
    """One declared schema subject and why it is in the denominator."""

    subject: str
    disposition: SubjectDisposition
    origins: tuple[str, ...]
    executable_origins: tuple[str, ...]
    requires_package: bool
    reason: str | None

    def to_payload(self) -> JSONDocument:
        return {
            "subject": self.subject,
            "disposition": self.disposition,
            "origins": list(self.origins),
            "executable_origins": list(self.executable_origins),
            "requires_package": self.requires_package,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ProviderDenominator:
    """The complete set of subjects one schema run has to account for."""

    subjects: tuple[DenominatorSubject, ...]
    findings: tuple[str, ...]

    @property
    def declaration_digest(self) -> str:
        """Digest over both declarations, for binding into a run receipt."""

        return hash_payload(
            {
                "subjects": [subject.to_payload() for subject in self.subjects],
                "findings": list(self.findings),
            }
        )

    @property
    def generation_required(self) -> tuple[str, ...]:
        return tuple(item.subject for item in self.subjects if item.disposition == "generation_required")

    def subject(self, token: str) -> DenominatorSubject | None:
        normalized = token.strip().lower().replace("_", "-")
        return next((item for item in self.subjects if item.subject == normalized), None)

    def to_payload(self) -> JSONDocument:
        return {
            "declaration_digest": self.declaration_digest,
            "subjects": [subject.to_payload() for subject in self.subjects],
            "findings": list(self.findings),
        }


def _disposition(spec: SchemaSubjectSpec, executable: tuple[str, ...]) -> tuple[SubjectDisposition, str | None]:
    if spec.inference_excluded_reason is not None:
        return "declared_non_applicable", spec.inference_excluded_reason
    if not executable:
        return "no_executable_route", spec.package_not_required_reason or (
            "every declared origin is reserved or compatibility-only, so no acquisition route admits material"
        )
    return "generation_required", None


def derive_provider_denominator() -> ProviderDenominator:
    """Cross-derive the provider denominator from the two live declarations.

    The result is computed at call time from imported declarations only. It
    never consults the committed package tree, the schema registry, the source
    frontier or a run receipt, so a subject whose package was deleted, whose
    frontier root was dropped, or whose run never started is still counted.
    """

    lifecycle_by_origin = {str(spec.origin): spec.lifecycle for spec in origin_specs()}
    findings: list[str] = []
    claimed: dict[str, list[str]] = {}
    subjects: list[DenominatorSubject] = []

    for spec in SCHEMA_SUBJECTS:
        executable: list[str] = []
        for origin in spec.origins:
            lifecycle = lifecycle_by_origin.get(origin)
            if lifecycle is None:
                findings.append(f"subject {spec.token} declares origin {origin}, which no OriginSpec registers")
                continue
            claimed.setdefault(origin, []).append(spec.token)
            if lifecycle == _EXECUTABLE:
                executable.append(origin)
        disposition, reason = _disposition(spec, tuple(executable))
        subjects.append(
            DenominatorSubject(
                subject=spec.token,
                disposition=disposition,
                origins=tuple(spec.origins),
                executable_origins=tuple(executable),
                requires_package=spec.requires_package,
                reason=reason,
            )
        )

    for origin, lifecycle in sorted(lifecycle_by_origin.items()):
        if lifecycle != _EXECUTABLE:
            continue
        owners = claimed.get(origin, ())
        if not owners:
            findings.append(f"executable origin {origin} is claimed by no declared schema subject")
        elif len(owners) > 1:
            findings.append(f"executable origin {origin} is claimed by {len(owners)} subjects: {', '.join(owners)}")

    return ProviderDenominator(subjects=tuple(sorted(subjects)), findings=tuple(sorted(findings)))


__all__ = [
    "DenominatorSubject",
    "ProviderDenominator",
    "SubjectDisposition",
    "derive_provider_denominator",
]
