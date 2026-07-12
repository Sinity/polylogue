"""Deterministic, archive-independent cohort and sample manifests.

Analytical packets need to name their population and reproduce their selected
sample without relying on source row order.  This module deliberately accepts
ordinary object-reference candidates rather than a delegation-specific row so
the same manifest discipline applies to any query unit.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256

_UNKNOWN = "unknown"


@dataclass(frozen=True)
class CohortCandidate:
    """One query result eligible for a deterministic cohort.

    ``object_ref`` is the stable selected identity.  ``dimensions`` supplies
    optional stratum values and ``template_key`` enables exact-template
    sensitivity caps without making the primitive aware of prompt semantics.
    """

    object_ref: str
    dimensions: Mapping[str, str | None] = field(default_factory=dict)
    template_key: str | None = None
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class CohortSpec:
    """Inputs that define a reproducible population and sample selection."""

    population_query: str
    archive_cursor: str
    seed: str
    requested_size: int
    strata: tuple[str, ...] = ()
    exact_template_cap: int | None = None

    def __post_init__(self) -> None:
        if self.requested_size < 0:
            raise ValueError("requested_size must be non-negative")
        if self.exact_template_cap is not None and self.exact_template_cap < 1:
            raise ValueError("exact_template_cap must be at least one when provided")


@dataclass(frozen=True)
class CohortStratumCount:
    """Population and selected counts for one declared stratum."""

    key: tuple[tuple[str, str], ...]
    population_count: int
    eligible_count: int
    selected_count: int


@dataclass(frozen=True)
class CohortManifest:
    """Byte-stable record of a cohort population and deterministic sample."""

    manifest_id: str
    spec: CohortSpec
    population_count: int
    eligible_count: int
    selected_refs: tuple[str, ...]
    excluded_counts: tuple[tuple[str, int], ...]
    stratum_counts: tuple[CohortStratumCount, ...]
    template_counts: tuple[tuple[str, int], ...]
    shortfall: int

    def to_payload(self) -> dict[str, object]:
        """Return the canonical, JSON-serializable manifest representation."""

        return {
            "manifest_id": self.manifest_id,
            "spec": asdict(self.spec),
            "population_count": self.population_count,
            "eligible_count": self.eligible_count,
            "selected_refs": list(self.selected_refs),
            "excluded_counts": dict(self.excluded_counts),
            "stratum_counts": [
                {
                    "key": dict(count.key),
                    "population_count": count.population_count,
                    "eligible_count": count.eligible_count,
                    "selected_count": count.selected_count,
                }
                for count in self.stratum_counts
            ],
            "template_counts": dict(self.template_counts),
            "shortfall": self.shortfall,
        }

    def to_json(self) -> str:
        """Serialize the manifest with stable keys and separators."""

        return json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CohortDrift:
    """Explicit difference between two independently compiled manifests."""

    changed: bool
    added_refs: tuple[str, ...]
    removed_refs: tuple[str, ...]
    cursor_changed: bool


def _stratum_key(candidate: CohortCandidate, fields: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple((field, candidate.dimensions.get(field) or _UNKNOWN) for field in fields)


def _rank(spec: CohortSpec, candidate: CohortCandidate) -> tuple[str, str]:
    material = "\0".join((spec.seed, spec.archive_cursor, candidate.object_ref))
    return sha256(material.encode("utf-8")).hexdigest(), candidate.object_ref


def _manifest_id(spec: CohortSpec, candidates: Sequence[CohortCandidate]) -> str:
    payload = {
        "spec": asdict(spec),
        "population": [
            {
                "object_ref": candidate.object_ref,
                "dimensions": dict(sorted(candidate.dimensions.items())),
                "template_key": candidate.template_key,
                "exclusion_reason": candidate.exclusion_reason,
            }
            for candidate in sorted(candidates, key=lambda item: item.object_ref)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def compile_cohort_manifest(spec: CohortSpec, candidates: Sequence[CohortCandidate]) -> CohortManifest:
    """Compile a deterministic stratified sample manifest.

    Each stratum contributes in round-robin order after a stable seeded rank.
    This prevents a large or early stratum from consuming the entire requested
    sample while preserving reproducibility.  Template caps are applied across
    the complete sample; excluded candidates remain counted but never selected.
    """

    candidate_by_ref = {candidate.object_ref: candidate for candidate in candidates}
    if len(candidate_by_ref) != len(candidates):
        raise ValueError("cohort candidates must have unique object_ref values")
    if any(not candidate.object_ref for candidate in candidates):
        raise ValueError("cohort candidate object_ref must not be empty")

    exclusions = Counter(
        candidate.exclusion_reason for candidate in candidates if candidate.exclusion_reason is not None
    )
    included = [candidate for candidate in candidates if candidate.exclusion_reason is None]
    groups: dict[tuple[tuple[str, str], ...], list[CohortCandidate]] = defaultdict(list)
    population_groups: Counter[tuple[tuple[str, str], ...]] = Counter()
    for candidate in candidates:
        population_groups[_stratum_key(candidate, spec.strata)] += 1
    for candidate in included:
        groups[_stratum_key(candidate, spec.strata)].append(candidate)
    for group in groups.values():
        group.sort(key=lambda candidate: _rank(spec, candidate))

    selected: list[CohortCandidate] = []
    selected_templates: Counter[str] = Counter()
    group_positions = dict.fromkeys(groups, 0)
    active_groups = sorted(groups)
    while active_groups and len(selected) < spec.requested_size:
        next_active: list[tuple[tuple[str, str], ...]] = []
        for key in active_groups:
            group = groups[key]
            position = group_positions[key]
            while position < len(group):
                candidate = group[position]
                position += 1
                template = candidate.template_key
                if (
                    template is not None
                    and spec.exact_template_cap is not None
                    and selected_templates[template] >= spec.exact_template_cap
                ):
                    continue
                selected.append(candidate)
                if template is not None:
                    selected_templates[template] += 1
                break
            group_positions[key] = position
            if position < len(group):
                next_active.append(key)
            if len(selected) == spec.requested_size:
                break
        active_groups = next_active

    selected_by_stratum = Counter(_stratum_key(candidate, spec.strata) for candidate in selected)
    stratum_counts = tuple(
        CohortStratumCount(
            key=key,
            population_count=population_groups[key],
            eligible_count=len(groups.get(key, ())),
            selected_count=selected_by_stratum[key],
        )
        for key in sorted(population_groups)
    )
    template_counts = Counter(candidate.template_key or _UNKNOWN for candidate in candidates)
    return CohortManifest(
        manifest_id=_manifest_id(spec, candidates),
        spec=spec,
        population_count=len(candidates),
        eligible_count=len(included),
        selected_refs=tuple(candidate.object_ref for candidate in selected),
        excluded_counts=tuple(sorted(exclusions.items())),
        stratum_counts=stratum_counts,
        template_counts=tuple(sorted(template_counts.items())),
        shortfall=max(spec.requested_size - len(selected), 0),
    )


def compare_cohort_manifests(previous: CohortManifest, current: CohortManifest) -> CohortDrift:
    """Describe population/cursor drift without silently reusing a manifest."""

    previous_refs = set(previous.selected_refs)
    current_refs = set(current.selected_refs)
    return CohortDrift(
        changed=previous.manifest_id != current.manifest_id,
        added_refs=tuple(sorted(current_refs - previous_refs)),
        removed_refs=tuple(sorted(previous_refs - current_refs)),
        cursor_changed=previous.spec.archive_cursor != current.spec.archive_cursor,
    )


__all__ = [
    "CohortCandidate",
    "CohortDrift",
    "CohortManifest",
    "CohortSpec",
    "CohortStratumCount",
    "compare_cohort_manifests",
    "compile_cohort_manifest",
]
