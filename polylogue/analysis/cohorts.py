"""Deterministic, archive-independent cohort and sample manifests.

Analytical packets need to name their population and reproduce their selected
sample without relying on source row order.  This module deliberately accepts
ordinary object-reference candidates rather than a delegation-specific row so
the same manifest discipline applies to any query unit.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import ClassVar

from pydantic import ConfigDict

from polylogue.storage.sqlite.connection_profile import scratch_connection_context

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

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

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

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    key: tuple[tuple[str, str], ...]
    population_count: int
    eligible_count: int
    selected_count: int


@dataclass(frozen=True)
class CohortManifest:
    """Byte-stable record of a cohort population and deterministic sample."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

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


def compile_cohort_manifest(
    spec: CohortSpec, candidates: Iterable[CohortCandidate], *, checkpoint: Callable[[], None] = lambda: None
) -> CohortManifest:
    """Spool the population and compile the original deterministic sample.

    The scratch relation owns population storage, uniqueness and seeded ranking.
    Only selected refs and the declared aggregate output remain in Python memory.
    """

    checkpoint()
    with (
        scratch_connection_context(prefix="polylogue-cohort-", filename="population.db") as scratch,
    ):
        scratch.execute("PRAGMA temp_store = FILE")
        scratch.execute("PRAGMA cache_size = -2048")
        scratch.execute(
            "CREATE TABLE population (ref BLOB PRIMARY KEY, stratum TEXT NOT NULL, rank TEXT NOT NULL, "
            "template TEXT, exclusion TEXT, canonical TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0)"
        )
        scratch.execute("CREATE INDEX selection ON population(stratum, exclusion, consumed, rank, ref)")
        cancellation: BaseException | None = None

        def progress() -> int:
            nonlocal cancellation
            try:
                checkpoint()
            except BaseException as exc:
                cancellation = exc
                return 1
            return 0

        scratch.set_progress_handler(progress, 1000)
        try:
            for candidate in candidates:
                checkpoint()
                if not candidate.object_ref:
                    raise ValueError("cohort candidate object_ref must not be empty")
                canonical = json.dumps(
                    {
                        "object_ref": candidate.object_ref,
                        "dimensions": dict(candidate.dimensions),
                        "template_key": candidate.template_key,
                        "exclusion_reason": candidate.exclusion_reason,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                try:
                    scratch.execute(
                        "INSERT INTO population(ref,stratum,rank,template,exclusion,canonical) VALUES(?,?,?,?,?,?)",
                        (
                            candidate.object_ref.encode("utf-8", "surrogatepass"),
                            json.dumps(_stratum_key(candidate, spec.strata)),
                            _rank(spec, candidate)[0] if candidate.exclusion_reason is None else "",
                            json.dumps(candidate.template_key) if candidate.template_key is not None else None,
                            json.dumps(candidate.exclusion_reason) if candidate.exclusion_reason is not None else None,
                            canonical,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError("cohort candidates must have unique object_ref values") from exc
            digest = sha256(b'{"population":[')
            separator = b""
            for (canonical,) in scratch.execute("SELECT canonical FROM population ORDER BY ref"):
                checkpoint()
                digest.update(separator)
                digest.update(canonical.encode("utf-8"))
                separator = b","
            digest.update(b'],"spec":')
            digest.update(json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode("utf-8"))
            digest.update(b"}")
            population_count, eligible_count = scratch.execute(
                "SELECT COUNT(*),COUNT(*) FILTER(WHERE exclusion IS NULL) FROM population"
            ).fetchone()
            group_counts = sorted(
                (
                    (tuple(tuple(pair) for pair in json.loads(key)), key, count, eligible)
                    for key, count, eligible in scratch.execute(
                        "SELECT stratum,COUNT(*),COUNT(*) FILTER(WHERE exclusion IS NULL) FROM population GROUP BY stratum"
                    )
                ),
                key=lambda item: item[0],
            )
            selected: list[str] = []
            selected_templates: Counter[str] = Counter()
            selected_by_stratum: Counter[tuple[tuple[str, str], ...]] = Counter()
            active_groups = [(key, encoded) for key, encoded, _, eligible in group_counts if eligible]
            while active_groups and len(selected) < spec.requested_size:
                next_active = []
                for key, encoded in active_groups:
                    while True:
                        checkpoint()
                        row = scratch.execute(
                            "SELECT ref,template FROM population WHERE stratum=? AND exclusion IS NULL "
                            "AND consumed=0 ORDER BY rank,ref LIMIT 1",
                            (encoded,),
                        ).fetchone()
                        if row is None:
                            break
                        ref, template_json = row
                        scratch.execute("UPDATE population SET consumed=1 WHERE ref=?", (ref,))
                        template = None if template_json is None else json.loads(template_json)
                        if (
                            template is not None
                            and spec.exact_template_cap is not None
                            and selected_templates[template] >= spec.exact_template_cap
                        ):
                            continue
                        selected.append(ref.decode("utf-8", "surrogatepass"))
                        selected_by_stratum[key] += 1
                        if template is not None:
                            selected_templates[template] += 1
                        next_active.append((key, encoded))
                        break
                    if len(selected) == spec.requested_size:
                        break
                active_groups = next_active
            exclusions = tuple(
                sorted(
                    (json.loads(reason), count)
                    for reason, count in scratch.execute(
                        "SELECT exclusion,COUNT(*) FROM population WHERE exclusion IS NOT NULL GROUP BY exclusion"
                    )
                )
            )
            template_counts: Counter[str] = Counter()
            for template, count in scratch.execute("SELECT template,COUNT(*) FROM population GROUP BY template"):
                checkpoint()
                template_counts[(json.loads(template) if template is not None else None) or _UNKNOWN] += count
            return CohortManifest(
                manifest_id=digest.hexdigest(),
                spec=spec,
                population_count=population_count,
                eligible_count=eligible_count,
                selected_refs=tuple(selected),
                excluded_counts=exclusions,
                stratum_counts=tuple(
                    CohortStratumCount(key, count, eligible, selected_by_stratum[key])
                    for key, _, count, eligible in group_counts
                ),
                template_counts=tuple(sorted(template_counts.items())),
                shortfall=max(spec.requested_size - len(selected), 0),
            )
        except sqlite3.Error:
            if cancellation is not None:
                raise cancellation from None
            raise


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
