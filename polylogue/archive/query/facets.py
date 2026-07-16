"""Compute scoped vs global facet aggregates over session queries.

Facets are aggregate counts over a result set:

* **Scoped** facets are computed against the current query/filter set —
  they describe the distribution within the user's narrowed view.
* **Global** facets are computed against the archive as a whole — they
  describe the universe the scoped view was carved out of.

The substrate function :func:`compute_facets` is the single canonical
implementation. Surfaces (daemon HTTP, MCP, CLI, Python API) call into
the :class:`~polylogue.api.archive.Polylogue` facade method
``Polylogue.facets`` which delegates here. See #1269 (slice D of #873).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.archive.session.domain_models import SessionSummary


@dataclass(frozen=True)
class FacetBuckets:
    """Aggregate counts derived from a set of session summaries.

    Counts are keyed by the bucket label (origin family, tag name, etc.)
    and value the number of sessions in the input set that carry
    the bucket. ``total_sessions``/``total_messages`` describe the
    input set itself.

    The provider-role, material-origin, per-message, action, flag, and
    canonical repo families are populated by SQL-backed aggregators because
    they require scanning tables beyond the session summary surface.
    """

    origins: dict[str, int] = field(default_factory=dict)
    tags: dict[str, int] = field(default_factory=dict)
    repos: dict[str, int] = field(default_factory=dict)
    role_counts: dict[str, int] = field(default_factory=dict)
    material_origins: dict[str, int] = field(default_factory=dict)
    message_types: dict[str, int] = field(default_factory=dict)
    action_types: dict[str, int] = field(default_factory=dict)
    has_flags: dict[str, int] = field(default_factory=dict)
    omitted: dict[str, int] = field(default_factory=dict)
    total_sessions: int = 0
    total_messages: int = 0


@dataclass(frozen=True)
class FacetSet:
    """Scoped + global facet pair with optional IDF weighting.

    * ``scoped`` describes the sessions matching the active filter
      chain — empty buckets if the user has narrowed away every value.
    * ``global_`` describes the unfiltered archive.
    * ``scoped_to_query`` is ``True`` whenever the active filter chain
      narrows the result set away from the global view (i.e. it has any
      effective predicate).
    * ``idf`` carries inverse-document-frequency weights per facet
      value, computed against the *global* universe so that a bucket
      appearing in nearly every session is "noise" (low IDF) and a
      rare bucket is "signal" (high IDF). The map is keyed by facet
      family name (``origins``, ``tags``) and then by value.
    """

    scoped: FacetBuckets
    global_: FacetBuckets
    scoped_to_query: bool
    idf: dict[str, dict[str, float]] = field(default_factory=dict)


def compute_facets(
    summaries: Iterable[SessionSummary],
) -> FacetBuckets:
    """Roll a sequence of summaries into per-bucket counts.

    Pure function: no I/O, no async, no repository access. Used by both
    scoped and global computation paths.
    """

    origins: dict[str, int] = {}
    tags: dict[str, int] = {}
    total_sessions = 0
    total_messages = 0
    for s in summaries:
        total_sessions += 1
        total_messages += s.message_count or 0
        origin_key = s.origin.value
        origins[origin_key] = origins.get(origin_key, 0) + 1
        # ``tags`` are M2M user-tags; deduplicate within a single
        # session so a doubly-applied tag does not double-count.
        for tag in set(s.tags):
            tags[tag] = tags.get(tag, 0) + 1
    return FacetBuckets(
        origins=origins,
        tags=tags,
        total_sessions=total_sessions,
        total_messages=total_messages,
    )


def compute_idf(buckets: FacetBuckets) -> dict[str, dict[str, float]]:
    """Compute per-value inverse-document-frequency over a bucket set.

    ``idf(value) = log(N / df(value))`` using natural log, where ``N``
    is ``total_sessions`` and ``df`` is the value's count.
    Returns ``{}`` if the universe is empty.

    Higher IDF means the value partitions the archive more strongly
    (rare = signal); IDF near zero means the value is shared by nearly
    every session (common = noise).
    """

    total = buckets.total_sessions
    if total <= 0:
        return {}
    out: dict[str, dict[str, float]] = {}
    for family_name, family in (
        ("origins", buckets.origins),
        ("tags", buckets.tags),
        ("repos", buckets.repos),
        ("role_counts", buckets.role_counts),
        ("material_origins", buckets.material_origins),
        ("message_types", buckets.message_types),
        ("action_types", buckets.action_types),
        ("has_flags", buckets.has_flags),
    ):
        family_out: dict[str, float] = {}
        for value, count in family.items():
            if count <= 0:
                continue
            family_out[value] = math.log(total / count)
        if family_out:
            out[family_name] = family_out
    return out


__all__ = ["FacetBuckets", "FacetSet", "compute_facets", "compute_idf"]
