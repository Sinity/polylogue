"""``metric:<hash>`` -- content-addressed metric definitions (rxdo.9.1).

A :class:`MetricDefinition` canonicalizes construct, unit source, filters,
material_origin mask, aggregation, and exclusions into one hash exactly like
the query:<hash> canonicalizer (rxdo.2, riding :mod:`polylogue.insights.
measurement.canon`). Findings that cite a ``metric_ref`` are therefore
either directly comparable (identical hash) or visibly incomparable
(different hash) -- comparing "cost" computed under two different
denominator/exclusion rules can no longer silently produce a wrong number.
This is precisely the failure class behind the 7.69x Codex cost inflation
(input tokens included cached) and the 376.6B stale-token figure (see
``docs/design/analysis-rigor.md``, mechanism A).

This module is the SOLE identity/schema owner for ``metric:<hash>``
(rxdo.9.1 authoritative corrective scope, 2026-07-13). Any statistics
registry (9l5.7's ``MeasureSpec``) must construct/consume
:class:`MetricDefinition` here rather than mint a second identity --
:attr:`MetricDefinition.ref` is a pure function of content, so two
independent call sites describing the same metric always resolve to the
same ref.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from polylogue.insights.measurement.canon import content_ref

NullPolicy = Literal["suppress", "zero", "exclude", "separate-unknown"]
MeasurementAuthority = Literal["provider-reported", "catalog-estimated", "heuristic", "structural"]
Grain = Literal["physical", "logical"]
ProvenanceMixing = Literal["single-authority", "mixed-declared"]


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """A canonicalizable, content-addressed metric definition.

    Fields correspond to the rxdo.9.1 authoritative corrective contract:
    construct, formula/component refs, unit, grain, denominator/null
    policy, required enumeration/frame/authority, confounds, provenance
    mixing, and output schema.
    """

    construct: str
    unit: str
    unit_source: str
    aggregation: str
    grain: Grain = "logical"
    filters: frozenset[str] = frozenset()
    material_origin_mask: frozenset[str] = frozenset()
    exclusions: frozenset[str] = frozenset()
    denominator_expr: str | None = None
    null_policy: NullPolicy = "separate-unknown"
    required_enumeration: str = "exact"
    required_frame: str = ""
    measurement_authority: tuple[MeasurementAuthority, ...] = ()
    confounds: frozenset[str] = frozenset()
    provenance_mixing: ProvenanceMixing = "single-authority"
    formula_version: int = 1
    output_schema: str = ""
    # Component refs for derived metrics (e.g. ratio numerator/denominator
    # metric:<hash> refs, rxdo.9.2). Empty for a plain aggregate metric.
    component_refs: tuple[str, ...] = ()
    formula_kind: Literal["aggregate", "ratio"] = "aggregate"

    def canonical_payload(self) -> dict[str, object]:
        """The exact payload hashed to produce :attr:`ref`."""

        return {
            "construct": self.construct,
            "unit": self.unit,
            "unit_source": self.unit_source,
            "aggregation": self.aggregation,
            "grain": self.grain,
            "filters": self.filters,
            "material_origin_mask": self.material_origin_mask,
            "exclusions": self.exclusions,
            "denominator_expr": self.denominator_expr,
            "null_policy": self.null_policy,
            "required_enumeration": self.required_enumeration,
            "required_frame": self.required_frame,
            # measurement_authority is an order-independent envelope (see
            # is_compatible_with, which compares it as a set) -- sort before
            # hashing so two definitions built with the same authorities in
            # a different construction order resolve to the same ref.
            "measurement_authority": sorted(self.measurement_authority),
            "confounds": self.confounds,
            "provenance_mixing": self.provenance_mixing,
            "formula_version": self.formula_version,
            "output_schema": self.output_schema,
            "component_refs": list(self.component_refs),
            "formula_kind": self.formula_kind,
        }

    @property
    def ref(self) -> str:
        """The content-addressed ``metric:<hash>`` identity for this definition."""

        return content_ref("metric", self.canonical_payload())

    def is_compatible_with(self, other: MetricDefinition) -> bool:
        """Whether ``other`` shares this metric's grain/frame/authority envelope.

        Used to fail closed on cross-metric composition (ratios,
        comparisons) rather than silently comparing incompatible
        measurements.
        """

        if self.grain != other.grain:
            return False
        if self.required_frame and other.required_frame and self.required_frame != other.required_frame:
            return False
        if self.provenance_mixing == "single-authority" and other.provenance_mixing == "single-authority":
            self_authority = set(self.measurement_authority)
            other_authority = set(other.measurement_authority)
            if self_authority and other_authority and not self_authority & other_authority:
                return False
        return True


class DuplicateMetricIdentityError(ValueError):
    """A registry entry would create a second identity for an equivalent metric."""


@dataclass(slots=True)
class MetricRegistry:
    """In-process registry mapping ``metric:<hash>`` refs to their definitions.

    This is the completeness-audit surface for the rxdo.9.1 corrective AC
    ("creating an equivalent second MeasureSpec identity is impossible or
    rejected"): registration is idempotent by content hash (re-registering
    an identical definition is a no-op), and registering a *different*
    definition under a friendly *name* already bound to a different ref
    raises rather than silently shadowing.
    """

    _by_ref: dict[str, MetricDefinition] = field(default_factory=dict)
    _by_name: dict[str, str] = field(default_factory=dict)

    def register(self, definition: MetricDefinition, *, name: str | None = None) -> str:
        ref = definition.ref
        existing = self._by_ref.get(ref)
        if existing is not None and existing.canonical_payload() != definition.canonical_payload():
            # SHA-256 collision on differing canonical payloads should not
            # happen in practice; guard the invariant explicitly rather than
            # assume it. Compare canonical_payload(), not dataclass field
            # equality: two definitions can share a ref (identical canonical
            # payload) while differing in field *construction* order --
            # e.g. measurement_authority=("a", "b") vs ("b", "a") -- which
            # dataclass `!=` would flag as distinct even though they are the
            # same identity by definition.
            raise DuplicateMetricIdentityError(f"metric ref {ref!r} already bound to a different definition")
        self._by_ref[ref] = definition
        if name is not None:
            bound = self._by_name.get(name)
            if bound is not None and bound != ref:
                raise DuplicateMetricIdentityError(
                    f"friendly name {name!r} already bound to {bound!r}, cannot rebind to {ref!r}"
                )
            self._by_name[name] = ref
        return ref

    def get(self, ref: str) -> MetricDefinition | None:
        return self._by_ref.get(ref)

    def resolve(self, name: str) -> MetricDefinition | None:
        ref = self._by_name.get(name)
        return self._by_ref.get(ref) if ref is not None else None

    def __len__(self) -> int:
        return len(self._by_ref)


__all__ = [
    "DuplicateMetricIdentityError",
    "Grain",
    "MeasurementAuthority",
    "MetricDefinition",
    "MetricRegistry",
    "NullPolicy",
    "ProvenanceMixing",
]
