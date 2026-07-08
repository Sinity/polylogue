"""Rigor audit reports — per-product evidence/inference/fallback rollups (#1275).

The audit walks each insight product covered by
:mod:`polylogue.insights.rigor`, fetches a bounded sample of rows from
the live archive, and rolls up:

- total rows inspected
- rows that carry a non-empty evidence payload
- rows that carry a non-empty inference payload
- rows flagged as fallback by any of the contract's ``fallback_markers``
- rows whose materialization versions are below the current targets
  (``stale_version_count``)
- a 4-bucket confidence distribution (``low``/``mid``/``high``/``unknown``)

The runner is deliberately read-only and bounded by ``sample_limit`` so
that ``polylogue insights audit`` answers in seconds against a real
archive. Callers can widen the sample for closer audits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

from pydantic import Field

from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.insights.rigor import (
    RigorContract,
    get_rigor_contract,
    resolve_payload,
    rigor_exemption_reason,
)

CoverageStatus = Literal["covered", "uncovered", "exempt"]

DEFAULT_AUDIT_SAMPLE_LIMIT = 500


class InsightRigorAuditQuery(ArchiveInsightModel):
    """Audit scope. Empty ``insights`` means every contract row."""

    insights: tuple[str, ...] = ()
    sample_limit: int = DEFAULT_AUDIT_SAMPLE_LIMIT


class ConfidenceDistribution(ArchiveInsightModel):
    """4-bucket confidence histogram."""

    low: int = 0
    mid: int = 0
    high: int = 0
    unknown: int = 0


class InsightRigorAuditEntry(ArchiveInsightModel):
    """Per-product rigor profile for one insight."""

    insight_name: str
    display_name: str
    coverage_status: CoverageStatus = "covered"
    sample_size: int = 0
    evidence_count: int = 0
    inference_count: int = 0
    fallback_count: int = 0
    stale_version_count: int = 0
    has_evidence_payload: bool = False
    has_inference_payload: bool = False
    has_fallback_markers: bool = False
    has_confidence_field: bool = False
    confidence_distribution: ConfidenceDistribution = Field(default_factory=ConfidenceDistribution)
    version_targets: dict[str, int] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()
    error: str | None = None


class InsightRigorAuditReport(ArchiveInsightModel):
    """Whole-archive rigor profile across every audited product."""

    sample_limit: int
    entries: tuple[InsightRigorAuditEntry, ...] = ()


def _bucket_confidence(value: float) -> str:
    if value < 0.34:
        return "low"
    if value < 0.67:
        return "mid"
    return "high"


def _payload_is_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str | bytes):
        return len(value) > 0
    if isinstance(value, dict | list | tuple | set | frozenset):
        return len(value) > 0
    # Pydantic models / dataclasses / numbers / bools: presence is by identity.
    return True


def _version_is_stale(row: object, version_name: str, current_version: int) -> bool:
    """A row is stale if it carries the named version and it's below current."""

    # Versions can live either directly on the row or under provenance /
    # inference_provenance / enrichment_provenance sub-objects, depending on
    # the insight kind. Search in declared order.
    for path in (
        (version_name,),
        ("provenance", version_name),
        ("inference_provenance", version_name),
        ("enrichment_provenance", version_name),
    ):
        observed = resolve_payload(row, path)
        if observed is None:
            continue
        try:
            return int(cast(int, observed)) < int(current_version)
        except (TypeError, ValueError):
            return False
    return False


def _audit_one(rows: Sequence[object], contract: RigorContract) -> InsightRigorAuditEntry:
    distribution = ConfidenceDistribution()
    evidence_count = 0
    inference_count = 0
    fallback_count = 0
    stale_version_count = 0
    for row in rows:
        if contract.evidence_payload and _payload_is_present(resolve_payload(row, contract.evidence_payload)):
            evidence_count += 1
        if contract.inference_payload and _payload_is_present(resolve_payload(row, contract.inference_payload)):
            inference_count += 1
        if contract.fallback_markers:
            for marker in contract.fallback_markers:
                if bool(resolve_payload(row, marker)):
                    fallback_count += 1
                    break
        row_is_stale = False
        for version in contract.version_fields:
            if _version_is_stale(row, version.name, version.current_version):
                row_is_stale = True
                break
        if row_is_stale:
            stale_version_count += 1
        if contract.confidence_field:
            observed = resolve_payload(row, contract.confidence_field)
            if observed is None:
                distribution = distribution.model_copy(update={"unknown": distribution.unknown + 1})
                continue
            try:
                bucket = _bucket_confidence(float(cast(float, observed)))
            except (TypeError, ValueError):
                distribution = distribution.model_copy(update={"unknown": distribution.unknown + 1})
                continue
            distribution = distribution.model_copy(update={bucket: getattr(distribution, bucket) + 1})
    return InsightRigorAuditEntry(
        insight_name=contract.insight_name,
        display_name=contract.display_name,
        sample_size=len(rows),
        evidence_count=evidence_count,
        inference_count=inference_count,
        fallback_count=fallback_count,
        stale_version_count=stale_version_count,
        has_evidence_payload=bool(contract.evidence_payload),
        has_inference_payload=bool(contract.inference_payload),
        has_fallback_markers=bool(contract.fallback_markers),
        has_confidence_field=bool(contract.confidence_field),
        confidence_distribution=distribution,
        version_targets={version.name: version.current_version for version in contract.version_fields},
        notes=(contract.notes,) if contract.notes else (),
    )


async def build_insight_rigor_audit_report(
    operations: object,
    query: InsightRigorAuditQuery | None = None,
) -> InsightRigorAuditReport:
    """Audit the rigor profile of every registered insight product.

    Iterates :data:`polylogue.insights.registry.INSIGHT_REGISTRY` -- every
    product Polylogue ships, not just the ones with a declared rigor
    contract (9e5.28) -- so a product losing its contract shows up as
    ``coverage_status="uncovered"`` instead of silently vanishing from the
    report. A product with a contract is fully audited (a bounded sample
    fetched and classified as before); a product with neither a contract
    nor a listed exemption gets a zero-sample ``"uncovered"`` stub; a
    product in :data:`polylogue.insights.rigor.RIGOR_EXEMPT` gets a
    zero-sample ``"exempt"`` stub carrying its justification in ``notes``.
    """

    from polylogue.insights.registry import INSIGHT_REGISTRY

    request = query or InsightRigorAuditQuery()
    targeted = set(request.insights) if request.insights else None
    entries: list[InsightRigorAuditEntry] = []
    for insight_name, insight_type in INSIGHT_REGISTRY.items():
        if targeted is not None and insight_name not in targeted:
            continue
        contract = get_rigor_contract(insight_name)
        if contract is not None:
            rows: list[object] = []
            error: str | None = None
            try:
                rows = await _fetch_rows(operations, insight_name, request.sample_limit)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            entry = _audit_one(rows, contract)
            if error is not None:
                entry = entry.model_copy(update={"error": error})
            entries.append(entry)
            continue
        exemption = rigor_exemption_reason(insight_name)
        entries.append(
            InsightRigorAuditEntry(
                insight_name=insight_name,
                display_name=insight_type.display_name,
                coverage_status="exempt" if exemption is not None else "uncovered",
                notes=(exemption,) if exemption is not None else (),
            )
        )
    return InsightRigorAuditReport(sample_limit=request.sample_limit, entries=tuple(entries))


async def _fetch_rows(
    operations: object,
    insight_name: str,
    sample_limit: int,
) -> list[object]:
    """Dispatch via the insights registry to operations.

    Imported lazily so the audit module does not pull the registry at
    module import time (the registry already imports
    :mod:`polylogue.insights.archive`).
    """

    from polylogue.insights.registry import (
        fetch_insights_async,
        get_insight_type,
    )

    insight_type = get_insight_type(insight_name)
    rows = await fetch_insights_async(insight_type, operations, limit=sample_limit)
    return list(rows)


__all__ = [
    "ConfidenceDistribution",
    "DEFAULT_AUDIT_SAMPLE_LIMIT",
    "InsightRigorAuditEntry",
    "InsightRigorAuditQuery",
    "InsightRigorAuditReport",
    "build_insight_rigor_audit_report",
]
