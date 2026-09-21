"""Per-product rigor contract matrix + audit runner tests (#1275)."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from polylogue.analysis.archive import (
    CostRollupInsight,
    SessionProfileInsight,
    SessionTagRollupInsight,
)
from polylogue.analysis.archive_models import (
    ArchiveEnrichmentProvenance,
    ArchiveInferenceProvenance,
    ArchiveInsightProvenance,
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
)
from polylogue.analysis.audit import (
    InsightRigorAuditQuery,
    _audit_one,
    build_insight_rigor_audit_report,
)
from polylogue.analysis.confidence import ConfidenceBand
from polylogue.analysis.fallback import FallbackReason
from polylogue.analysis.registry import INSIGHT_REGISTRY
from polylogue.analysis.rigor import (
    RigorFieldContract,
    get_rigor_contract,
    invalid_nullable_field_contracts,
    list_rigor_contracts,
    missing_numeric_field_coverage,
    missing_numeric_item_models,
    resolve_payload,
    rigor_contract_names,
)
from polylogue.archive.semantic.pricing import CostBasisPayload
from polylogue.storage.runtime.store_constants import (
    SESSION_ENRICHMENT_VERSION,
    SESSION_INFERENCE_VERSION,
    SESSION_INSIGHT_MATERIALIZER_VERSION,
)


def _provenance() -> ArchiveInsightProvenance:
    return ArchiveInsightProvenance(
        materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
        materialized_at="2026-05-18T00:00:00+00:00",
    )


def _inference_provenance() -> ArchiveInferenceProvenance:
    return ArchiveInferenceProvenance(
        materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
        materialized_at="2026-05-18T00:00:00+00:00",
        inference_version=SESSION_INFERENCE_VERSION,
        inference_family="default",
    )


def _enrichment_provenance() -> ArchiveEnrichmentProvenance:
    return ArchiveEnrichmentProvenance(
        materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
        materialized_at="2026-05-18T00:00:00+00:00",
        enrichment_version=SESSION_ENRICHMENT_VERSION,
        enrichment_family="default",
    )


def _profile(session_id: str = "c1") -> SessionProfileInsight:
    return SessionProfileInsight(
        session_id=session_id,
        logical_session_id=session_id,
        origin="claude-code",
        provenance=_provenance(),
        evidence=SessionEvidencePayload(message_count=10, word_count=200),
        inference_provenance=_inference_provenance(),
        inference=SessionInferencePayload(support_level=ConfidenceBand.STRONG),
        enrichment_provenance=_enrichment_provenance(),
        enrichment=SessionEnrichmentPayload(confidence=0.8, support_level=ConfidenceBand.STRONG),
    )


def _degraded_profile(*, fallback: bool, confidence: float, session_id: str = "c1") -> SessionProfileInsight:
    """A profile row whose enrichment payload carries the audit's markers.

    ``session_profiles`` is the contract that declares
    ``fallback_markers=(("enrichment", "fallback_reasons"),)`` and
    ``confidence_field=("enrichment", "confidence")``; the audit runner reads
    exactly those paths.
    """
    return _profile(session_id).model_copy(
        update={
            "enrichment": SessionEnrichmentPayload(
                confidence=confidence,
                support_level=ConfidenceBand.STRONG,
                fallback_reasons=(FallbackReason.NO_USER_TURNS,) if fallback else (),
            )
        }
    )


def _tag_rollup() -> SessionTagRollupInsight:
    return SessionTagRollupInsight(
        tag="design",
        session_count=3,
        explicit_count=2,
        auto_count=1,
        origin_breakdown={"claude-code": 3},
        repo_breakdown={"polylogue": 3},
        provenance=_provenance(),
    )


# --- Rigor matrix coverage ---


def test_rigor_matrix_covers_all_session_products() -> None:
    """Every product the issue calls out must have a contract row."""

    required = {
        "session_profiles",
        "threads",
        "session_tag_rollups",
    }
    actual = set(rigor_contract_names())
    assert required.issubset(actual), f"missing contracts: {required - actual}"


def test_rigor_matrix_or_exemption_covers_every_registered_insight() -> None:
    """No registered insight product may fall through both the matrix and the
    exemption list (9e5.28) -- every product is either contracted or an
    explicitly justified exemption."""

    from polylogue.analysis.rigor import RIGOR_EXEMPT

    registry_names = set(INSIGHT_REGISTRY.keys())
    contracted = set(rigor_contract_names())
    exempt = set(RIGOR_EXEMPT.keys())
    uncovered = registry_names - contracted - exempt
    assert not uncovered, f"registered insights with neither a contract nor an exemption: {uncovered}"
    assert contracted.isdisjoint(exempt), f"insights in both the matrix and the exemption list: {contracted & exempt}"


def test_rigor_matrix_entries_reference_registry_names() -> None:
    """Every rigor contract name must map to a registered insight type."""

    registry_names = set(INSIGHT_REGISTRY.keys())
    for contract in list_rigor_contracts():
        assert contract.insight_name in registry_names, contract.insight_name


def test_rigor_contract_lookup_round_trips() -> None:
    contract = get_rigor_contract("session_profiles")
    assert contract is not None
    assert contract.fallback_markers == (("enrichment", "fallback_reasons"),)
    assert contract.confidence_field == ("enrichment", "confidence")
    assert get_rigor_contract("not-a-real-insight") is None


def test_cost_rollup_confidence_declares_priced_session_denominator() -> None:
    """The public confidence value cannot be published without priced rows."""

    contract = get_rigor_contract("cost_rollups")
    assert contract is not None
    assert contract.field_contracts == (
        RigorFieldContract(
            field_path=("confidence",),
            provenance_class="derived",
            denominator_field=("priced_session_count",),
            evidence_tier="cost-pricing-rollup",
        ),
    )
    assert missing_numeric_field_coverage() == ()


def test_numeric_field_policy_rejects_unjustified_public_field() -> None:
    contract = get_rigor_contract("cost_rollups")
    assert contract is not None
    missing_confidence = contract.model_copy(
        update={
            "field_contracts": (),
            "field_exemptions": tuple(
                exemption for exemption in contract.field_exemptions if exemption.field_path != ("confidence",)
            ),
        }
    )
    contracts = tuple(
        missing_confidence if item.insight_name == "cost_rollups" else item for item in list_rigor_contracts()
    )

    assert missing_numeric_field_coverage(contracts) == (("cost_rollups", ("confidence",)),)


def test_numeric_field_policy_discovers_new_registered_nested_numeric_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """The policy reads the registry's production item model, not an allowlist.

    Removing the recursive model walk or the descriptor's ``item_model``
    wiring makes a new public metric bypass its rigor contract.
    """

    class ExtendedCostBasisPayload(CostBasisPayload):
        unclassified_metric: int = 0

    class ExtendedCostRollupInsight(CostRollupInsight):
        basis: ExtendedCostBasisPayload = ExtendedCostBasisPayload()

    original = INSIGHT_REGISTRY["cost_rollups"]
    monkeypatch.setitem(INSIGHT_REGISTRY, "cost_rollups", replace(original, item_model=ExtendedCostRollupInsight))

    assert missing_numeric_field_coverage() == (("cost_rollups", ("basis", "unclassified_metric")),)


def test_nullable_contract_policy_rejects_non_nullable_registered_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """A null-over-empty contract must target a production model that permits null."""

    class NonNullableCostRollupInsight(CostRollupInsight):
        confidence: float = 0.0

    original = INSIGHT_REGISTRY["cost_rollups"]
    monkeypatch.setitem(INSIGHT_REGISTRY, "cost_rollups", replace(original, item_model=NonNullableCostRollupInsight))

    assert invalid_nullable_field_contracts() == (("cost_rollups", ("confidence",)),)


def test_nullable_contract_policy_rejects_contract_that_disables_null_refusal() -> None:
    """A field contract cannot opt out of its promised null-over-empty behavior."""

    contract = get_rigor_contract("cost_rollups")
    assert contract is not None
    disabled_refusal = contract.model_copy(
        update={
            "field_contracts": (contract.field_contracts[0].model_copy(update={"nullable_when_ungrounded": False}),)
        }
    )
    contracts = tuple(
        disabled_refusal if item.insight_name == "cost_rollups" else item for item in list_rigor_contracts()
    )

    assert invalid_nullable_field_contracts(contracts) == (("cost_rollups", ("confidence",)),)


def test_every_registered_insight_declares_its_item_model() -> None:
    """Registry descriptors expose production response models to the policy."""

    assert missing_numeric_item_models() == ()


def test_resolve_payload_walks_attributes_and_dicts() -> None:
    row = _degraded_profile(fallback=True, confidence=0.42)
    assert resolve_payload(row, ("enrichment", "fallback_reasons")) == (FallbackReason.NO_USER_TURNS,)
    assert resolve_payload(row, ("enrichment", "confidence")) == 0.42
    assert resolve_payload(row, ("enrichment", "missing")) is None
    # Dict pathway:
    assert resolve_payload({"a": {"b": 1}}, ("a", "b")) == 1
    assert resolve_payload(None, ("a",)) is None


# --- Audit-runner classification ---


def test_audit_one_classifies_evidence_inference_and_fallback() -> None:
    contract = get_rigor_contract("session_profiles")
    assert contract is not None
    rows = [
        _degraded_profile(fallback=False, confidence=0.9),
        _degraded_profile(fallback=False, confidence=0.5),
        _degraded_profile(fallback=True, confidence=0.1),
    ]
    entry = _audit_one(rows, contract)
    assert entry.sample_size == 3
    assert entry.evidence_count == 3
    assert entry.inference_count == 3
    assert entry.fallback_count == 1
    assert entry.has_fallback_markers is True
    assert entry.confidence_distribution.low == 1
    assert entry.confidence_distribution.mid == 1
    assert entry.confidence_distribution.high == 1


def test_audit_one_handles_empty_sample() -> None:
    contract = get_rigor_contract("session_profiles")
    assert contract is not None
    entry = _audit_one([], contract)
    assert entry.sample_size == 0
    assert entry.evidence_count == 0
    assert entry.inference_count == 0
    assert entry.fallback_count == 0


def test_audit_one_detects_stale_version_rows() -> None:
    contract = get_rigor_contract("session_profiles")
    assert contract is not None
    fresh = _degraded_profile(fallback=False, confidence=0.9)
    stale = fresh.model_copy(
        update={
            "inference_provenance": ArchiveInferenceProvenance(
                materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
                materialized_at="2026-05-18T00:00:00+00:00",
                inference_version=max(0, SESSION_INFERENCE_VERSION - 1),
                inference_family="default",
            )
        }
    )
    entry = _audit_one([fresh, stale], contract)
    # Pick the count based on whether SESSION_INFERENCE_VERSION is bumpable.
    # When version=0, stale_count cannot be >0; otherwise the stale row counts.
    expected = 1 if SESSION_INFERENCE_VERSION > 0 else 0
    assert entry.stale_version_count == expected


def test_audit_one_profile_records_folded_enrichment_confidence() -> None:
    contract = get_rigor_contract("session_profiles")
    assert contract is not None
    entry = _audit_one(
        [
            _profile("c1"),
            _profile("c2").model_copy(
                update={
                    "enrichment": SessionEnrichmentPayload(
                        confidence=0.2,
                        support_level=ConfidenceBand.WEAK,
                    )
                }
            ),
        ],
        contract,
    )
    assert entry.confidence_distribution.high == 1
    assert entry.confidence_distribution.low == 1


# --- End-to-end report through the dispatch shim ---


class _FakeOperations:
    """Minimal duck-typed operations for the audit runner.

    The runner calls ``fetch_insights_async(insight_type, operations, limit=N)``,
    which calls ``operations.<operations_method_name>(query)``. We map each
    method to a synthetic row list.
    """

    def __init__(
        self,
        profiles: list[object],
        tags: list[object],
    ) -> None:
        self._payload = {
            "list_session_profile_insights": profiles,
            "list_session_tag_rollup_insights": tags,
            "list_thread_insights": [],
        }

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        rows = self._payload.get(name)
        if rows is None:
            raise AttributeError(name)

        async def _call(_query: object) -> list[object]:
            return list(rows)

        return _call


def test_build_insight_rigor_audit_report_aggregates_across_products() -> None:
    operations = _FakeOperations(
        profiles=[
            _degraded_profile(fallback=False, confidence=0.9, session_id="c1"),
            _degraded_profile(fallback=True, confidence=0.2, session_id="c2"),
        ],
        tags=[_tag_rollup()],
    )
    report = asyncio.run(build_insight_rigor_audit_report(operations, InsightRigorAuditQuery()))
    by_name = {entry.insight_name: entry for entry in report.entries}
    profiles = by_name["session_profiles"]
    assert profiles.sample_size == 2
    assert profiles.evidence_count == 2
    assert profiles.inference_count == 2
    assert profiles.fallback_count == 1
    assert profiles.has_fallback_markers is True
    tag = by_name["session_tag_rollups"]
    assert tag.sample_size == 1
    assert tag.has_evidence_payload is False  # tag rollups are aggregate
    assert report.sample_limit == InsightRigorAuditQuery().sample_limit


def test_audit_runner_respects_insight_filter() -> None:
    operations = _FakeOperations(profiles=[_profile("c1")], tags=[])
    report = asyncio.run(
        build_insight_rigor_audit_report(
            operations,
            InsightRigorAuditQuery(insights=("session_profiles",)),
        )
    )
    names = [entry.insight_name for entry in report.entries]
    assert names == ["session_profiles"]


#: Products whose backing rows genuinely carry no reliable materialization
#: version (live query-time aggregates over a hardcoded/sentinel value, or no
#: provenance field at all) -- see each contract's ``notes`` for why.
_PRODUCTS_WITHOUT_VERSION_FIELDS = frozenset({"archive_coverage", "cost_rollups", "usage_timeline", "archive_debt"})


@pytest.mark.parametrize("contract", list_rigor_contracts(), ids=lambda c: c.insight_name)
def test_each_contract_declares_at_least_one_version_field(contract) -> None:  # type: ignore[no-untyped-def]
    if contract.insight_name in _PRODUCTS_WITHOUT_VERSION_FIELDS:
        assert contract.notes, f"{contract.insight_name} needs notes justifying the missing version field"
        assert contract.version_fields == ()
        return
    assert len(contract.version_fields) >= 1, contract.insight_name


class _BrokenOperations:
    """Operations that raise on every list call — exercises error capture."""

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        async def _call(_query: object) -> list[object]:
            raise RuntimeError(f"simulated failure in {name}")

        return _call


def test_build_report_covers_every_registered_insight_not_just_contracted_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit iterates INSIGHT_REGISTRY, not list_rigor_contracts() (9e5.28):
    a registered product with no contract must appear as coverage_status
    "uncovered", never silently vanish from the report."""
    import polylogue.analysis.audit as audit_mod

    operations = _FakeOperations(profiles=[_profile("c1")], tags=[])
    monkeypatch.setattr(audit_mod, "get_rigor_contract", lambda name: None)
    report = asyncio.run(
        build_insight_rigor_audit_report(operations, InsightRigorAuditQuery(insights=("session_profiles",)))
    )
    [entry] = report.entries
    assert entry.insight_name == "session_profiles"
    assert entry.coverage_status == "uncovered"
    assert entry.sample_size == 0
    assert entry.error is None


def test_build_report_marks_exempt_products_distinctly_from_uncovered(monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.analysis.audit as audit_mod

    operations = _FakeOperations(profiles=[], tags=[])
    monkeypatch.setattr(audit_mod, "get_rigor_contract", lambda name: None)
    monkeypatch.setattr(audit_mod, "rigor_exemption_reason", lambda name: "test-only exemption justification")
    report = asyncio.run(
        build_insight_rigor_audit_report(operations, InsightRigorAuditQuery(insights=("session_profiles",)))
    )
    [entry] = report.entries
    assert entry.coverage_status == "exempt"
    assert entry.notes == ("test-only exemption justification",)


def test_build_report_covers_all_11_registered_insights_by_default() -> None:
    """Every currently-registered insight shows up in an unfiltered report,
    each either genuinely audited (covered, has a contract) or a stub
    (uncovered/exempt) -- none are silently skipped."""
    operations = _FakeOperations(profiles=[], tags=[])
    report = asyncio.run(build_insight_rigor_audit_report(operations, InsightRigorAuditQuery(sample_limit=1)))
    names = {entry.insight_name for entry in report.entries}
    assert names == set(INSIGHT_REGISTRY.keys())
    for entry in report.entries:
        assert entry.coverage_status in ("covered", "uncovered", "exempt")
        if entry.coverage_status != "covered":
            assert entry.sample_size == 0


def test_build_report_records_per_product_error_without_aborting() -> None:
    operations = _BrokenOperations()
    report = asyncio.run(
        build_insight_rigor_audit_report(
            operations,
            InsightRigorAuditQuery(insights=("session_profiles", "threads")),
        )
    )
    by_name = {entry.insight_name: entry for entry in report.entries}
    assert set(by_name) == {"session_profiles", "threads"}
    for entry in by_name.values():
        assert entry.error is not None
        assert entry.sample_size == 0
