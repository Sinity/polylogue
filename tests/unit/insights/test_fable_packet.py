from __future__ import annotations

from polylogue.insights.cohorts import CohortCandidate, CohortManifest, CohortSpec, compile_cohort_manifest
from polylogue.insights.fable_packet import (
    DelegationPacketLabel,
    DelegationPacketRow,
    compile_private_fable_packet,
)


def _manifest() -> CohortManifest:
    return compile_cohort_manifest(
        CohortSpec("delegations where mapping_state:resolved", "index:24:fable", "seed", 2),
        [
            CohortCandidate("delegation:one"),
            CohortCandidate("delegation:two"),
        ],
    )


def test_private_packet_reports_coverage_labels_distributions_and_limits() -> None:
    manifest = _manifest()
    packet = compile_private_fable_packet(
        manifest=manifest,
        rows=[
            DelegationPacketRow("delegation:one", "action", "resolved", "a" * 64),
            DelegationPacketRow("delegation:two", "action", "unresolved", "b" * 64),
            DelegationPacketRow("delegation:edge", "edge", "edge_only", None),
        ],
        annotation_schema_id="delegation-discourse/v1",
        labels=[
            DelegationPacketLabel(
                "delegation:one", "directive_mode", "direct", "batch-a", True, True, 0.9, ("block:a",)
            ),
            DelegationPacketLabel(
                "delegation:two", "directive_mode", "direct", "batch-b", True, True, 0.8, ("block:b",)
            ),
            DelegationPacketLabel(
                "delegation:two", "directive_mode", "collaborative", "batch-c", True, True, 0.7, ("block:b",)
            ),
            DelegationPacketLabel("delegation:one", "rationale", None, "batch-a", True, True, 0.9, ("block:a",)),
        ],
    )

    assert packet.status == "complete"
    assert packet.action_observed_count == 2
    assert packet.edge_only_count == 1
    assert packet.unresolved_count == 1
    assert packet.annotation_batches == ("batch-a", "batch-b", "batch-c")
    assert packet.disagreement_count == 1
    assert {
        (item.field, item.value, item.count, item.denominator_n, item.missing_n) for item in packet.distributions
    } == {
        ("directive_mode", "collaborative", 1, 3, 0),
        ("directive_mode", "direct", 2, 3, 0),
    }
    assert "no_comparative_authoritarianism_success_utility_or_routing_quality_claims" in packet.limits


def test_private_packet_fails_closed_when_accepted_labels_lack_evidence() -> None:
    manifest = _manifest()
    packet = compile_private_fable_packet(
        manifest=manifest,
        rows=[DelegationPacketRow("delegation:one", "action", "resolved", "a" * 64)],
        annotation_schema_id="delegation-discourse/v1",
        labels=[DelegationPacketLabel("delegation:one", "directive_mode", "direct", "batch-a", True, True, 1.0, ())],
    )

    assert packet.status == "not_supported"
    assert packet.not_supported_reasons == (
        "accepted_label_missing_evidence",
        "selected_refs_missing_from_structural_population",
    )
