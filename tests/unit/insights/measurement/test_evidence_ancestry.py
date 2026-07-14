from __future__ import annotations

from polylogue.insights.measurement.evidence_ancestry import (
    EvidenceEdge,
    EvidenceNode,
    walk_evidence_ancestry,
)

_ROOT = "finding:root"
_CHILD = "query:child"


def _clean_graph() -> tuple[dict[str, EvidenceNode], list[EvidenceEdge]]:
    nodes = {
        _ROOT: EvidenceNode(ref=_ROOT, kind="finding", epoch="epoch-5", definition_version="v1", frame_ref="frame-a"),
        _CHILD: EvidenceNode(ref=_CHILD, kind="query", epoch="epoch-5", definition_version="v1", frame_ref="frame-a"),
    }
    edges = [EvidenceEdge(src_ref=_ROOT, dst_ref=_CHILD)]
    return nodes, edges


def test_clean_dag_has_no_flags_and_clean_status() -> None:
    nodes, edges = _clean_graph()

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "clean"
    assert report.flags == ()


def test_circular_ancestry_is_detected_with_path_witness() -> None:
    nodes, edges = _clean_graph()
    edges.append(EvidenceEdge(src_ref=_CHILD, dst_ref=_ROOT))

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "blocked"
    circular = report.flags_of("circular-ancestry")
    assert len(circular) == 1
    assert circular[0].path == (_ROOT, _CHILD, _ROOT)


def test_self_referential_detector_output_is_flagged_circular_without_a_literal_cycle() -> None:
    nodes, edges = _clean_graph()

    report = walk_evidence_ancestry(_ROOT, nodes, edges, detector_output_refs=frozenset({_CHILD}))

    assert report.status == "blocked"
    assert len(report.flags_of("circular-ancestry")) == 1
    assert "own prior output" in report.flags_of("circular-ancestry")[0].detail


def test_epoch_skew_is_flagged_but_does_not_block() -> None:
    nodes, edges = _clean_graph()
    nodes[_CHILD] = EvidenceNode(
        ref=_CHILD, kind="query", epoch="epoch-3", definition_version="v1", frame_ref="frame-a"
    )

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "flagged"
    assert len(report.flags_of("epoch-skew")) == 1
    assert report.flags_of("epoch-skew")[0].path == (_ROOT, _CHILD)


def test_definition_incompatible_is_flagged_but_does_not_block() -> None:
    nodes, edges = _clean_graph()
    nodes[_CHILD] = EvidenceNode(
        ref=_CHILD, kind="query", epoch="epoch-5", definition_version="v2", frame_ref="frame-a"
    )

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "flagged"
    assert len(report.flags_of("definition-incompatible")) == 1


def test_expired_ref_blocks() -> None:
    nodes, edges = _clean_graph()
    nodes[_CHILD] = EvidenceNode(ref=_CHILD, kind="query", ref_state="expired")

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "blocked"
    assert len(report.flags_of("expired-ref")) == 1


def test_stale_ref_blocks() -> None:
    nodes, edges = _clean_graph()
    nodes[_CHILD] = EvidenceNode(ref=_CHILD, kind="query", ref_state="stale")

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "blocked"
    assert len(report.flags_of("stale-ref")) == 1


def test_missing_ref_blocks_when_node_absent_from_the_graph() -> None:
    nodes = {_ROOT: EvidenceNode(ref=_ROOT, kind="finding")}
    edges = [EvidenceEdge(src_ref=_ROOT, dst_ref=_CHILD)]

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "blocked"
    missing = report.flags_of("missing-ref")
    assert len(missing) == 1
    assert missing[0].path == (_ROOT, _CHILD)


def test_missing_ref_blocks_when_node_present_but_explicitly_missing_state() -> None:
    nodes, edges = _clean_graph()
    nodes[_CHILD] = EvidenceNode(ref=_CHILD, kind="query", ref_state="missing")

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "blocked"
    assert len(report.flags_of("missing-ref")) == 1


def test_ambiguous_ref_blocks() -> None:
    nodes, edges = _clean_graph()
    nodes[_CHILD] = EvidenceNode(ref=_CHILD, kind="query", ref_state="ambiguous")

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "blocked"
    assert len(report.flags_of("ambiguous-ref")) == 1


def test_quarantined_ref_blocks() -> None:
    nodes, edges = _clean_graph()
    nodes[_CHILD] = EvidenceNode(ref=_CHILD, kind="query", ref_state="quarantined")

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "blocked"
    assert len(report.flags_of("quarantined-ref")) == 1


def test_frame_coverage_drift_via_differing_frame_ref_is_flagged_but_does_not_block() -> None:
    nodes, edges = _clean_graph()
    nodes[_CHILD] = EvidenceNode(
        ref=_CHILD, kind="query", epoch="epoch-5", definition_version="v1", frame_ref="frame-b"
    )

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "flagged"
    assert len(report.flags_of("frame-coverage-drift")) == 1


def test_frame_coverage_drift_via_incomplete_coverage_flag() -> None:
    nodes, edges = _clean_graph()
    nodes[_CHILD] = EvidenceNode(
        ref=_CHILD,
        kind="query",
        epoch="epoch-5",
        definition_version="v1",
        frame_ref=None,
        frame_coverage_complete=False,
    )

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "flagged"
    assert len(report.flags_of("frame-coverage-drift")) == 1


def test_private_or_excised_blocks() -> None:
    nodes, edges = _clean_graph()
    nodes[_CHILD] = EvidenceNode(ref=_CHILD, kind="query", ref_state="private")

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "blocked"
    assert len(report.flags_of("private-or-excised")) == 1


def test_distinct_failures_on_the_same_node_are_never_flattened_into_one_flag() -> None:
    nodes, edges = _clean_graph()
    nodes[_CHILD] = EvidenceNode(
        ref=_CHILD,
        kind="query",
        epoch="epoch-3",  # skewed
        definition_version="v2",  # incompatible
        ref_state="stale",  # blocking
        frame_ref="frame-b",  # drifted
    )

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    kinds = {flag.kind for flag in report.flags}
    assert kinds == {"epoch-skew", "definition-incompatible", "stale-ref", "frame-coverage-drift"}
    assert report.status == "blocked"


def test_does_not_revisit_a_shared_descendant_reached_by_two_paths() -> None:
    shared = "query:shared"
    nodes = {
        _ROOT: EvidenceNode(ref=_ROOT, kind="finding"),
        "a": EvidenceNode(ref="a", kind="query"),
        "b": EvidenceNode(ref="b", kind="query"),
        shared: EvidenceNode(ref=shared, kind="query"),
    }
    edges = [
        EvidenceEdge(src_ref=_ROOT, dst_ref="a"),
        EvidenceEdge(src_ref=_ROOT, dst_ref="b"),
        EvidenceEdge(src_ref="a", dst_ref=shared),
        EvidenceEdge(src_ref="b", dst_ref=shared),
    ]

    report = walk_evidence_ancestry(_ROOT, nodes, edges)

    assert report.status == "clean"
    assert report.flags == ()
