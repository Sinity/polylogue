"""Tests proving each pathology composer's claimed structural property.

These are structural assertions on returned ``ComposedPathology`` values. The
composer is the test-infrastructure route that supplies ordered raw payloads
to future production-ingestion metamorphic tests.
"""

from __future__ import annotations

import json

import pytest

from tests.infra.pathology_composer import (
    ComposedPathology,
    compose_append_revision_chain,
    compose_fork_prefix_tail_lineage,
    compose_multi_session_bundle,
    compose_pathologies,
    compose_quarantined_head_arrangement,
    compose_vintage_variant_pair,
    compose_whale_scale_component,
    extract_new_shape_turns,
    extract_old_shape_turns,
)

# ---------------------------------------------------------------------------
# 1. Append revision chain
# ---------------------------------------------------------------------------


def test_revision_chain_has_claimed_length_and_growing_message_counts() -> None:
    result = compose_append_revision_chain(revision_count=5, messages_per_revision=3)

    assert len(result.sessions) == 5
    message_counts = [len(session.messages) for session in result.sessions]
    assert message_counts == [3, 6, 9, 12, 15]


def test_revision_chain_all_revisions_share_one_archive_session_id() -> None:
    result = compose_append_revision_chain(session_id="chain-abc", revision_count=3)

    ids = {session.id for session in result.sessions}
    assert ids == {"chain-abc"}


def test_revision_chain_self_describing_identity_toggle() -> None:
    with_identity = compose_append_revision_chain(session_id="rc", with_self_describing_identity=True)
    without_identity = compose_append_revision_chain(session_id="rc", with_self_describing_identity=False)

    assert all(session.metadata.get("source_identity") == "rc" for session in with_identity.sessions)
    assert all("source_identity" not in session.metadata for session in without_identity.sessions)


def test_revision_chain_rejects_degenerate_parameters() -> None:
    with pytest.raises(ValueError):
        compose_append_revision_chain(revision_count=0)
    with pytest.raises(ValueError):
        compose_append_revision_chain(messages_per_revision=0)


# ---------------------------------------------------------------------------
# 2. Fork / prefix-tail lineage
# ---------------------------------------------------------------------------


def test_lineage_child_shares_exact_message_id_and_text_prefix_with_parent() -> None:
    result = compose_fork_prefix_tail_lineage(shared_prefix_len=4, child_tail_len=2)
    parent, child = result.sessions

    parent_prefix = [(m.id, m.text) for m in list(parent.messages)[:4]]
    child_prefix = [(m.id, m.text) for m in list(child.messages)[:4]]
    assert parent_prefix == child_prefix


def test_lineage_child_tail_diverges_from_parent() -> None:
    result = compose_fork_prefix_tail_lineage(shared_prefix_len=3, child_tail_len=2)
    parent, child = result.sessions

    parent_ids = {m.id for m in parent.messages}
    child_tail_ids = {m.id for m in list(child.messages)[3:]}
    assert child_tail_ids.isdisjoint(parent_ids)
    assert len(child_tail_ids) == 2


def test_lineage_child_parent_id_points_at_parent() -> None:
    result = compose_fork_prefix_tail_lineage(parent_id="p1", child_id="c1")
    parent, child = result.sessions

    assert child.parent_id == "p1"
    assert parent.id == "p1"


def test_lineage_cycle_candidate_produces_genuine_two_cycle() -> None:
    result = compose_fork_prefix_tail_lineage(parent_id="p2", child_id="c2", cycle_candidate=True)
    parent, child = result.sessions

    assert child.parent_id == "p2"
    assert parent.parent_id == "c2"


def test_lineage_without_cycle_candidate_parent_has_no_parent_id() -> None:
    result = compose_fork_prefix_tail_lineage(cycle_candidate=False)
    parent, _child = result.sessions

    assert parent.parent_id is None


# ---------------------------------------------------------------------------
# 3. Multi-session bundle
# ---------------------------------------------------------------------------


def test_multi_session_bundle_has_claimed_session_multiplicity() -> None:
    records = [{"seq": i} for i in range(9)]
    result = compose_multi_session_bundle(records, session_count=3)

    (grouped_jsonl,) = result.raw_payloads
    assert isinstance(grouped_jsonl, str)
    lines = [json.loads(line) for line in grouped_jsonl.splitlines() if line]
    session_ids = {entry["sessionId"] for entry in lines}

    assert len(lines) == 9
    assert len(session_ids) == 3


def test_multi_session_bundle_preserves_all_input_records() -> None:
    expected_markers = [f"rec-{i}" for i in range(6)]
    records = [{"seq": i, "marker": marker} for i, marker in enumerate(expected_markers)]
    result = compose_multi_session_bundle(records, session_count=2)

    (grouped_jsonl,) = result.raw_payloads
    assert isinstance(grouped_jsonl, str)
    lines = [json.loads(line) for line in grouped_jsonl.splitlines() if line]
    markers = sorted(entry["marker"] for entry in lines)
    assert markers == sorted(expected_markers)


def test_multi_session_bundle_rejects_single_session_count() -> None:
    with pytest.raises(ValueError):
        compose_multi_session_bundle([{"a": 1}], session_count=1)


def test_multi_session_bundle_rejects_empty_records() -> None:
    with pytest.raises(ValueError):
        compose_multi_session_bundle([], session_count=2)


# ---------------------------------------------------------------------------
# 4. Whale-scale component
# ---------------------------------------------------------------------------


def test_whale_component_materializes_far_fewer_messages_than_declared() -> None:
    result = compose_whale_scale_component(declared_message_count=50_000, materialized_message_count=16)
    (session,) = result.sessions

    assert len(session.messages) == 16
    scale = session.metadata["_whale_scale"]
    assert isinstance(scale, dict)
    assert scale["declared_message_count"] == 50_000
    assert scale["materialized_message_count"] == 16
    assert len(session.messages) < scale["declared_message_count"]


def test_whale_component_rejects_materializing_the_actual_scale() -> None:
    with pytest.raises(ValueError):
        compose_whale_scale_component(materialized_message_count=10_000)


def test_whale_component_rejects_materialized_exceeding_declared() -> None:
    with pytest.raises(ValueError):
        compose_whale_scale_component(declared_message_count=10, materialized_message_count=20)


# ---------------------------------------------------------------------------
# 5. Quarantined-head arrangement
# ---------------------------------------------------------------------------


def test_quarantined_head_parent_reference_is_not_among_returned_sessions() -> None:
    result = compose_quarantined_head_arrangement(child_id="child-x", missing_parent_id="ghost-parent")

    session_ids = {session.id for session in result.sessions}
    assert "child-x" in session_ids
    assert "ghost-parent" not in session_ids
    assert result.sessions[0].parent_id == "ghost-parent"


def test_quarantined_head_metadata_names_the_missing_parent() -> None:
    result = compose_quarantined_head_arrangement(missing_parent_id="ghost-2")
    assert result.metadata["missing_parent_native_id"] == "ghost-2"


# ---------------------------------------------------------------------------
# 6. Vintage-variant pair
# ---------------------------------------------------------------------------


def test_vintage_variant_pair_has_different_wire_shape() -> None:
    result = compose_vintage_variant_pair()
    old_shape, new_shape = result.raw_payloads
    assert isinstance(old_shape, dict)
    assert isinstance(new_shape, dict)

    assert "messages" in old_shape
    assert "conversation" in new_shape
    assert "messages" not in new_shape
    assert "conversation" not in old_shape


def test_vintage_variant_pair_extracted_content_is_equal_despite_shape_difference() -> None:
    turns = (("user", "one"), ("assistant", "two"), ("user", "three"))
    result = compose_vintage_variant_pair(turns=turns)
    old_shape, new_shape = result.raw_payloads
    assert isinstance(old_shape, dict)
    assert isinstance(new_shape, dict)

    old_turns = extract_old_shape_turns(old_shape)
    new_turns = extract_new_shape_turns(new_shape)

    assert old_turns == new_turns == list(turns)


# ---------------------------------------------------------------------------
# Composition and ingestion order
# ---------------------------------------------------------------------------


def test_composition_nests_existing_pathologies_and_orders_flat_raw_payloads() -> None:
    bundle = compose_multi_session_bundle([{"record": 1}, {"record": 2}], session_count=2)
    variants = compose_vintage_variant_pair()
    nested = compose_pathologies(bundle, variants, name="raw-shapes")
    composed = compose_append_revision_chain(session_id="nested-append", revision_count=2).compose(
        nested,
        raw_ingestion_order=(2, 0, 1),
    )

    assert composed.components == (compose_append_revision_chain(session_id="nested-append", revision_count=2), nested)
    assert nested.components == (bundle, variants)
    assert composed.raw_ingestion_order == (2, 0, 1)
    assert composed.raw_payloads_in_ingestion_order == (
        variants.raw_payloads[1],
        bundle.raw_payloads[0],
        variants.raw_payloads[0],
    )


@pytest.mark.parametrize("order", [(0, 0), (0,), (0, 2), ("0", 1)])
def test_raw_ingestion_order_must_be_a_complete_index_permutation(order: tuple[object, ...]) -> None:
    pathology = compose_vintage_variant_pair()

    with pytest.raises(ValueError, match="permutation"):
        pathology.with_raw_ingestion_order(order)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ComposedPathology provenance contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        compose_append_revision_chain(),
        compose_fork_prefix_tail_lineage(),
        compose_multi_session_bundle([{"a": 1}, {"b": 2}]),
        compose_whale_scale_component(),
        compose_quarantined_head_arrangement(),
        compose_vintage_variant_pair(),
    ],
)
def test_every_composed_pathology_carries_provenance(result: ComposedPathology) -> None:
    """Every zoo member must be labeled with the pathology it carries and the
    bead/issue that motivated it (polylogue-yazae growth rule)."""
    assert result.pathology
    assert result.motivated_by
    assert result.description
    assert result.sessions or result.raw_payloads
