"""Frozen excision coordinates retain the inputs needed for exact replay."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from polylogue.operations.machine_plan_context import context_from_replay, replay_context
from polylogue.operations.mutation_transaction import build_plan
from polylogue.security.excision import (
    ContainerDisposition,
    ContainerItem,
    ContainerMember,
    ExcisionRawTarget,
    ExcisionTarget,
    excision_target_from_replay,
    excision_target_replay,
)
from polylogue.storage.accepted_marker_inputs import MarkerInputExcisionTarget


def _target() -> ExcisionTarget:
    return ExcisionTarget(
        session_id="codex:neutral",
        session_exists=True,
        session_content_hash=bytes(range(32)),
        raw_targets=(ExcisionRawTarget("raw-neutral", bytes(reversed(range(32))), "neutral.json"),),
        message_ids=("neutral-message",),
        block_ids=("neutral-block",),
        hook_event_ids=("neutral-hook",),
        fact_raw_ids=("raw-fact",),
        sidecar_raw_ids=("raw-sidecar",),
        containers=ContainerDisposition(
            members=(ContainerMember("generation", "item", "coordinate", bytes(range(32))),),
            removable_items=(ContainerItem("generation", "item", bytes(range(32))),),
            retained_items=(ContainerItem("generation", "shared", None),),
        ),
        material_ids=("neutral-material",),
        material_blob_hashes=(bytes(range(32)),),
        marker_input_targets=(MarkerInputExcisionTarget("marker", "raw-neutral", "a" * 64, "accepted", "stream", 1),),
    )


def _context(target: ExcisionTarget) -> dict[str, object]:
    return {
        "session_id": target.session_id,
        "actor": "synthetic:operator",
        "found": target.found,
        "reason": "synthetic removal",
        "cascade_lineage": False,
        "lineage_dependent_session_ids": [],
        "source_marker_inputs_pending": 0,
        "source_marker_inputs_accepted": 1,
        "marker_input_digests": ["a" * 64],
        "targets": [excision_target_replay(target)],
        "user_frame_epoch": 11,
    }


def test_excision_replay_retains_exact_binary_targets_without_opening_tiers() -> None:
    target = _target()
    context = _context(target)
    encoded = replay_context("mutate-session-excision", context)
    assert encoded is not None
    decoded = context_from_replay("mutate-session-excision", encoded)
    assert decoded == context
    targets = decoded["targets"]
    assert isinstance(targets, list)
    assert excision_target_from_replay(targets[0]) == target


@pytest.mark.parametrize("field", ["session_content_hash", "raw_targets", "message_ids", "marker_input_targets"])
def test_original_coordinates_participate_in_excision_plan_identity(field: str) -> None:
    target = _target()
    original = _context(target)
    altered = deepcopy(original)
    targets = altered["targets"]
    assert isinstance(targets, list) and isinstance(targets[0], dict)
    targets[0][field] = bytes(reversed(range(32))).hex() if field == "session_content_hash" else []

    def plan(context: dict[str, object]) -> str:
        return build_plan(
            operation="mutate-session-excision",
            destructive_class="excise",
            target_refs=("session:codex:neutral",),
            affected_tiers=("source", "index", "embeddings", "user"),
            reversible=False,
            context=context,
        ).plan_hash

    assert plan(original) != plan(altered)


def test_excision_replay_refuses_unknown_nested_fields_and_missing_coordinates() -> None:
    value = excision_target_replay(_target())
    markers = value["marker_input_targets"]
    assert isinstance(markers, list) and isinstance(markers[0], dict)
    markers[0]["completion_proved"] = True
    with pytest.raises(ValueError):
        excision_target_from_replay(value)
    value = excision_target_replay(_target())
    del value["message_ids"]
    with pytest.raises(ValueError):
        excision_target_from_replay(value)


def test_excision_replay_has_no_target_count_cap() -> None:
    target = replace(_target(), message_ids=tuple(f"neutral-message-{i}" for i in range(1025)))
    context = _context(target)
    encoded = replay_context("mutate-session-excision", context)
    assert encoded is not None
    assert context_from_replay("mutate-session-excision", encoded) == context


def test_excision_replay_refuses_a_different_or_duplicated_cascade() -> None:
    context = _context(_target())
    context["lineage_dependent_session_ids"] = ["codex:unrelated"]
    context["cascade_lineage"] = True
    with pytest.raises(ValueError):
        replay_context("mutate-session-excision", context)
    context["lineage_dependent_session_ids"] = ["codex:neutral"]
    with pytest.raises(ValueError):
        replay_context("mutate-session-excision", context)


def test_user_population_precondition_participates_in_frozen_plan_identity() -> None:
    original = _context(_target())
    changed = {**original, "user_frame_epoch": 12}

    def identity(context: dict[str, object]) -> str:
        return build_plan(
            operation="mutate-session-excision",
            destructive_class="excise",
            target_refs=("session:codex:neutral",),
            affected_tiers=("source", "index", "embeddings", "user"),
            reversible=False,
            context=context,
        ).plan_hash

    assert identity(original) != identity(changed)
    with pytest.raises(ValueError):
        replay_context(
            "mutate-session-excision", {key: value for key, value in original.items() if key != "user_frame_epoch"}
        )
