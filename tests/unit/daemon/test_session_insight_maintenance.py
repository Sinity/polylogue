"""Exact sealed-page translation tests for session insight maintenance."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.daemon.convergence import SelectedSessionCounts, SelectedSessionOutcome, SelectedSessionTarget
from polylogue.daemon.derivation import DerivationFrame
from polylogue.daemon.session_insight_maintenance import SessionInsightMaintenance
from polylogue.operations.insight_acceptance import AcceptedInsightPart, AcceptedInsightTarget


def _part(*targets: AcceptedInsightTarget) -> AcceptedInsightPart:
    return AcceptedInsightPart(
        preview_ref="preview:session-part",
        authorization_ref="authorization:session-part",
        plan_hash="a" * 64,
        ordinal=0,
        page_count=1,
        manifest_digest="b" * 64,
        previous_preview_ref=None,
        scope_kind="explicit",
        index_generation="index-generation:/archive/generation-1/index.db",
        recipe_version="2",
        targets=targets,
    )


class _Owner:
    def __init__(self, outcomes: tuple[SelectedSessionOutcome, ...]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[DerivationFrame, tuple[SelectedSessionTarget, ...], str, str]] = []

    async def converge_selected(
        self,
        frame: DerivationFrame,
        *,
        targets: tuple[SelectedSessionTarget, ...],
        expected_generation: str,
        expected_recipe: str,
        stop_requested: Callable[[], str | None],
    ) -> tuple[SelectedSessionOutcome, ...]:
        assert stop_requested() is None
        self.calls.append((frame, targets, expected_generation, expected_recipe))
        return self.outcomes


def _frame(session_ids: tuple[str, ...]) -> DerivationFrame:
    return DerivationFrame(
        archive_root="/archive",
        source_revision="index-generation:/archive/generation-1/index.db",
        recipe_versions={"session_profile": "2"},
        scope=session_ids,
    )


def test_plan_binding_uses_the_actual_opened_index_path() -> None:
    """Synthesizing a generation token instead of checking the pinned path makes this red."""

    owner = _Owner(())
    maintenance = SessionInsightMaintenance(owner, frame_factory=_frame)  # type: ignore[arg-type]

    assert maintenance.plan_binding(opened_index_path=Path("/archive/generation-1/index.db")) == (
        "index-generation:/archive/generation-1/index.db",
        "2",
    )


def test_plan_binding_refuses_a_frame_for_a_different_opened_generation() -> None:
    """Accepting the active frame after a pinned-generation change makes this red."""

    owner = _Owner(())

    def other_generation(_: tuple[str, ...]) -> DerivationFrame:
        return DerivationFrame(
            archive_root="/archive",
            source_revision="index-generation:/archive/generation-2/index.db",
            recipe_versions={"session_profile": "2"},
            scope=(),
        )

    maintenance = SessionInsightMaintenance(owner, frame_factory=other_generation)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="supplied opened index generation"):
        maintenance.plan_binding(opened_index_path=Path("/archive/generation-1/index.db"))


@pytest.mark.asyncio
async def test_converge_part_preserves_exact_targets_and_complete_owner_receipts() -> None:
    """Replacing sealed refs with discovery, or summing view counts, makes this red."""

    owner = _Owner(
        (
            SelectedSessionOutcome(
                "a",
                "already_satisfied",
                "input:a",
                "output:a",
                SelectedSessionCounts(1, 2, 3),
                False,
            ),
            SelectedSessionOutcome(
                "b",
                "published",
                "input:b",
                "output:b",
                SelectedSessionCounts(1, 5, 8),
                True,
            ),
        )
    )
    maintenance = SessionInsightMaintenance(owner, frame_factory=_frame)  # type: ignore[arg-type]
    part = _part(AcceptedInsightTarget("session:a", "required"), AcceptedInsightTarget("session:b", "excess"))

    receipt = await maintenance.converge_part(part, stop_requested=lambda: None)

    assert owner.calls == [
        (
            _frame(("a", "b")),
            (SelectedSessionTarget("a", "required"), SelectedSessionTarget("b", "excess")),
            part.index_generation,
            part.recipe_version,
        )
    ]
    assert receipt.remaining_unattempted_target_refs == ()
    assert [(item.target_ref, item.disposition) for item in receipt.targets] == [
        ("session:a", "already_satisfied"),
        ("session:b", "published"),
    ]
    assert receipt.targets[0].certified_counts.profiles == 1
    assert receipt.targets[1].certified_counts.work_events == 5
    assert receipt.targets[1].certified_counts.phases == 8
    assert receipt.targets[0].publication_known_committed is False
    assert receipt.targets[1].publication_known_committed is True


@pytest.mark.asyncio
async def test_converge_part_returns_only_the_owner_unattempted_suffix() -> None:
    """Turning a cancelled suffix into pending receipts would make this red."""

    owner = _Owner(
        (
            SelectedSessionOutcome(
                "a",
                "published",
                "input:a",
                "output:a",
                SelectedSessionCounts(1, 0, 0),
                True,
            ),
        )
    )
    maintenance = SessionInsightMaintenance(owner, frame_factory=_frame)  # type: ignore[arg-type]
    part = _part(
        AcceptedInsightTarget("session:a", "required"),
        AcceptedInsightTarget("session:b", "required"),
        AcceptedInsightTarget("session:c", "excess"),
    )

    receipt = await maintenance.converge_part(part, stop_requested=lambda: None)

    assert [item.target_ref for item in receipt.targets] == ["session:a"]
    assert receipt.remaining_unattempted_target_refs == ("session:b", "session:c")


@pytest.mark.asyncio
async def test_converge_ingest_sessions_uses_fresh_generation_and_accepted_recipe() -> None:
    """Replacing the accepted recipe with live config makes this red."""

    owner = _Owner(
        (
            SelectedSessionOutcome(
                "a",
                "published",
                "input:a",
                "output:a",
                SelectedSessionCounts(1, 2, 3),
                True,
            ),
        )
    )
    maintenance = SessionInsightMaintenance(owner, frame_factory=_frame)  # type: ignore[arg-type]

    receipt = await maintenance.converge_ingest_sessions(
        ("a", "b"),
        expected_recipe="accepted-recipe",
        stop_requested=lambda: None,
    )

    assert owner.calls == [
        (
            _frame(("a", "b")),
            (SelectedSessionTarget("a", "required"), SelectedSessionTarget("b", "required")),
            "index-generation:/archive/generation-1/index.db",
            "accepted-recipe",
        )
    ]
    assert [item.target_ref for item in receipt.targets] == ["session:a"]
    assert receipt.remaining_unattempted_target_refs == ("session:b",)


@pytest.mark.asyncio
async def test_converge_part_refuses_owner_outcome_for_a_different_sealed_target() -> None:
    """Ignoring owner outcome order could attach a committed write to another authority ref."""

    owner = _Owner(
        (
            SelectedSessionOutcome(
                "other",
                "published",
                "input:other",
                "output:other",
                SelectedSessionCounts(1, 0, 0),
                True,
            ),
        )
    )
    maintenance = SessionInsightMaintenance(owner, frame_factory=_frame)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="accepted target order"):
        await maintenance.converge_part(
            _part(AcceptedInsightTarget("session:a", "required")), stop_requested=lambda: None
        )
