"""A full insight rebuild is planned by scope, never by enumerating sessions (polylogue-buuxr)."""

from __future__ import annotations

from typing import Any

from polylogue.operations.mutation_actuators import InsightsRebuildActuator, InsightsRebuildArgs


class _Archive:
    index_db_path = "/archive/index.db"

    def list_summaries(self, **_kwargs: Any) -> list[object]:
        raise AssertionError("a full-scope plan must not enumerate the archive")

    def resolve_session_id(self, session_id: str) -> str:
        return session_id


def test_full_scope_plan_does_not_enumerate_sessions() -> None:
    """Anti-vacuity: restore ``list_summaries(limit=1_000_000)`` in prepare and this raises."""
    plan = InsightsRebuildActuator().prepare(InsightsRebuildArgs(archive=_Archive()))  # type: ignore[arg-type]

    assert plan.target_refs == ()
    assert plan.context["scope_kind"] == "full"
    assert plan.context["targets"] == []


def test_full_scope_plan_identity_is_stable_across_prepares() -> None:
    first = InsightsRebuildActuator().prepare(InsightsRebuildArgs(archive=_Archive()))  # type: ignore[arg-type]
    second = InsightsRebuildActuator().prepare(InsightsRebuildArgs(archive=_Archive()))  # type: ignore[arg-type]
    assert first.plan_hash == second.plan_hash


def test_explicit_scope_still_names_its_sessions() -> None:
    plan = InsightsRebuildActuator().prepare(
        InsightsRebuildArgs(archive=_Archive(), session_ids=("s1", "s2"))  # type: ignore[arg-type]
    )
    assert plan.context["scope_kind"] == "explicit"
    assert len(plan.target_refs) == 2
