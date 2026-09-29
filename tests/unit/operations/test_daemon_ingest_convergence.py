"""Ingest profile convergence records whether it finished (#5639 review)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.operations.daemon_ingest import IngestExecution
from polylogue.operations.insight_acceptance import (
    InsightCertifiedCounts,
    SessionInsightPartReceipt,
    SessionInsightTargetReceipt,
)


def _target(session_id: str, disposition: str) -> SessionInsightTargetReceipt:
    return SessionInsightTargetReceipt(
        target_ref=f"session:{session_id}",
        disposition=disposition,  # type: ignore[arg-type]
        input_binding=None,
        output_binding=None,
        certified_counts=InsightCertifiedCounts(profiles=0),
        publication_known_committed=disposition == "published",
    )


class _Receipt:
    complete = True

    def __init__(self, pages: list[tuple[str, ...]]) -> None:
        self._pages = pages

    def session_page(self, cursor: str | None) -> tuple[str, ...]:
        if cursor is None:
            return self._pages[0]
        index = next(i for i, page in enumerate(self._pages) if page[-1] == cursor)
        return self._pages[index + 1] if index + 1 < len(self._pages) else ()


def _execution(dispositions: dict[str, str], attempted: list[tuple[str, ...]]) -> Any:
    async def converge_ingest_sessions(session_ids: tuple[str, ...], **_kwargs: object) -> SessionInsightPartReceipt:
        attempted.append(session_ids)
        return SessionInsightPartReceipt(targets=tuple(_target(sid, dispositions[sid]) for sid in session_ids))

    return SimpleNamespace(
        profile_convergence_complete=None,
        started_mutation=SimpleNamespace(plan=SimpleNamespace(context={"recipe_version": "r1"})),
        runtime=SimpleNamespace(converge_ingest_sessions=converge_ingest_sessions),
        stop_reason=lambda: None,
        check_stop=lambda: None,
    )


@pytest.mark.asyncio
async def test_stopped_profile_convergence_is_recorded_as_incomplete() -> None:
    """A page with a retryable target stops convergence, and the stop is recorded.

    The later page is never attempted. Anti-vacuity: restoring the bare
    ``break`` that fell through to a ``completed`` finalization leaves
    ``profile_convergence_complete`` true after an unconverged run.
    """
    attempted: list[tuple[str, ...]] = []
    execution = _execution({"a": "published", "b": "pending", "c": "published"}, attempted)

    parts = await IngestExecution.converge_profiles(execution, cast(Any, _Receipt([("a", "b"), ("c",)])))

    assert attempted == [("a", "b")]
    assert len(parts) == 1
    assert execution.profile_convergence_complete is False


@pytest.mark.asyncio
async def test_fully_converged_profiles_are_recorded_as_complete() -> None:
    attempted: list[tuple[str, ...]] = []
    execution = _execution({"a": "published", "b": "already_satisfied", "c": "published"}, attempted)

    await IngestExecution.converge_profiles(execution, cast(Any, _Receipt([("a", "b"), ("c",)])))

    assert attempted == [("a", "b"), ("c",)]
    assert execution.profile_convergence_complete is True
