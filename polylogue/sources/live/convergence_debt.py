"""Helpers for post-ingest convergence debt classification."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ConvergenceDebt:
    path: Path
    stage: str
    error: str | None = None
    # True when this row reflects a stage's deliberate bounded-backpressure
    # deferral (``ConvergenceStage.false_means_pending``, e.g. "insights
    # deferred until quiet") rather than a genuine check/execute failure.
    # Drives ``convergence_debt.status`` ("deferred" vs "failed") so daemon
    # health/alerting doesn't mistake normal backpressure for breakage.
    deferred: bool = False


def debt_by_path(debts: Iterable[ConvergenceDebt]) -> dict[Path, tuple[ConvergenceDebt, ...]]:
    grouped: dict[Path, list[ConvergenceDebt]] = {}
    for debt in debts:
        grouped.setdefault(debt.path, []).append(debt)
    return {path: tuple(items) for path, items in grouped.items()}


def convergence_debt_from_states(paths: Iterable[Path], states: object) -> list[ConvergenceDebt]:
    if not isinstance(states, dict):
        return [
            ConvergenceDebt(path=path, stage="convergence", error="converger returned invalid state") for path in paths
        ]
    debt: list[ConvergenceDebt] = []
    for path in paths:
        state = states.get(path)
        if state is None:
            debt.append(ConvergenceDebt(path=path, stage="convergence", error="missing convergence state"))
            continue
        if bool(getattr(state, "converged", False)):
            continue
        debt.extend(convergence_debt_from_state(path, state))
    return debt


def stage_state_value(value: object) -> str:
    """Normalize a ``StageState`` (or its plain string mirror) to its value."""
    return str(getattr(value, "value", value))


def is_deferred_stage_state(value: object) -> bool:
    """Return whether ``value`` reflects a deliberate backpressure deferral.

    ``StageState.PENDING`` (``daemon/convergence.py``) is set only for two
    non-failure reasons: a ``false_means_pending`` stage returned ``False``
    after doing bounded successful work, or a downstream stage is queued
    behind an unfinished barrier stage. Neither is an error, so debt rows
    carrying this state should read as "deferred", not "failed".
    """
    return stage_state_value(value) == "pending"


def convergence_debt_from_state(path: Path, state: object) -> list[ConvergenceDebt]:
    stages = getattr(state, "stages", None)
    last_error = getattr(state, "last_error", None)
    if not isinstance(stages, dict) or not stages:
        return [ConvergenceDebt(path=path, stage="convergence", error=optional_error(last_error))]
    debts: list[ConvergenceDebt] = []
    for stage_name, stage_state in stages.items():
        state_value = stage_state_value(stage_state)
        if state_value in {"done", "skipped"}:
            continue
        debts.append(
            ConvergenceDebt(
                path=path,
                stage=str(stage_name),
                error=optional_error(last_error) or f"stage state: {state_value}",
                deferred=state_value == "pending",
            )
        )
    if not debts:
        debts.append(ConvergenceDebt(path=path, stage="convergence", error=optional_error(last_error)))
    return debts


def optional_error(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


__all__ = [
    "ConvergenceDebt",
    "convergence_debt_from_state",
    "convergence_debt_from_states",
    "debt_by_path",
    "is_deferred_stage_state",
    "optional_error",
    "stage_state_value",
]
