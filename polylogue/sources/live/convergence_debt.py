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


def convergence_debt_from_state(path: Path, state: object) -> list[ConvergenceDebt]:
    stages = getattr(state, "stages", None)
    last_error = getattr(state, "last_error", None)
    if not isinstance(stages, dict) or not stages:
        return [ConvergenceDebt(path=path, stage="convergence", error=optional_error(last_error))]
    debts: list[ConvergenceDebt] = []
    for stage_name, stage_state in stages.items():
        state_value = getattr(stage_state, "value", stage_state)
        if state_value in {"done", "skipped"}:
            continue
        debts.append(
            ConvergenceDebt(
                path=path,
                stage=str(stage_name),
                error=optional_error(last_error) or f"stage state: {state_value}",
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
    "optional_error",
]
