"""Session-scoped live convergence helper."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from polylogue.sources.live.convergence_debt import ConvergenceDebt, convergence_debt_from_state
from polylogue.sources.live.cursor import CursorStore


def converge_known_sessions(
    *,
    cursor: CursorStore,
    converger: object,
    paths: Iterable[Path],
    started: float,
    archive_root: Path | None = None,
) -> tuple[set[Path], float, dict[str, float], list[ConvergenceDebt]] | None:
    converge_sessions = getattr(converger, "converge_sessions", None)
    if not callable(converge_sessions):
        return None

    unique_paths = tuple(sorted(dict.fromkeys(paths)))
    by_path = _schema_archive_session_ids_for_source_paths(archive_root, unique_paths)
    if not by_path or not all(by_path.get(path) for path in unique_paths):
        return None

    session_ids = tuple(dict.fromkeys(session_id for ids in by_path.values() for session_id in ids))
    states, timings = converge_sessions(session_ids)
    completed = {
        path
        for path, ids in by_path.items()
        if ids and all(bool(getattr(_state_for(states, session_id), "converged", False)) for session_id in ids)
    }
    debt_items: list[ConvergenceDebt] = []
    for path, ids in by_path.items():
        for session_id in ids:
            state = _state_for(states, session_id)
            if state is not None and bool(getattr(state, "converged", False)):
                continue
            if state is None:
                debt_items.append(
                    ConvergenceDebt(
                        path=path,
                        stage="convergence",
                        error=f"missing convergence state for {session_id}",
                    )
                )
            else:
                debt_items.extend(convergence_debt_from_state(path, state))
    return (
        completed,
        time.perf_counter() - started,
        {stage_name: float(elapsed) for stage_name, elapsed in timings.items()},
        debt_items,
    )


def _state_for(states: object, session_id: str) -> Any | None:
    if not isinstance(states, dict):
        return None
    return states.get(session_id)


def _schema_archive_session_ids_for_source_paths(
    archive_root: Path | None,
    paths: tuple[Path, ...],
) -> dict[Path, tuple[str, ...]]:
    if archive_root is None:
        return {}
    from polylogue.sources.live.batch_observability import session_ids_for_source_path

    return {path: session_ids_for_source_path(path, archive_root=archive_root) for path in paths}
