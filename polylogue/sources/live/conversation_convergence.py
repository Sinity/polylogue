"""Conversation-scoped live convergence helper."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from polylogue.sources.live.convergence_debt import ConvergenceDebt, convergence_debt_from_state
from polylogue.sources.live.cursor import CursorStore


def converge_known_conversations(
    *,
    cursor: CursorStore,
    converger: object,
    paths: Iterable[Path],
    started: float,
) -> tuple[set[Path], float, dict[str, float], list[ConvergenceDebt]] | None:
    converge_conversations = getattr(converger, "converge_conversations", None)
    if not callable(converge_conversations):
        return None

    unique_paths = tuple(sorted(dict.fromkeys(paths)))
    from polylogue.storage.source_conversations import conversation_ids_for_source_paths

    try:
        with cursor._connect() as conn:
            by_path = conversation_ids_for_source_paths(conn, unique_paths)
    except Exception:
        return None
    if not by_path or not all(by_path.get(path) for path in unique_paths):
        return None

    conversation_ids = tuple(dict.fromkeys(conversation_id for ids in by_path.values() for conversation_id in ids))
    states, timings = converge_conversations(conversation_ids)
    completed = {
        path
        for path, ids in by_path.items()
        if ids
        and all(bool(getattr(_state_for(states, conversation_id), "converged", False)) for conversation_id in ids)
    }
    debt_items: list[ConvergenceDebt] = []
    for path, ids in by_path.items():
        for conversation_id in ids:
            state = _state_for(states, conversation_id)
            if state is not None and bool(getattr(state, "converged", False)):
                continue
            if state is None:
                debt_items.append(
                    ConvergenceDebt(
                        path=path,
                        stage="convergence",
                        error=f"missing convergence state for {conversation_id}",
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


def _state_for(states: object, conversation_id: str) -> Any | None:
    if not isinstance(states, dict):
        return None
    return states.get(conversation_id)
