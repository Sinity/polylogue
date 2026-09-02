"""A retry or reorder attempt must not clear a poison-pill cursor state.

Anti-vacuity: making ``reset_failures`` clear an excluded record turns the
final assertion red.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.cursor_lifecycle import (
    CursorLifecycleState,
    classify_cursor_lifecycle_state,
)


def test_reset_failures_does_not_revive_an_excluded_cursor(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "ops.sqlite")
    source = tmp_path / "capture.jsonl"
    source.write_text('{"event":"one"}\n', encoding="utf-8")

    store.mark_failed(source)
    assert classify_cursor_lifecycle_state(store.get_record(source)) is CursorLifecycleState.RETRY_PENDING

    store.reset_failures(source)
    store.mark_excluded(source)
    assert classify_cursor_lifecycle_state(store.get_record(source)) is CursorLifecycleState.EXCLUDED

    store.reset_failures(source)
    assert classify_cursor_lifecycle_state(store.get_record(source)) is CursorLifecycleState.EXCLUDED
