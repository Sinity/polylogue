"""Neutral original Codex state exports with declared graph operands."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from polylogue.sources.sqlite_export import logical_export_bytes
from polylogue.sources.sqlite_snapshot import member_export_scope
from tests.infra.retained_parser_payloads import _codex_thread_state_snapshot_bytes


def codex_thread_graph_snapshot_bytes(
    directory: Path,
    label: str,
    *,
    threads: Sequence[tuple[str, str]],
    spawn_edges: Sequence[tuple[str, str, str]],
) -> bytes:
    _codex_thread_state_snapshot_bytes(directory, label)
    state_path = directory / label / "state_5.sqlite"
    with sqlite3.connect(state_path) as state:
        state.execute("DELETE FROM threads")
        for identity, title in threads:
            state.execute(
                "INSERT INTO threads VALUES(?,?,?,?,?,?,?,?,?,?)",
                (identity, title, "/work", 1, 1, "cli", None, None, None, 0),
            )
        for parent, child, status in spawn_edges:
            state.execute("INSERT INTO thread_spawn_edges VALUES(?,?,?)", (parent, child, status))
    return logical_export_bytes(state_path, scope=member_export_scope(state_path))
