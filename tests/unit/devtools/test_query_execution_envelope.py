"""Tests for the live query execution envelope lab command."""

from __future__ import annotations

from pathlib import Path

from devtools.query_execution_envelope import _proc_memory, _temp_used_bytes


def test_proc_memory_is_nonnegative() -> None:
    rss, pss, swap = _proc_memory()
    assert rss >= 0
    assert pss >= 0
    assert swap >= 0


def test_temp_usage_missing_path_is_zero(tmp_path: Path) -> None:
    assert _temp_used_bytes(tmp_path / "missing") == 0
