"""polylogue-e98k: cgroup memory.max/memory.high readers used by the mmap budget check.

Production dependency exercised: the real ``read_cgroup_memory_max_bytes``/
``read_cgroup_memory_high_bytes`` -> ``_read_cgroup_int`` -> ``_cgroup_file``
chain, not a reimplemented parser. Reverting either reader to always return
``None`` (or to mis-treat the literal ``"max"`` value as a real 0-byte limit)
would make these tests fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core import metrics as metrics_module
from polylogue.core.metrics import read_cgroup_memory_high_bytes, read_cgroup_memory_max_bytes


def _patch_cgroup_file(monkeypatch: pytest.MonkeyPatch, cgroup_dir: Path) -> None:
    def _fake_cgroup_file(name: str) -> Path:
        return cgroup_dir / name

    monkeypatch.setattr(metrics_module, "_cgroup_file", _fake_cgroup_file)


def test_read_cgroup_memory_max_bytes_parses_numeric_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "memory.max").write_text("19327352832\n")  # 18 GiB
    _patch_cgroup_file(monkeypatch, tmp_path)
    assert read_cgroup_memory_max_bytes() == 19327352832


def test_read_cgroup_memory_high_bytes_parses_numeric_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "memory.high").write_text("15032385536\n")  # 14 GiB
    _patch_cgroup_file(monkeypatch, tmp_path)
    assert read_cgroup_memory_high_bytes() == 15032385536


def test_read_cgroup_memory_max_bytes_treats_literal_max_as_unlimited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "memory.max").write_text("max\n")
    _patch_cgroup_file(monkeypatch, tmp_path)
    assert read_cgroup_memory_max_bytes() is None


def test_read_cgroup_memory_max_bytes_missing_file_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No cgroup file at all (no controller / not in a cgroup) degrades to None,
    never an exception -- the dev-machine / cloud-sandbox ordinary case."""
    _patch_cgroup_file(monkeypatch, tmp_path)  # directory exists but no memory.max inside
    assert read_cgroup_memory_max_bytes() is None


def test_read_cgroup_memory_max_bytes_no_cgroup_path_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_module, "_cgroup_file", lambda name: None)
    assert read_cgroup_memory_max_bytes() is None
    assert read_cgroup_memory_high_bytes() is None
