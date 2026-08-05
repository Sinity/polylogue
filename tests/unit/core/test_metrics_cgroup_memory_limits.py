"""polylogue-e98k: cgroup memory.max/memory.high readers used by the mmap budget check.

Production dependency exercised: the real ``read_cgroup_memory_max_bytes``/
``read_cgroup_memory_high_bytes`` -> ancestor-aware limit reader, not a
reimplemented parser. Reverting either reader to always return ``None``, to
read only the leaf cgroup, or to mis-treat the literal ``"max"`` value as a
real 0-byte limit would make these tests fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core import metrics as metrics_module
from polylogue.core.metrics import read_cgroup_memory_high_bytes, read_cgroup_memory_max_bytes


def _patch_cgroup_hierarchy(monkeypatch: pytest.MonkeyPatch, cgroup_root: Path, cgroup_path: str = "/") -> None:
    monkeypatch.setattr(metrics_module, "_CGROUP_V2_ROOT", cgroup_root)
    monkeypatch.setattr(metrics_module, "read_cgroup_path", lambda: cgroup_path)


def test_read_cgroup_memory_max_bytes_parses_numeric_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "memory.max").write_text("19327352832\n")  # 18 GiB
    _patch_cgroup_hierarchy(monkeypatch, tmp_path)
    assert read_cgroup_memory_max_bytes() == 19327352832


def test_read_cgroup_memory_high_bytes_parses_numeric_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "memory.high").write_text("15032385536\n")  # 14 GiB
    _patch_cgroup_hierarchy(monkeypatch, tmp_path)
    assert read_cgroup_memory_high_bytes() == 15032385536


def test_read_cgroup_memory_max_bytes_treats_literal_max_as_unlimited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "memory.max").write_text("max\n")
    _patch_cgroup_hierarchy(monkeypatch, tmp_path)
    assert read_cgroup_memory_max_bytes() is None


def test_read_cgroup_memory_max_bytes_missing_file_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No cgroup file at all (no controller / not in a cgroup) degrades to None,
    never an exception -- the dev-machine / cloud-sandbox ordinary case."""
    _patch_cgroup_hierarchy(monkeypatch, tmp_path)  # directory exists but no memory.max inside
    assert read_cgroup_memory_max_bytes() is None


def test_read_cgroup_memory_limits_use_minimum_finite_nested_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cgroup"
    parent = root / "parent"
    leaf = parent / "leaf"
    leaf.mkdir(parents=True)
    (root / "memory.max").write_text("max\n")
    (root / "memory.high").write_text("21474836480\n")  # 20 GiB
    (parent / "memory.max").write_text("12884901888\n")  # 12 GiB
    (parent / "memory.high").write_text("10737418240\n")  # 10 GiB
    (leaf / "memory.max").write_text("17179869184\n")  # 16 GiB
    (leaf / "memory.high").write_text("max\n")
    _patch_cgroup_hierarchy(monkeypatch, root, "/parent/leaf")

    assert read_cgroup_memory_max_bytes() == 12884901888
    assert read_cgroup_memory_high_bytes() == 10737418240


def test_read_cgroup_memory_limits_return_none_when_nested_hierarchy_is_unlimited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cgroup"
    parent = root / "parent"
    leaf = parent / "leaf"
    leaf.mkdir(parents=True)
    for cgroup_dir in (root, parent, leaf):
        (cgroup_dir / "memory.max").write_text("max\n")
        (cgroup_dir / "memory.high").write_text("max\n")
    _patch_cgroup_hierarchy(monkeypatch, root, "/parent/leaf")

    assert read_cgroup_memory_max_bytes() is None
    assert read_cgroup_memory_high_bytes() is None


def test_read_cgroup_memory_max_bytes_no_cgroup_path_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_module, "read_cgroup_path", lambda: None)
    assert read_cgroup_memory_max_bytes() is None
    assert read_cgroup_memory_high_bytes() is None
