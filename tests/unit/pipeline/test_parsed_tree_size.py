from __future__ import annotations

import os
from pathlib import Path

import pytest

from polylogue.pipeline import parsed_tree_size


def test_cgroup_v2_limit_paths_include_nested_service_ancestry(tmp_path: Path) -> None:
    cgroup_root = tmp_path / "cgroup"
    service = cgroup_root / "user.slice" / "user-1000.slice" / "worker.service"
    service.mkdir(parents=True)
    membership = tmp_path / "self-cgroup"
    membership.write_text("0::/user.slice/user-1000.slice/worker.service\n")

    paths = parsed_tree_size._cgroup_memory_limit_paths(
        cgroup_v2_root=cgroup_root,
        cgroup_v1_root=tmp_path / "cgroup-v1-memory",
        proc_cgroup_path=membership,
    )

    assert service / "memory.max" in paths
    assert cgroup_root / "user.slice" / "memory.max" in paths
    assert cgroup_root / "memory.max" in paths


def test_effective_memory_caps_host_ram_to_nested_cgroup_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    limit_path = tmp_path / "worker-memory.max"
    limit_path.write_text(str(10 * 1024**3))
    monkeypatch.setattr(os, "sysconf", lambda key: 32 * 1024**3 if key == "SC_PHYS_PAGES" else 1)
    monkeypatch.setattr(parsed_tree_size, "_cgroup_memory_limit_paths", lambda: (limit_path,))

    assert parsed_tree_size.effective_physical_memory_bytes() == 10 * 1024**3
