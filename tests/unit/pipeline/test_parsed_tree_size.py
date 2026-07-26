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


def test_cgroup_v1_paths_use_combined_memory_controller_mount(tmp_path: Path) -> None:
    mount = tmp_path / "cgroup-cpu-memory"
    worker = mount / "worker.service"
    worker.mkdir(parents=True)
    membership = tmp_path / "self-cgroup"
    membership.write_text("5:cpu,memory:/slice/worker.service\n")
    mountinfo = tmp_path / "self-mountinfo"
    mountinfo.write_text(f"42 25 0:42 /slice {mount} rw - cgroup cgroup rw,cpu,memory\n")

    paths = parsed_tree_size._cgroup_memory_limit_paths(
        cgroup_v2_root=tmp_path / "cgroup-v2",
        cgroup_v1_root=tmp_path / "canonical-memory",
        proc_cgroup_path=membership,
        proc_mountinfo_path=mountinfo,
    )

    assert worker / "memory.limit_in_bytes" in paths
    assert mount / "memory.limit_in_bytes" in paths


def test_effective_memory_caps_host_ram_to_nested_cgroup_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    limit_path = tmp_path / "worker-memory.max"
    limit_path.write_text(str(10 * 1024**3))
    monkeypatch.setattr(os, "sysconf", lambda key: 32 * 1024**3 if key == "SC_PHYS_PAGES" else 1)
    monkeypatch.setattr(parsed_tree_size, "_cgroup_memory_limit_paths", lambda: (limit_path,))

    assert parsed_tree_size.effective_physical_memory_bytes() == 10 * 1024**3
