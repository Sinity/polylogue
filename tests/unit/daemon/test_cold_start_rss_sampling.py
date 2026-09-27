"""Bounded procfs process-tree RSS sampling."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tests.infra.daemon_cold_start import _proc_children_supported, _process_tree_rss


def _fake_process(proc_root: Path, pid: int, rss_kib: int, task_children: dict[int, list[int]]) -> None:
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    (process / "status").write_text(f"Name:\ttest\nVmRSS:\t{rss_kib} kB\n", encoding="ascii")
    # /proc/<pid>/stat field 22 (starttime) binds process identity against PID reuse.
    stat_fields = ["S", *("0" for _ in range(18)), str(pid + 1000)]
    (process / "stat").write_text(f"{pid} (test) {' '.join(stat_fields)}\n", encoding="ascii")
    for task_id, child_pids in task_children.items():
        children = process / "task" / str(task_id) / "children"
        children.parent.mkdir(parents=True)
        children.write_text(" ".join(map(str, child_pids)), encoding="ascii")


def test_process_tree_rss_follows_children_from_secondary_task(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _fake_process(proc_root, 100, 10, {100: [], 101: [200]})
    _fake_process(proc_root, 200, 23, {200: []})

    measured = _process_tree_rss(100, proc_root=proc_root)

    assert measured == {
        "rss_bytes": 33 * 1024,
        "process_count": 2,
        "task_count": 3,
        "truncated": False,
        "process_identities": [
            {"pid": 100, "start_time_ticks": 1100},
            {"pid": 200, "start_time_ticks": 1200},
        ],
    }


def test_process_tree_rss_reports_task_scan_limit(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _fake_process(proc_root, 100, 10, {100: [], 101: [], 102: []})

    measured = _process_tree_rss(100, proc_root=proc_root, max_task_ids=2)

    assert measured is not None
    assert measured["task_count"] == 2
    assert measured["truncated"] is True


def test_process_tree_rss_discards_partial_pid_at_children_read_limit(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _fake_process(proc_root, 100, 10, {100: [200, 123456]})
    _fake_process(proc_root, 200, 23, {200: []})
    _fake_process(proc_root, 1234, 9999, {1234: []})
    (proc_root / "100" / "task" / "100" / "children").write_text("200 123456", encoding="ascii")

    measured = _process_tree_rss(100, proc_root=proc_root, children_file_limit_bytes=7)

    assert measured is not None
    assert measured["rss_bytes"] == 33 * 1024
    assert measured["process_count"] == 2
    assert measured["truncated"] is True


def test_process_tree_rss_marks_disappeared_child_as_incomplete(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _fake_process(proc_root, 100, 10, {100: [200]})

    measured = _process_tree_rss(100, proc_root=proc_root)

    assert measured is not None
    assert measured["rss_bytes"] == 10 * 1024
    assert measured["truncated"] is True


def test_proc_children_support_probe_tracks_kernel_exposure(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    native_task = proc_root / str(os.getpid()) / "task" / str(threading.get_native_id())
    native_task.mkdir(parents=True)

    assert _proc_children_supported(proc_root) is False
    (native_task / "children").touch()
    assert _proc_children_supported(proc_root) is True
