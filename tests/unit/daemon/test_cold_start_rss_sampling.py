"""Bounded procfs process-tree RSS sampling."""

from __future__ import annotations

from pathlib import Path

from tests.infra.daemon_cold_start import _process_tree_rss


def _fake_process(proc_root: Path, pid: int, rss_kib: int, task_children: dict[int, list[int]]) -> None:
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    (process / "status").write_text(f"Name:\ttest\nVmRSS:\t{rss_kib} kB\n", encoding="ascii")
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
    }


def test_process_tree_rss_reports_task_scan_limit(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _fake_process(proc_root, 100, 10, {100: [], 101: [], 102: []})

    measured = _process_tree_rss(100, proc_root=proc_root, max_task_ids=2)

    assert measured is not None
    assert measured["task_count"] == 2
    assert measured["truncated"] is True
