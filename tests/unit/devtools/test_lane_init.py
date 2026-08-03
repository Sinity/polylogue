from __future__ import annotations

import json
from pathlib import Path

from devtools import lane_init


def test_recommended_workers_fair_share_floor_and_cap() -> None:
    assert lane_init.recommended_workers(16, cpu_count=24) == 1
    assert lane_init.recommended_workers(8, cpu_count=24) == 3
    assert lane_init.recommended_workers(2, cpu_count=24) == 4  # capped
    assert lane_init.recommended_workers(0, cpu_count=24) == 4  # clamped lanes
    assert lane_init.recommended_workers(64, cpu_count=24) == 1  # floor


def test_lane_record_shape_and_ledger_append(tmp_path: Path) -> None:
    record = lane_init.lane_record(
        worktree=tmp_path / "lane-x",
        branch="feature/x",
        base_sha="abc123def",
        beads=["polylogue-aaa", "polylogue-bbb"],
        venv=True,
        workers=2,
    )
    assert record["lane"] == "lane-x"
    assert record["status"] == "provisioned"
    assert record["beads"] == ["polylogue-aaa", "polylogue-bbb"]
    path = lane_init.append_ledger(tmp_path, record)
    lane_init.append_ledger(tmp_path, record)
    assert path == tmp_path / lane_init.LEDGER_RELPATH
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2 and lines[0]["branch"] == "feature/x"


def test_guard_check_flags_escaped_resolution(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    python = lane / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    # a fake "python" that reports a module path OUTSIDE the worktree
    python.write_text("#!/bin/sh\necho /somewhere/else/polylogue/__init__.py\n")
    python.chmod(0o755)
    error = lane_init._guard_check(lane)
    assert error is not None and "guard violation" in error
