from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from devtools import bd_reimport_guard as guard


@pytest.fixture(autouse=True)
def _isolated_snapshot_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect snapshot files under tmp_path instead of the real /tmp path.

    Prevents test runs from colliding with each other or with a real
    in-flight git hook using the same hook-name snapshot file.
    """
    monkeypatch.setattr(guard, "_snapshot_path", lambda hook_name: tmp_path / f"snapshot-{hook_name}.jsonl")


def test_snapshot_writes_live_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = {"polylogue-a": {"id": "polylogue-a", "updated_at": "2026-07-01T00:00:00Z"}}
    monkeypatch.setattr(guard, "_export_live_state", lambda: state)

    rc = guard.cmd_snapshot("post-checkout")

    assert rc == 0
    snap_path = guard._snapshot_path("post-checkout")
    assert json.loads(snap_path.read_text()) == state


def test_check_and_repair_is_noop_without_prior_snapshot() -> None:
    # No snapshot file exists for this hook name -- nothing to compare.
    rc = guard.cmd_check_and_repair("never-snapshotted")
    assert rc == 0


def test_check_and_repair_restores_clobbered_bead_and_leaves_unchanged_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {
        "polylogue-fresh": {"id": "polylogue-fresh", "updated_at": "2026-07-19T23:00:00Z", "title": "closed live"},
        "polylogue-stable": {"id": "polylogue-stable", "updated_at": "2026-07-18T10:00:00Z", "title": "unchanged"},
    }
    after = {
        # reimport clobbered this back to an older committed state
        "polylogue-fresh": {"id": "polylogue-fresh", "updated_at": "2026-07-10T08:00:00Z", "title": "closed live"},
        "polylogue-stable": {"id": "polylogue-stable", "updated_at": "2026-07-18T10:00:00Z", "title": "unchanged"},
    }
    states = iter([after])
    monkeypatch.setattr(guard, "_export_live_state", lambda: next(states))

    snap_path = guard._snapshot_path("post-checkout")
    snap_path.write_text(json.dumps(before))

    recorded: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
        recorded.append(cmd)
        if cmd[:2] == ["bd", "import"]:
            restored_ids = [json.loads(line)["id"] for line in Path(cmd[2]).read_text().splitlines() if line.strip()]
            assert restored_ids == ["polylogue-fresh"], "only the clobbered bead should be restored"
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = guard.cmd_check_and_repair("post-checkout")

    assert rc == 0
    assert not snap_path.exists(), "snapshot must be consumed"
    assert any(cmd[:2] == ["bd", "import"] for cmd in recorded)
    assert any(cmd[:2] == ["bd", "export"] for cmd in recorded)


def test_check_and_repair_restores_vanished_bead(monkeypatch: pytest.MonkeyPatch) -> None:
    before = {"polylogue-gone": {"id": "polylogue-gone", "updated_at": "2026-07-19T23:00:00Z"}}
    after: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(guard, "_export_live_state", lambda: after)

    snap_path = guard._snapshot_path("post-merge")
    snap_path.write_text(json.dumps(before))

    recorded: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
        recorded.append(cmd)
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = guard.cmd_check_and_repair("post-merge")

    assert rc == 0
    assert any(cmd[:2] == ["bd", "import"] for cmd in recorded)


def test_check_and_repair_noop_when_nothing_clobbered(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"polylogue-stable": {"id": "polylogue-stable", "updated_at": "2026-07-18T10:00:00Z"}}
    monkeypatch.setattr(guard, "_export_live_state", lambda: state)

    snap_path = guard._snapshot_path("post-checkout")
    snap_path.write_text(json.dumps(state))

    recorded: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
        recorded.append(cmd)
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = guard.cmd_check_and_repair("post-checkout")

    assert rc == 0
    assert recorded == [], "no bd import/export calls should fire when nothing was clobbered"


def test_main_requires_command_and_hook_name() -> None:
    assert guard.main([]) == 1
    assert guard.main(["snapshot"]) == 1


def test_main_rejects_unknown_command() -> None:
    assert guard.main(["frobnicate", "post-checkout"]) == 1


def test_main_dispatches_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_cmd_snapshot(hook_name: str) -> int:
        calls.append(hook_name)
        return 0

    monkeypatch.setattr(guard, "cmd_snapshot", fake_cmd_snapshot)
    rc = guard.main(["snapshot", "post-checkout"])
    assert rc == 0
    assert calls == ["post-checkout"]
