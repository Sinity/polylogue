"""Coordination envelope tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from polylogue.coordination.envelope import CommandResult, build_coordination_envelope


class FakeRunner:
    def __init__(
        self,
        root: Path,
        *,
        beads_rows: list[dict[str, object]] | None,
        gates: list[dict[str, object]] | None = None,
        merge_slot: dict[str, object] | None = None,
    ) -> None:
        self.root = root
        self.beads_rows = beads_rows
        self.gates = gates
        self.merge_slot = merge_slot

    def __call__(self, args: Sequence[str], cwd: Path | None) -> CommandResult:
        key = tuple(args)
        if key[:4] == ("git", "-C", str(self.root), "rev-parse") and key[-1] == "--show-toplevel":
            return CommandResult(key, 0, str(self.root) + "\n", "")
        if key[:4] == ("git", "-C", str(self.root), "branch"):
            return CommandResult(key, 0, "feature/coordination-envelope\n", "")
        if key[:4] == ("git", "-C", str(self.root), "rev-parse"):
            return CommandResult(key, 0, "abcdef123456\n", "")
        if key[:4] == ("git", "-C", str(self.root), "status"):
            return CommandResult(key, 0, " M polylogue/coordination/envelope.py\n", "")
        if key == ("bd", "list", "--status=in_progress", "--json"):
            return CommandResult(key, 0, json.dumps(self.beads_rows or []), "")
        if key == ("bd", "hooks", "list", "--json"):
            return CommandResult(
                key,
                0,
                json.dumps(
                    {
                        "hooks": [
                            {
                                "Name": "pre-commit",
                                "Installed": True,
                                "Version": "1.0.4",
                                "IsShim": True,
                                "Outdated": False,
                            },
                            {
                                "Name": "pre-push",
                                "Installed": True,
                                "Version": "1.0.4",
                                "IsShim": True,
                                "Outdated": False,
                            },
                        ]
                    }
                ),
                "",
            )
        if key == ("bd", "gate", "list", "--json"):
            return CommandResult(key, 0, json.dumps(self.gates), "")
        if key == ("bd", "merge-slot", "check", "--json"):
            return CommandResult(
                key,
                0,
                json.dumps(self.merge_slot or {"id": "polylogue-merge-slot", "available": False, "error": "not found"}),
                "",
            )
        if key == ("ps", "-eo", "pid,ppid,comm,args", "--no-headers"):
            return CommandResult(
                key,
                0,
                "101 1 codex codex --profile full\n202 1 polylogued polylogued run --api-port 8766\n",
                "",
            )
        return CommandResult(key, 1, "", "unexpected command")


def test_coordination_envelope_uses_beads_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".beads").mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    index = archive / "index.db"
    index.touch()
    monkeypatch.setattr("polylogue.coordination.envelope.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.coordination.envelope.active_index_db_path", lambda: index)

    payload = build_coordination_envelope(
        cwd=root,
        runner=FakeRunner(
            root,
            beads_rows=[
                {
                    "id": "polylogue-s7ae.1",
                    "title": "Coordination envelope",
                    "status": "in_progress",
                    "priority": 1,
                    "assignee": "Sinity",
                    "labels": ["area:coordination"],
                    "updated_at": "2026-07-04T18:00:00Z",
                }
            ],
        ),
    )

    assert payload.work_item.source == "beads"
    assert payload.work_item.ref == "polylogue-s7ae.1"
    assert payload.work_item.confidence == 0.95
    assert payload.repo.branch == "feature/coordination-envelope"
    assert payload.beads is not None
    assert payload.beads.hooks_all_installed is True
    assert payload.beads.hooks_outdated_count == 0
    assert [hook.name for hook in payload.beads.hooks] == ["pre-commit", "pre-push"]
    assert payload.beads.open_gate_count == 0
    assert payload.beads.merge_slot is not None
    assert payload.beads.merge_slot.id == "polylogue-merge-slot"
    assert payload.beads.merge_slot.error == "not found"
    assert payload.resource_episodes
    assert any(overlap.kind == "resource-episode" and not overlap.blocking for overlap in payload.overlaps)


def test_coordination_envelope_falls_back_to_git_without_beads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    index = archive / "index.db"
    index.touch()
    monkeypatch.setattr("polylogue.coordination.envelope.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.coordination.envelope.active_index_db_path", lambda: index)

    payload = build_coordination_envelope(cwd=root, runner=FakeRunner(root, beads_rows=None))

    assert payload.work_item.source == "git"
    assert payload.work_item.ref == "feature/coordination-envelope"
    assert payload.work_item.confidence == 0.35
    assert payload.work_item.provenance.note == "no .beads workspace found"
    assert payload.beads is None


def test_coordination_envelope_reports_beads_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".beads").mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    index = archive / "index.db"
    index.touch()
    monkeypatch.setattr("polylogue.coordination.envelope.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.coordination.envelope.active_index_db_path", lambda: index)

    payload = build_coordination_envelope(
        cwd=root,
        runner=FakeRunner(
            root,
            beads_rows=[{"id": "polylogue-s7ae.1", "status": "in_progress"}],
            gates=[
                {
                    "id": "gate-1",
                    "title": "Await review",
                    "status": "open",
                    "metadata": {"gate_type": "human", "await_id": "review-1"},
                }
            ],
            merge_slot={"id": "polylogue-merge-slot", "available": True, "status": "free", "waiters": ["agent-a"]},
        ),
    )

    assert payload.beads is not None
    assert payload.beads.open_gate_count == 1
    assert payload.beads.gates[0].id == "gate-1"
    assert payload.beads.gates[0].gate_type == "human"
    assert payload.beads.gates[0].await_id == "review-1"
    assert payload.beads.merge_slot is not None
    assert payload.beads.merge_slot.available is True
    assert payload.beads.merge_slot.waiters == ("agent-a",)


def test_coordination_view_projection_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".beads").mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    index = archive / "index.db"
    index.touch()
    monkeypatch.setattr("polylogue.coordination.envelope.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.coordination.envelope.active_index_db_path", lambda: index)

    payload = build_coordination_envelope(
        view="work-item",
        cwd=root,
        runner=FakeRunner(root, beads_rows=[{"id": "polylogue-s7ae.1", "status": "in_progress"}]),
    )

    assert payload.view == "work-item"
    assert payload.peers == ()
    assert payload.resource_episodes == ()
    assert payload.overlaps == ()
    assert payload.archive is None
    assert payload.beads is not None
