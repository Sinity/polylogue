"""Coordination envelope tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from polylogue.coordination.envelope import CommandResult, build_coordination_envelope


class FakeRunner:
    def __init__(self, root: Path, *, beads_rows: list[dict[str, object]] | None) -> None:
        self.root = root
        self.beads_rows = beads_rows

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
