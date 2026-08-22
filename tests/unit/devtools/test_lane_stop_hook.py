"""Tests for the automatic SubagentStop lane lifecycle hook."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from devtools.lane_stop_hook import process_stop


def _git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True)


def _lane(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    lane = repo / ".claude" / "worktrees" / "agent-test"
    lane.parent.mkdir(parents=True)
    _git(repo, "worktree", "add", "-b", "worktree-agent-test", str(lane), "master")
    return repo, lane


def _payload(tmp_path: Path, lane: Path) -> dict[str, str]:
    transcript = tmp_path / "agent.jsonl"
    transcript.write_text(json.dumps({"cwd": str(lane), "type": "assistant"}) + "\n")
    return {"transcript_path": str(transcript), "session_id": "test"}


def test_stop_hook_packages_clean_lane_and_releases_lock(tmp_path: Path) -> None:
    repo, lane = _lane(tmp_path)
    (lane / "change.txt").write_text("change\n")
    _git(lane, "add", "change.txt")
    _git(lane, "commit", "-m", "change")
    _git(repo, "worktree", "lock", "--reason", "active test lane", str(lane))

    response = process_stop(_payload(tmp_path, lane))

    common = Path(_git(lane, "rev-parse", "--git-common-dir").stdout.strip())
    handoff = common / "polylogue" / "lane-handoffs" / "worktree-agent-test.json"
    payload = json.loads(handoff.read_text())
    assert payload["status"] == "ready-for-assimilation"
    assert payload["commits"] == [_git(lane, "rev-parse", "HEAD").stdout.strip()]
    assert payload["changed_paths"] == ["change.txt"]
    assert payload["unlocked"] is True
    assert "completion lock released" in str(response["systemMessage"])


def test_stop_hook_preserves_dirty_lane_lock_and_records_blocker(tmp_path: Path) -> None:
    repo, lane = _lane(tmp_path)
    (lane / "dirty.txt").write_text("dirty\n")
    _git(repo, "worktree", "lock", "--reason", "active test lane", str(lane))

    response = process_stop(_payload(tmp_path, lane))

    common = Path(_git(lane, "rev-parse", "--git-common-dir").stdout.strip())
    handoff = common / "polylogue" / "lane-handoffs" / "worktree-agent-test.json"
    payload = json.loads(handoff.read_text())
    assert payload["status"] == "blocked"
    assert payload["unlocked"] is False
    porcelain = _git(repo, "worktree", "list", "--porcelain").stdout
    lane_block = next(block for block in porcelain.split("\n\n") if f"worktree {lane}" in block)
    assert "locked active test lane" in lane_block
    assert "blocked handoff" in str(response["systemMessage"])


def test_stop_hook_ignores_non_lane_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "agent.jsonl"
    transcript.write_text(json.dumps({"cwd": str(tmp_path)}) + "\n")

    assert process_stop({"transcript_path": str(transcript)}) == {"suppressOutput": True}
