"""SubagentStop hook for automatic lane handoff and lifecycle release."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from devtools.lane_finish import LaneHandoff, finish_lane
from devtools.verify_worktree import _git


def _transcript_cwd(transcript: Path) -> Path | None:
    try:
        lines = transcript.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        cwd = row.get("cwd") if isinstance(row, dict) else None
        if isinstance(cwd, str) and cwd:
            return Path(cwd).expanduser().resolve()
    return None


def _handoff_dir(worktree: Path) -> Path | None:
    common = _git(worktree, "rev-parse", "--git-common-dir")
    if common.returncode != 0:
        return None
    path = Path(common.stdout.strip())
    if not path.is_absolute():
        path = (worktree / path).resolve()
    return path / "polylogue" / "lane-handoffs"


def _write_handoff(worktree: Path, report: LaneHandoff) -> Path | None:
    if report.branch is None:
        return None
    directory = _handoff_dir(worktree)
    if directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{report.branch}.json"
    temporary = destination.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(json.dumps({**asdict(report), "ok": report.status != "blocked"}, indent=2) + "\n")
    temporary.replace(destination)
    return destination


def process_stop(payload: dict[str, Any]) -> dict[str, object]:
    transcript_value = payload.get("transcript_path")
    if not isinstance(transcript_value, str) or not transcript_value:
        return {"suppressOutput": True}
    worktree = _transcript_cwd(Path(transcript_value))
    if worktree is None or ".claude/worktrees/agent-" not in str(worktree):
        return {"suppressOutput": True}

    branch_probe = _git(worktree, "branch", "--show-current")
    branch = branch_probe.stdout.strip() if branch_probe.returncode == 0 else ""
    if not branch.startswith("worktree-agent-"):
        return {"suppressOutput": True}

    report = finish_lane(worktree)
    destination = _write_handoff(worktree, report)
    if destination is None:
        return {
            "systemMessage": f"Lane {branch} stopped, but its handoff could not be persisted; worktree lock was preserved."
        }
    if report.status == "blocked":
        return {"systemMessage": f"Lane {branch} stopped with a blocked handoff at {destination}: {report.error}"}
    return {
        "systemMessage": (
            f"Lane {branch} handoff recorded at {destination} "
            f"({len(report.commits)} commit(s), {len(report.changed_paths)} changed path(s)); completion lock released."
        )
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            payload = {}
        response = process_stop(payload)
    except Exception as exc:  # hook failures must not strand a completed agent
        response = {"systemMessage": f"Lane completion hook failed safely; worktree lock was preserved: {exc}"}
    print(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
