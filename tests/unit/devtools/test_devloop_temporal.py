from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from devtools import devloop_temporal


def test_operating_log_events_parse_focus_and_checkpoint(tmp_path: Path) -> None:
    log = tmp_path / "OPERATING-LOG.md"
    log.write_text(
        "\n".join(
            [
                "## 2026-06-30 10:00:00 CEST — start slice",
                "## 2026-06-30 10:05:00 CEST — focus: Evidence -> Proof",
                "## not a timestamp",
            ]
        ),
        encoding="utf-8",
    )

    events, caveats = devloop_temporal.operating_log_events(log, limit=10)

    assert [event.kind for event in events] == ["checkpoint", "focus"]
    assert events[1].phase == "Proof"
    assert events[0].source_ref == f"{log}:1"
    assert caveats == ("operating_log_unparsed_headings",)


def test_structured_devloop_events_parse_jsonl(tmp_path: Path) -> None:
    event_log = tmp_path / "EVENTS.jsonl"
    event_log.write_text(
        "\n".join(
            [
                '{"schema_version": 1, "occurred_at": "2026-06-30 10:00:00 CEST", "title": "start slice"}',
                '{"schema_version": 1, "occurred_at": "2026-06-30T10:05:00+02:00", "title": "focus: Evidence -> Proof"}',
            ]
        ),
        encoding="utf-8",
    )

    events, caveats = devloop_temporal.structured_devloop_events(event_log, limit=10)

    assert [event.kind for event in events] == ["checkpoint", "focus"]
    assert events[1].phase == "Proof"
    assert events[0].source_ref == f"{event_log}:1"
    assert caveats == ()


def test_git_commit_events_reads_local_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "agent@example.test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Agent"], cwd=repo, check=True)
    (repo / "file.txt").write_text("content\n", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "test: add file"],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": "2026-06-30T10:00:00+02:00",
            "GIT_COMMITTER_DATE": "2026-06-30T10:00:00+02:00",
        },
    )

    events, caveats = devloop_temporal.git_commit_events(repo, limit=5, since=None)

    assert len(events) == 1
    assert events[0].family == "git"
    assert events[0].kind == "commit"
    assert events[0].label == "test: add file"
    assert caveats == ()


def test_build_report_uses_shared_temporal_window(tmp_path: Path) -> None:
    log = tmp_path / "OPERATING-LOG.md"
    log.write_text("## 2026-06-30 10:00:00 CEST — focus: Direction -> Evidence\n", encoding="utf-8")
    args = argparse.Namespace(
        repo=Path("/missing/repo"),
        log=log,
        event_log=tmp_path / "missing-events.jsonl",
        since="2026-06-30T00:00:00+02:00",
        until="2026-06-30T23:59:59+02:00",
        max_commits=0,
        max_log_events=10,
        out=None,
        json=True,
    )

    report = devloop_temporal.build_report(args)

    window = report["temporal_window"]
    assert isinstance(window, dict)
    assert window["event_count"] == 1
    assert window["family_counts"] == {"devloop-log": 1}
    assert report["source_counts"] == {"devloop_log_events": 1, "git_commit_events": 0}
    assert report["devloop_event_source"] == "markdown_fallback"


def test_build_report_prefers_structured_event_log(tmp_path: Path) -> None:
    log = tmp_path / "OPERATING-LOG.md"
    log.write_text("## 2026-06-30 10:00:00 CEST — markdown event\n", encoding="utf-8")
    event_log = tmp_path / "EVENTS.jsonl"
    event_log.write_text(
        '{"schema_version": 1, "occurred_at": "2026-06-30 10:05:00 CEST", "title": "structured event"}\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        repo=Path("/missing/repo"),
        log=log,
        event_log=event_log,
        since="2026-06-30T00:00:00+02:00",
        until="2026-06-30T23:59:59+02:00",
        max_commits=0,
        max_log_events=10,
        out=None,
        json=True,
    )

    report = devloop_temporal.build_report(args)

    window = report["temporal_window"]
    assert isinstance(window, dict)
    assert window["event_count"] == 1
    assert window["events"][0]["label"] == "structured event"
    assert report["devloop_event_source"] == "structured_jsonl"
