"""Tests for the per-file acquisition decision log + unclaimed-file sweep.

Covers the observability gap this module closes: a real structured log trail
per file the acquisition path considers (``file_acquisition_decision``), and
an explicit sweep that logs files under a watched root that no detector
claimed (``file_acquisition_unclaimed``) while never descending into a
``.git`` directory anywhere under that root.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from polylogue.sources.live import acquisition_log
from polylogue.sources.live.acquisition_log import (
    AcquisitionStageTimings,
    UnclaimedSweepResult,
    default_file_claim_check,
    iter_files_excluding_git,
    log_file_acquisition_decision,
    log_unclaimed_file,
    sweep_unclaimed_files,
)


def test_iter_files_excluding_git_never_descends_into_git_directory(tmp_path: Path) -> None:
    """A ``.git`` directory anywhere under root -- not only at the top -- is pruned
    before ``os.walk`` descends into it, so a file inside it is never yielded."""
    root = tmp_path / "claude-projects"
    root.mkdir()
    (root / "session-a.jsonl").write_text('{"sessionId": "a"}\n')

    nested = root / "some-project"
    nested.mkdir()
    (nested / "session-b.jsonl").write_text('{"sessionId": "b"}\n')

    git_dir = root / ".git"
    (git_dir / "objects").mkdir(parents=True)
    (git_dir / "config").write_text("[core]\n")
    (git_dir / "objects" / "deadbeef").write_text("not a session")

    # A .git directory nested deeper than the top level must also be pruned.
    nested_git = nested / ".git"
    nested_git.mkdir()
    (nested_git / "HEAD").write_text("ref: refs/heads/master\n")

    found = {str(p.relative_to(root)) for p in iter_files_excluding_git(root)}

    assert found == {"session-a.jsonl", str(Path("some-project") / "session-b.jsonl")}
    assert not any(".git" in part for path in found for part in Path(path).parts)


def test_iter_files_excluding_git_missing_root_yields_nothing(tmp_path: Path) -> None:
    assert list(iter_files_excluding_git(tmp_path / "does-not-exist")) == []


def test_iter_files_excluding_git_single_file_root(tmp_path: Path) -> None:
    single = tmp_path / "solo.jsonl"
    single.write_text("{}")
    assert list(iter_files_excluding_git(single)) == [single]


def test_sweep_unclaimed_files_logs_unrecognized_file_and_skips_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "watched-root"
    root.mkdir()
    unclaimed_file = root / "mystery.dat"
    unclaimed_file.write_bytes(b"not a claude session, not anything recognized\x00\x01")
    claimed_file = root / "known.jsonl"
    claimed_file.write_text('{"sessionId": "abc", "uuid": "1"}\n')

    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n")

    logged: list[dict[str, object]] = []
    mock_logger = MagicMock()
    mock_logger.warning.side_effect = lambda event, **kw: logged.append({"event": event, **kw})
    monkeypatch.setattr(acquisition_log, "logger", mock_logger)

    def is_claimed(path: Path) -> tuple[bool, str]:
        if path.name == "known.jsonl":
            return True, "matched"
        return False, "no detector matched (test stub)"

    result = sweep_unclaimed_files(root, source_name="claude-code", is_claimed=is_claimed)

    assert isinstance(result, UnclaimedSweepResult)
    assert result.scanned == 2  # .git/config must never be counted
    assert result.unclaimed == (unclaimed_file,)
    assert len(logged) == 1
    assert logged[0]["event"] == "file_acquisition_unclaimed"
    assert logged[0]["path"] == str(unclaimed_file)
    assert logged[0]["reason"] == "no detector matched (test stub)"
    assert logged[0]["source_name"] == "claude-code"
    assert logged[0]["size"] == unclaimed_file.stat().st_size


def test_default_file_claim_check_refuses_unrecognized_shape(tmp_path: Path) -> None:
    unrecognized = tmp_path / "unrecognized.json"
    unrecognized.write_text('{"totally": "unrelated", "shape": true}')

    claimed, reason = default_file_claim_check(unrecognized)

    assert claimed is False
    assert "no detector matched" in reason


def test_default_file_claim_check_accepts_recognized_shape(tmp_path: Path) -> None:
    claude_code_file = tmp_path / "session.jsonl"
    claude_code_file.write_text('{"sessionId": "abc-123", "uuid": "u1", "type": "user", "cwd": "/tmp"}\n')

    claimed, reason = default_file_claim_check(claude_code_file)

    assert claimed is True
    assert reason


def test_default_file_claim_check_reports_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text("")

    claimed, reason = default_file_claim_check(empty)

    assert claimed is False
    assert reason == "empty file"


def test_log_file_acquisition_decision_emits_evidence_and_timings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    mock_logger = MagicMock()
    mock_logger.info.side_effect = lambda event, **kw: captured.update({"event": event, **kw})
    monkeypatch.setattr(acquisition_log, "logger", mock_logger)

    timings = AcquisitionStageTimings()
    with timings.stage("detect"):
        pass

    log_file_acquisition_decision(
        path="/watched/root/a.jsonl",
        size=123,
        mtime=1700000000.0,
        origin="claude-code-session",
        evidence="claude.looks_like_code (envelope marker, #3428)",
        stage_timings=timings,
        source_name="claude-code",
    )

    assert captured["event"] == "file_acquisition_decision"
    assert captured["path"] == "/watched/root/a.jsonl"
    assert captured["origin"] == "claude-code-session"
    assert captured["evidence"] == "claude.looks_like_code (envelope marker, #3428)"
    stage_timings_ms = captured["stage_timings_ms"]
    assert isinstance(stage_timings_ms, dict)
    assert "detect" in stage_timings_ms
    assert captured["source_name"] == "claude-code"


def test_log_file_acquisition_decision_normalizes_unrecognized_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    mock_logger = MagicMock()
    mock_logger.info.side_effect = lambda event, **kw: captured.update({"event": event, **kw})
    monkeypatch.setattr(acquisition_log, "logger", mock_logger)

    for origin in (None, "UNKNOWN", "unknown"):
        log_file_acquisition_decision(
            path="/watched/root/b.dat",
            size=1,
            mtime=None,
            origin=origin,
            evidence="no detector matched (record)",
        )
        assert captured["origin"] == "UNRECOGNIZED"


def test_log_unclaimed_file_emits_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    mock_logger = MagicMock()
    mock_logger.warning.side_effect = lambda event, **kw: captured.update({"event": event, **kw})
    monkeypatch.setattr(acquisition_log, "logger", mock_logger)

    log_unclaimed_file(
        path="/watched/root/c.dat",
        size=99,
        mtime=None,
        reason="suffix not in watched set",
        source_name="inbox",
    )

    assert captured["event"] == "file_acquisition_unclaimed"
    assert captured["reason"] == "suffix not in watched set"
    assert captured["source_name"] == "inbox"
