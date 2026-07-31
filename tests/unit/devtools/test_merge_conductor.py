from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from devtools import merge_conductor


def test_classify_file_beads_jsonl() -> None:
    assert merge_conductor.classify_file(".beads/issues.jsonl") == "beads-jsonl"


def test_classify_file_generated_surface() -> None:
    assert merge_conductor.classify_file("docs/plans/topology-target.yaml") == "generated-surface"
    assert merge_conductor.classify_file("docs/topology-status.md") == "generated-surface"
    assert merge_conductor.classify_file("docs/generated/mcp-tool-index.md") == "generated-surface"


def test_classify_file_schema_migration() -> None:
    assert merge_conductor.classify_file("polylogue/storage/sqlite/migrations/source/008_add_col.sql") == (
        "schema-migration"
    )
    assert merge_conductor.classify_file("polylogue/storage/sqlite/lifecycle.py") == "schema-migration"


def test_classify_file_hooks_config() -> None:
    assert merge_conductor.classify_file(".claude/settings.json") == "hooks-config"
    assert merge_conductor.classify_file(".githooks/pre-commit") == "hooks-config"


def test_classify_file_other_defaults_to_escalate_bucket() -> None:
    assert merge_conductor.classify_file("polylogue/daemon/convergence.py") == "other"


def test_auto_resolvable_classes_are_beads_and_generated_only() -> None:
    assert {"beads-jsonl", "generated-surface"} == merge_conductor.AUTO_RESOLVABLE_CLASSES


def _fake_gh_run(pr_view_payload: dict[str, object], diff_files: list[str]) -> object:
    def _run(cmd: list[str], **kwargs: object) -> MagicMock:
        if cmd[:2] == ["gh", "pr"] and cmd[2] == "view":
            return MagicMock(returncode=0, stdout=json.dumps(pr_view_payload), stderr="")
        if cmd[:2] == ["gh", "pr"] and cmd[2] == "diff":
            return MagicMock(returncode=0, stdout="\n".join(diff_files) + "\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    return _run


def test_fetch_pr_report_clean_when_mergeable_and_no_escalate_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_gh_run(
            {
                "number": 1,
                "title": "t",
                "headRefName": "feature/x",
                "mergeable": "MERGEABLE",
                "mergeStateStatus": "CLEAN",
                "statusCheckRollup": [{"state": "SUCCESS"}],
            },
            [".beads/issues.jsonl"],
        ),
    )
    report = merge_conductor._fetch_pr_report(1)
    assert report.ok
    assert report.verdict == "CLEAN"
    assert report.escalation_files == []


def test_fetch_pr_report_auto_resolvable_when_conflicting_but_mechanical_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_gh_run(
            {
                "number": 2,
                "title": "t",
                "headRefName": "feature/y",
                "mergeable": "CONFLICTING",
                "mergeStateStatus": "DIRTY",
                "statusCheckRollup": [],
            },
            [".beads/issues.jsonl", "docs/cli-reference.md"],
        ),
    )
    report = merge_conductor._fetch_pr_report(2)
    assert report.ok
    assert report.verdict == "AUTO-RESOLVABLE"
    assert report.escalation_files == []


def test_fetch_pr_report_escalates_on_schema_migration_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_gh_run(
            {
                "number": 3,
                "title": "t",
                "headRefName": "feature/z",
                "mergeable": "CONFLICTING",
                "mergeStateStatus": "DIRTY",
                "statusCheckRollup": [],
            },
            [".beads/issues.jsonl", "polylogue/storage/sqlite/lifecycle.py"],
        ),
    )
    report = merge_conductor._fetch_pr_report(3)
    assert report.ok
    assert report.verdict == "ESCALATE"
    assert "polylogue/storage/sqlite/lifecycle.py" in report.escalation_files


def test_fetch_pr_report_degrades_on_gh_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: MagicMock(returncode=1, stdout="", stderr="gh: authentication required"),
    )
    report = merge_conductor._fetch_pr_report(4)
    assert not report.ok
    assert "authentication" in report.error


def test_find_contention_flags_shared_schema_migration_file() -> None:
    reports = [
        merge_conductor.PRReport(
            number=1,
            ok=True,
            verdict="ESCALATE",
            files_by_class={"schema-migration": ["polylogue/storage/sqlite/lifecycle.py"]},
        ),
        merge_conductor.PRReport(
            number=2,
            ok=True,
            verdict="ESCALATE",
            files_by_class={"schema-migration": ["polylogue/storage/sqlite/lifecycle.py"]},
        ),
        merge_conductor.PRReport(
            number=3,
            ok=True,
            verdict="CLEAN",
            files_by_class={},
        ),
    ]
    contention = merge_conductor._find_contention(reports)
    assert len(contention) == 1
    assert contention[0]["prs"] == [1, 2]
    assert contention[0]["file"] == "polylogue/storage/sqlite/lifecycle.py"


def test_find_contention_ignores_failed_reports() -> None:
    reports = [
        merge_conductor.PRReport(number=1, ok=False, error="boom"),
        merge_conductor.PRReport(number=2, ok=True, verdict="CLEAN"),
    ]
    assert merge_conductor._find_contention(reports) == []


def test_main_dry_run_never_touches_execute_path(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"execute": False}

    def _fake_execute(report: merge_conductor.PRReport, repo_root: object) -> str:
        called["execute"] = True
        return "should not run"

    monkeypatch.setattr(merge_conductor, "_execute_auto_resolve", _fake_execute)
    monkeypatch.setattr(
        merge_conductor,
        "_fetch_pr_report",
        lambda pr: merge_conductor.PRReport(number=pr, ok=True, verdict="AUTO-RESOLVABLE"),
    )

    rc = merge_conductor.main(["--pr", "1", "--json"])

    assert rc == 0
    assert called["execute"] is False


def test_main_execute_only_touches_auto_resolvable_prs(monkeypatch: pytest.MonkeyPatch) -> None:
    executed_numbers: list[int] = []

    def _fake_execute(report: merge_conductor.PRReport, repo_root: object) -> str:
        executed_numbers.append(report.number)
        return "RESOLVED and pushed"

    reports_by_pr = {
        1: merge_conductor.PRReport(number=1, ok=True, verdict="AUTO-RESOLVABLE"),
        2: merge_conductor.PRReport(number=2, ok=True, verdict="ESCALATE", escalation_files=["x.py"]),
    }
    monkeypatch.setattr(merge_conductor, "_execute_auto_resolve", _fake_execute)
    monkeypatch.setattr(merge_conductor, "_fetch_pr_report", lambda pr: reports_by_pr[pr])
    monkeypatch.setattr(merge_conductor, "_repo_root", lambda: object())

    rc = merge_conductor.main(["--pr", "1", "--pr", "2", "--execute", "--json"])

    assert rc == 0
    assert executed_numbers == [1]
