"""Behavior tests for devtools workspace backlog-calibration.

The tool exists so execution plans quote measured distributions instead of
guesses; these tests pin the measurement semantics that make those numbers
trustworthy: percentile math, right-censoring honesty (survival cohort),
discovery accounting that excludes the bulk
import day, and PR latency that reads the merge train rather than guessing
implementation time.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from devtools.backlog_calibration import (
    build_report,
    main,
    summarize_days,
)

AS_OF = datetime(2026, 7, 31, tzinfo=UTC)


def _bead(
    bead_id: str,
    *,
    status: str = "closed",
    priority: int = 1,
    issue_type: str = "task",
    created: str = "2026-07-04T00:00:00Z",
    closed: str | None = "2026-07-05T00:00:00Z",
    close_reason: str | None = None,
    dependencies: list[dict[str, str]] | None = None,
    dependency_count: int = 0,
) -> dict[str, object]:
    record: dict[str, object] = {
        "id": bead_id,
        "status": status,
        "priority": priority,
        "issue_type": issue_type,
        "created_at": created,
        "dependency_count": dependency_count,
    }
    if closed is not None:
        record["closed_at"] = closed
    if close_reason is not None:
        record["close_reason"] = close_reason
    if dependencies is not None:
        record["dependencies"] = dependencies
    return record


class TestSummarizeDays:
    def test_percentiles_interpolate_and_report_max(self) -> None:
        summary = summarize_days([1.0, 2.0, 3.0, 4.0])
        assert summary["n"] == 4
        assert summary["p50_days"] == pytest.approx(2.5)
        assert summary["max_days"] == pytest.approx(4.0)

    def test_empty_population_reports_n_zero_not_a_fit(self) -> None:
        assert summarize_days([]) == {"n": 0}


class TestBuildReport:
    def test_closed_lead_splits_by_priority_and_epic_membership(self) -> None:
        beads = [
            _bead("a", priority=0, closed="2026-07-04T06:00:00Z"),
            _bead("b", priority=2, closed="2026-07-10T00:00:00Z"),
            _bead(
                "c",
                priority=2,
                closed="2026-07-08T00:00:00Z",
                dependencies=[{"type": "parent-child", "depends_on_id": "epic"}],
            ),
        ]
        report = build_report(beads, as_of=AS_OF)
        lead = report["closed_lead_days"]
        assert lead["by_priority"]["P0"]["n"] == 1
        assert lead["by_priority"]["P0"]["p50_days"] == pytest.approx(0.25)
        assert lead["by_epic_membership"]["epic-child"]["n"] == 1
        assert lead["by_epic_membership"]["standalone"]["n"] == 2

    def test_survival_counts_open_beads_against_the_cohort(self) -> None:
        # Two old beads: one closed fast, one still open. Closed-only stats
        # would report a rosy median; survival must count the open one.
        beads = [
            _bead("fast", created="2026-07-01T00:00:00Z", closed="2026-07-01T12:00:00Z"),
            _bead("open", status="open", created="2026-07-01T00:00:00Z", closed=None),
            # Too young for the cohort: must be excluded entirely.
            _bead("young", status="open", created="2026-07-30T00:00:00Z", closed=None),
        ]
        report = build_report(beads, as_of=AS_OF)
        overall = report["survival"]["overall"]
        assert overall["n"] == 2
        assert overall["closed_within_1d_pct"] == pytest.approx(50.0)
        assert overall["closed_within_14d_pct"] == pytest.approx(50.0)

    def test_discovery_excludes_the_import_day_and_reports_ratio(self) -> None:
        beads = [
            # Import-day bulk: must not count toward the discovery ratio.
            *[_bead(f"imp{i}", status="open", created="2026-07-03T00:00:00Z", closed=None) for i in range(10)],
            _bead("d1", status="open", created="2026-07-04T01:00:00Z", closed=None),
            _bead("d2", status="open", created="2026-07-04T02:00:00Z", closed=None),
            _bead("d3", created="2026-07-04T03:00:00Z", closed="2026-07-05T00:00:00Z"),
        ]
        report = build_report(beads, as_of=AS_OF)
        discovery = report["discovery"]
        assert discovery["import_day_excluded"] == "2026-07-03"
        assert discovery["post_import_created"] == 3
        assert discovery["post_import_closed"] == 1
        assert discovery["created_per_close"] == pytest.approx(3.0)

    def test_pr_latency_buckets_by_changed_files(self) -> None:
        prs = [
            {
                "number": 1,
                "createdAt": "2026-07-30T00:00:00Z",
                "mergedAt": "2026-07-30T00:06:00Z",
                "changedFiles": 2,
            },
            {
                "number": 2,
                "createdAt": "2026-07-30T00:00:00Z",
                "mergedAt": "2026-07-30T02:00:00Z",
                "changedFiles": 40,
            },
            # Unmerged/malformed rows must be skipped, not crash.
            {"number": 3, "createdAt": "2026-07-30T00:00:00Z", "mergedAt": None},
        ]
        report = build_report([], as_of=AS_OF, prs=prs)
        latency = report["pr_merge_latency"]
        assert latency["n"] == 2
        assert latency["by_changed_files_days"]["1-2"]["n"] == 1
        assert latency["by_changed_files_days"]["31+"]["n"] == 1
        # 0.1h and 2.0h -> midpoint 1.05h (1.056 after day-precision rounding)
        assert latency["overall_latency_hours"]["p50_days"] == pytest.approx(1.05, rel=0.01)


class TestCli:
    def _write_export(self, tmp_path: Path) -> Path:
        path = tmp_path / "beads.jsonl"
        rows = [
            _bead("a", close_reason="already satisfied by #2"),
            _bead("b", status="open", closed=None),
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return path

    def test_json_output_is_parseable_and_complete(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        export = self._write_export(tmp_path)
        assert main(["--input", str(export), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["population"]["total"] == 2
        assert set(payload) >= {
            "population",
            "closed_lead_days",
            "survival",
            "discovery",
        }

    def test_human_output_carries_the_censoring_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        export = self._write_export(tmp_path)
        assert main(["--input", str(export)]) == 0
        out = capsys.readouterr().out
        assert "survivorship-biased" in out

    def test_invalid_jsonl_line_fails_with_location(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "ok"}\nnot-json\n')
        with pytest.raises(SystemExit, match="bad.jsonl:2"):
            main(["--input", str(path), "--json"])
