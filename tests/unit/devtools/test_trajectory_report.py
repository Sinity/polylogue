"""Behavior tests for devtools workspace trajectory-report.

The report exists so trajectory claims (accelerating vs plateauing, campaign
concentration, bead/code coupling) are computed, not narrated. These tests pin
the measurement semantics behind each visual: PR recovery from squash
subjects, week bucketing, rolling windows, burstiness statistics, data-derived
day classes, momentum verdicts, and the bulk-import-day exclusion.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from devtools.trajectory_report import (
    Commit,
    area_of,
    build_facts,
    classify_days,
    facts_json,
    gini,
    pearson,
    pr_merges,
    render,
    rolling_mean,
    top_decile_share,
    week_key,
)

NOW = dt.datetime(2026, 8, 1, 12, 0, tzinfo=dt.UTC)


def _ts(day: str, hour: int = 12) -> dt.datetime:
    return dt.datetime.fromisoformat(f"{day}T{hour:02d}:00:00+00:00")


def _commit(day: str, subject: str = "feat: x", files: tuple[str, ...] = ("polylogue/cli/a.py",)) -> Commit:
    return Commit(_ts(day), subject, files)


class TestPrMerges:
    def test_extracts_only_trailing_pr_suffix(self) -> None:
        rows = [
            (_ts("2026-07-01"), "feat: add thing (#123)"),
            (_ts("2026-07-02"), "fix: mention (#99) mid-subject not a merge"),
            (_ts("2026-07-03"), "chore(beads): bookkeeping"),
        ]
        assert pr_merges(rows) == [(_ts("2026-07-01"), 123)]


class TestBucketing:
    def test_week_key_is_iso_monday(self) -> None:
        assert week_key(dt.date(2026, 7, 30)) == "2026-07-27"  # Thursday -> Monday
        assert week_key(dt.date(2026, 7, 27)) == "2026-07-27"  # Monday is fixed point

    def test_area_of_maps_polylogue_subpackages_and_top_levels(self) -> None:
        assert area_of("polylogue/storage/repository/reads.py") == "polylogue/storage"
        assert area_of("polylogue/config.py") == "polylogue/(root)"
        assert area_of("tests/unit/devtools/test_x.py") == "tests"
        assert area_of(".beads/issues.jsonl") == "beads-sync"
        assert area_of("README.md") == "(repo root)"


class TestRollingMean:
    def test_trailing_window_and_short_prefix(self) -> None:
        days = ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"]
        counts = {"2026-07-01": 4, "2026-07-02": 0, "2026-07-03": 2, "2026-07-04": 6}
        out = rolling_mean(days, counts, window=2)
        assert out == [4.0, 2.0, 1.0, 4.0]


class TestBurstiness:
    def test_gini_zero_for_even_and_high_for_concentrated(self) -> None:
        assert gini([5, 5, 5, 5]) == 0.0
        assert gini([0, 0, 0, 20]) > 0.7

    def test_top_decile_share_names_the_busiest_days(self) -> None:
        values = [1] * 9 + [91]
        assert top_decile_share(values) == 0.91


class TestPearson:
    def test_perfect_and_degenerate(self) -> None:
        r = pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        assert r is not None and abs(r - 1.0) < 1e-9
        assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
        assert pearson([1.0], [1.0]) is None


class TestClassifyDays:
    def test_thresholds_come_from_nonzero_distribution(self) -> None:
        days = [f"2026-07-{d:02d}" for d in range(1, 13)]
        merges = dict(zip(days, [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 60], strict=True))
        classes, threshold = classify_days(merges, days)
        assert classes["2026-07-01"] == "quiet"
        assert classes["2026-07-03"] == "organic"
        assert classes["2026-07-12"] == "campaign"
        assert threshold >= 4


class TestBuildFacts:
    def _corpus(self) -> tuple[list[Commit], list[tuple[dt.datetime, str]], list[dict[str, object]]]:
        commits: list[Commit] = []
        first_parent: list[tuple[dt.datetime, str]] = []
        pr = 1
        # 70 days of activity: quiet first half, busy second half (accelerating).
        for i in range(70):
            day = (dt.date(2026, 5, 24) + dt.timedelta(days=i)).isoformat()
            per_day = 2 if i < 42 else 8
            for j in range(per_day):
                commits.append(_commit(day, f"feat: c{i}-{j} (#{pr})", ("polylogue/storage/x.py", "tests/t.py")))
                first_parent.append((_ts(day, hour=j % 24), f"feat: c{i}-{j} (#{pr})"))
                pr += 1
        beads: list[dict[str, object]] = [
            {
                "id": f"polylogue-{i}",
                "created_at": "2026-07-03T10:00:00Z",
                "closed_at": ("2026-07-20T10:00:00Z" if i % 2 == 0 else None),
                "status": "closed" if i % 2 == 0 else "open",
            }
            for i in range(10)
        ]
        return commits, first_parent, beads

    def test_momentum_verdict_accelerating_on_rising_pace(self) -> None:
        commits, first_parent, beads = self._corpus()
        facts = build_facts(commits, first_parent, beads, NOW)
        assert facts.momentum_verdict == "accelerating"
        assert facts.momentum_ratio is not None and facts.momentum_ratio > 1.15
        assert facts.recent_prs > facts.prior_prs > 0

    def test_import_day_excluded_from_coupling(self) -> None:
        commits, first_parent, beads = self._corpus()
        facts = build_facts(commits, first_parent, beads, NOW)
        assert facts.bead_import_day == "2026-07-03"
        # Coupling days span bead-era days minus the import day itself.
        assert facts.coupling_n == len([d for d in facts.days if d >= "2026-07-03"]) - 1

    def test_open_bead_reconstruction_reaches_current_open_count(self) -> None:
        commits, first_parent, beads = self._corpus()
        facts = build_facts(commits, first_parent, beads, NOW)
        assert facts.open_beads_by_day[-1][1] == 5  # 10 created, 5 closed

    def test_empty_history_yields_empty_facts(self) -> None:
        facts = build_facts([], [], [], NOW)
        assert facts.commits_total == 0
        assert facts.momentum_verdict == "insufficient data"


class TestRender:
    def test_report_is_self_contained_and_fully_interpolated(self) -> None:
        commits, first_parent, beads = TestBuildFacts()._corpus()
        facts = build_facts(commits, first_parent, beads, NOW)
        page = render(facts, Path("issues.jsonl"), NOW)
        for marker in (
            'id="trajectory"',
            'id="momentum"',
            'id="rhythm"',
            'id="areas"',
            'id="coupling"',
            'id="uncertainties"',
            "<svg",
            "ev-measured",
        ):
            assert marker in page
        # Every PROSE placeholder must have been interpolated.
        assert "{momentum_ratio}" not in page
        assert "{coupling_r}" not in page
        # Self-contained: no external requests.
        assert "http://" not in page
        assert 'src="' not in page

    def test_facts_json_round_trips_key_measurements(self) -> None:
        commits, first_parent, beads = TestBuildFacts()._corpus()
        facts = build_facts(commits, first_parent, beads, NOW)
        payload = facts_json(facts)
        assert payload["momentum"]["verdict"] == "accelerating"
        assert payload["bead_era"]["import_day"] == "2026-07-03"
        assert sum(week["commits"] for week in payload["weeks"]) == facts.commits_total
