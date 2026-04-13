from __future__ import annotations

import json
from pathlib import Path

from devtools.benchmark_campaign import (
    BENCHMARK_SCENARIOS,
    BenchmarkScenario,
    BenchmarkStat,
    CampaignResult,
    Regression,
    _compare_results,
    compare_artifacts,
    compile_benchmark_scenarios,
    render_index,
)


def test_compare_results_orders_regressions_by_worst_delta() -> None:
    current = [
        BenchmarkStat("a", "bench.a", "group", 2.0, 2.0, 1.0, 3.0, 0.1, 10, 0.5),
        BenchmarkStat("b", "bench.b", "group", 1.2, 1.2, 1.0, 1.3, 0.1, 10, 0.8),
    ]
    baseline = [
        {"fullname": "bench.a", "mean": 1.0},
        {"fullname": "bench.b", "mean": 1.0},
    ]

    regressions = _compare_results(current, baseline)

    assert [item.fullname for item in regressions] == ["bench.a", "bench.b"]
    assert regressions[0] == Regression("bench.a", 1.0, 2.0, 100.0)


def test_compare_artifacts_fails_when_threshold_is_exceeded(tmp_path: Path) -> None:
    baseline = CampaignResult(
        campaign="search-filters",
        description="baseline",
        commit="a" * 40,
        worktree_dirty=False,
        created_at="2026-04-11T00:00:00+00:00",
        workspace="/tmp/workspace",
        command=["pytest"],
        tests=["tests/benchmarks/test_search_filters.py"],
        notes=[],
        benchmark_count=1,
        runtime_seconds=1.0,
        exit_code=0,
        machine_info={},
        benchmarks=[{"fullname": "bench.a", "mean": 1.0}],
        slowest=[],
        compare_to=None,
        warn_pct=10.0,
        fail_pct=20.0,
        regressions=[],
        worst_regression_pct=None,
    )
    candidate = CampaignResult(
        campaign="search-filters",
        description="candidate",
        commit="b" * 40,
        worktree_dirty=False,
        created_at="2026-04-11T00:00:00+00:00",
        workspace="/tmp/workspace",
        command=["pytest"],
        tests=["tests/benchmarks/test_search_filters.py"],
        notes=[],
        benchmark_count=1,
        runtime_seconds=1.0,
        exit_code=0,
        machine_info={},
        benchmarks=[
            {
                "name": "bench.a",
                "fullname": "bench.a",
                "group": "group",
                "mean": 1.5,
                "median": 1.5,
                "minimum": 1.4,
                "maximum": 1.6,
                "stddev": 0.05,
                "rounds": 5,
                "ops": 0.66,
            }
        ],
        slowest=[],
        compare_to=None,
        warn_pct=10.0,
        fail_pct=20.0,
        regressions=[],
        worst_regression_pct=None,
    )
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline.__dict__), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate.__dict__), encoding="utf-8")

    assert compare_artifacts(baseline_path, candidate_path, 20.0) == 1


def test_render_index_lists_saved_artifacts(tmp_path: Path, monkeypatch) -> None:
    artifact_dir = tmp_path / ".local" / "benchmark-campaigns"
    artifact_dir.mkdir(parents=True)
    result = CampaignResult(
        campaign="search-filters",
        description="candidate",
        commit="b" * 40,
        worktree_dirty=False,
        created_at="2026-04-11T00:00:00+00:00",
        workspace="/tmp/workspace",
        command=["pytest"],
        tests=["tests/benchmarks/test_search_filters.py"],
        notes=[],
        benchmark_count=1,
        runtime_seconds=1.0,
        exit_code=0,
        machine_info={},
        benchmarks=[],
        slowest=[],
        compare_to=None,
        warn_pct=10.0,
        fail_pct=20.0,
        regressions=[],
        worst_regression_pct=None,
    )
    (artifact_dir / "2026-04-11-search-filters.json").write_text(json.dumps(result.__dict__), encoding="utf-8")
    (artifact_dir / "2026-04-11-search-filters.md").write_text("# report\n", encoding="utf-8")
    monkeypatch.setattr("devtools.benchmark_campaign.ROOT", tmp_path)

    rendered = render_index()

    assert "# Benchmark Campaign Artifacts" in rendered
    assert "`search-filters`" in rendered
    assert "[2026-04-11-search-filters.md](./2026-04-11-search-filters.md)" in rendered


def test_benchmark_scenario_compiles_to_campaign() -> None:
    scenario = BenchmarkScenario(
        scenario_id="action-events",
        description="action-event repair benchmark",
        tests=("tests/benchmarks/test_action_events.py",),
        notes=("Tracks action-event repair throughput.",),
        origin="generated.action-events",
        artifact_targets=("action_event_rows", "action_event_fts"),
        operation_targets=("benchmark.repair.action-events",),
        tags=("benchmark", "action-events"),
    )

    campaign = scenario.compile()

    assert campaign.name == "action-events"
    assert campaign.description == "action-event repair benchmark"
    assert campaign.tests == ("tests/benchmarks/test_action_events.py",)
    assert campaign.notes == ("Tracks action-event repair throughput.",)
    assert campaign.origin == "generated.action-events"
    assert campaign.artifact_targets == ("action_event_rows", "action_event_fts")
    assert campaign.operation_targets == ("benchmark.repair.action-events",)
    assert campaign.tags == ("benchmark", "action-events")


def test_compile_benchmark_scenarios_indexes_by_id() -> None:
    campaigns = compile_benchmark_scenarios(BENCHMARK_SCENARIOS)

    assert set(campaigns) == {"search-filters", "storage", "pipeline"}
    assert campaigns["search-filters"].tests == ("tests/benchmarks/test_search_filters.py",)
