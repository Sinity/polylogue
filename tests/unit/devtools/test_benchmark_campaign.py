from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Literal

from pytest import MonkeyPatch

from devtools.benchmark_campaign import (
    BenchmarkStat,
    CampaignResult,
    Regression,
    _compare_results,
    compare_artifacts,
    render_index,
    run_campaign,
)
from devtools.benchmark_catalog import BenchmarkCampaignEntry
from devtools.benchmark_scenario_catalog import (
    BENCHMARK_SCENARIO_INDEX,
    BENCHMARK_SCENARIOS,
    compile_benchmark_campaigns,
)
from polylogue.scenarios import (
    AssertionSpec,
    ExecutionKind,
    ExecutionResult,
    ExecutionSpec,
    ScenarioProjectionSourceKind,
    pytest_execution,
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


def test_render_index_lists_saved_artifacts(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
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


def test_campaign_result_round_trips_path_targets_from_artifact(tmp_path: Path) -> None:
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
        origin="generated.search-filters",
        path_targets=["search-filter-loop"],
        artifact_targets=["conversation_query_results"],
        operation_targets=["query-conversations", "benchmark.query.search-filters"],
        tags=["benchmark", "search"],
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(result.__dict__), encoding="utf-8")

    loaded = CampaignResult(**json.loads(artifact.read_text(encoding="utf-8")))

    assert loaded.path_targets == ["search-filter-loop"]
    assert loaded.origin == "generated.search-filters"


def test_benchmark_entry_exposes_tests_from_execution() -> None:
    scenario = BenchmarkCampaignEntry(
        name="action-events",
        description="action-event repair benchmark",
        execution=pytest_execution("tests/benchmarks/test_action_events.py"),
        assertion=AssertionSpec(benchmark_warn_pct=10.0, benchmark_fail_pct=20.0),
        notes=("Tracks action-event repair throughput.",),
        origin="generated.action-events",
        artifact_targets=("action_event_rows", "action_event_fts"),
        operation_targets=("benchmark.repair.action-events",),
        tags=("benchmark", "action-events"),
    )

    assert scenario.tests == ("tests/benchmarks/test_action_events.py",)
    assert scenario.warn_pct == 10.0
    assert scenario.fail_pct == 20.0
    assert scenario.origin == "generated.action-events"
    assert scenario.artifact_targets == ("action_event_rows", "action_event_fts")
    assert scenario.operation_targets == ("benchmark.repair.action-events",)
    assert scenario.tags == ("benchmark", "action-events")


def test_compile_benchmark_campaigns_indexes_by_name() -> None:
    campaigns = compile_benchmark_campaigns(BENCHMARK_SCENARIOS)

    assert set(campaigns) == {"search-filters", "storage", "pipeline"}
    assert campaigns["search-filters"].tests == ("tests/benchmarks/test_search_filters.py",)


def test_benchmark_scenario_index_tracks_authored_catalog() -> None:
    assert set(BENCHMARK_SCENARIO_INDEX) == {"search-filters", "storage", "pipeline"}


def test_benchmark_entry_compiles_its_own_projection_entry() -> None:
    campaign = BenchmarkCampaignEntry(
        name="startup-readiness",
        description="startup readiness benchmark",
        origin="authored.synthetic-benchmark",
        artifact_targets=("message_fts", "archive_readiness"),
        operation_targets=("project-archive-readiness", "readiness.startup.synthetic"),
        tags=("benchmark", "synthetic", "readiness"),
        projection_kind=ScenarioProjectionSourceKind.SYNTHETIC_BENCHMARK,
    )

    projection = campaign.to_projection_entry()

    assert projection.source_kind is ScenarioProjectionSourceKind.SYNTHETIC_BENCHMARK
    assert projection.name == "startup-readiness"
    assert projection.description == "startup readiness benchmark"
    assert projection.artifact_targets == ("message_fts", "archive_readiness")
    assert projection.operation_targets == ("project-archive-readiness", "readiness.startup.synthetic")


def test_run_campaign_executes_authored_pytest_through_shared_runtime(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    class _TmpDir:
        def __init__(self, path: Path) -> None:
            self.path = path

        def __enter__(self) -> str:
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> Literal[False]:
            return False

    captured_execution: ExecutionSpec | None = None
    captured_cwd: Path | None = None
    artifact_json = tmp_path / "candidate.json"
    artifact_md = tmp_path / "candidate.md"
    benchmark_dir = tmp_path / "bench-tmp"

    monkeypatch.setattr("devtools.benchmark_campaign.ROOT", tmp_path)
    monkeypatch.setattr("devtools.benchmark_campaign._git_output", lambda *_args: "a" * 40)
    monkeypatch.setattr("devtools.benchmark_campaign._worktree_dirty", lambda: False)
    monkeypatch.setattr(
        "devtools.benchmark_campaign.tempfile.TemporaryDirectory",
        lambda prefix: _TmpDir(benchmark_dir),
    )

    def fake_run_execution(execution: ExecutionSpec, *, cwd: Path | None) -> ExecutionResult:
        nonlocal captured_execution, captured_cwd
        captured_execution = execution
        captured_cwd = cwd
        benchmark_json_flag = next(
            target for target in execution.pytest_targets if target.startswith("--benchmark-json=")
        )
        Path(benchmark_json_flag.split("=", 1)[1]).write_text(
            json.dumps(
                {
                    "machine_info": {},
                    "benchmarks": [
                        {
                            "name": "bench.a",
                            "fullname": "bench.a",
                            "group": "group",
                            "stats": {
                                "mean": 1.0,
                                "median": 1.0,
                                "min": 0.9,
                                "max": 1.1,
                                "stddev": 0.05,
                                "rounds": 5,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return ExecutionResult(
            execution=execution,
            command=("pytest", *execution.pytest_targets),
            exit_code=0,
        )

    monkeypatch.setattr("devtools.benchmark_campaign.run_execution", fake_run_execution)

    campaign = BenchmarkCampaignEntry(
        name="search-filters",
        description="FTS benchmark",
        execution=pytest_execution("tests/benchmarks/test_search_filters.py"),
        assertion=AssertionSpec(benchmark_warn_pct=10.0, benchmark_fail_pct=20.0),
        notes=("Canonical domain.",),
    )

    result = run_campaign(
        campaign,
        json_out=artifact_json,
        markdown_out=artifact_md,
        compare_to=None,
        warn_pct=None,
        fail_pct=None,
    )

    assert captured_cwd == tmp_path
    assert captured_execution is not None
    assert captured_execution.kind is ExecutionKind.PYTEST
    assert "--benchmark-enable" in captured_execution.pytest_targets
    assert "tests/benchmarks/test_search_filters.py" in captured_execution.pytest_targets
    assert result.exit_code == 0
    assert result.warn_pct == 10.0
    assert result.fail_pct == 20.0
    assert result.command[0] == "pytest"
    assert artifact_json.exists()
    assert artifact_md.exists()
