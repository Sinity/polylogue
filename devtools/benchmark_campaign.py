"""Run and compare benchmark campaigns with durable artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark_catalog import BenchmarkCampaignEntry, build_benchmark_entries
from .execution_specs import ExecutionKind

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(".local/benchmark-campaigns")
STATUS_IGNORE_PREFIXES = (f"{ARTIFACT_DIR.as_posix()}/",)
DEFAULT_WARN_PCT = 10.0
DEFAULT_FAIL_PCT = 20.0
CAMPAIGNS = {entry.name: entry for entry in build_benchmark_entries()}


@dataclass(frozen=True)
class BenchmarkStat:
    name: str
    fullname: str
    group: str
    mean: float
    median: float
    minimum: float
    maximum: float
    stddev: float
    rounds: int
    ops: float | None


@dataclass(frozen=True)
class Regression:
    fullname: str
    baseline_mean: float
    current_mean: float
    delta_pct: float


@dataclass(frozen=True)
class CampaignResult:
    campaign: str
    description: str
    commit: str
    worktree_dirty: bool
    created_at: str
    workspace: str
    command: list[str]
    tests: list[str]
    notes: list[str]
    benchmark_count: int
    runtime_seconds: float
    exit_code: int
    machine_info: dict[str, Any]
    benchmarks: list[dict[str, Any]]
    slowest: list[dict[str, Any]]
    compare_to: str | None
    warn_pct: float
    fail_pct: float
    regressions: list[dict[str, Any]]
    worst_regression_pct: float | None
    origin: str = "authored"
    path_targets: list[str] = field(default_factory=list)
    artifact_targets: list[str] = field(default_factory=list)
    operation_targets: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _worktree_dirty() -> bool:
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    for line in status.splitlines():
        path = line[3:]
        if any(path.startswith(prefix) for prefix in STATUS_IGNORE_PREFIXES):
            continue
        if path:
            return True
    return False


def _default_artifact_path(campaign: str, suffix: str) -> Path:
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    return ROOT / ARTIFACT_DIR / f"{date}-{campaign}.{suffix}"


def _benchmark_key(entry: dict[str, Any]) -> str:
    return str(entry.get("fullname") or entry.get("fullfunc") or entry.get("name"))


def _parse_stats(raw: dict[str, Any]) -> BenchmarkStat:
    stats = raw["stats"]
    mean = float(stats["mean"])
    return BenchmarkStat(
        name=str(raw.get("name", "unknown")),
        fullname=_benchmark_key(raw),
        group=str(raw.get("group", "benchmark")),
        mean=mean,
        median=float(stats["median"]),
        minimum=float(stats["min"]),
        maximum=float(stats["max"]),
        stddev=float(stats.get("stddev", 0.0)),
        rounds=int(stats.get("rounds", 0)),
        ops=(1.0 / mean) if mean > 0 else None,
    )


def _load_campaign_result(path: Path) -> CampaignResult:
    return CampaignResult(**json.loads(path.read_text()))


def _compare_results(current: list[BenchmarkStat], baseline: list[dict[str, Any]]) -> list[Regression]:
    baseline_map = {str(item["fullname"]): item for item in baseline}
    regressions: list[Regression] = []
    for bench in current:
        previous = baseline_map.get(bench.fullname)
        if previous is None:
            continue
        baseline_mean = float(previous["mean"])
        if baseline_mean <= 0:
            continue
        delta_pct = ((bench.mean - baseline_mean) / baseline_mean) * 100.0
        regressions.append(
            Regression(
                fullname=bench.fullname,
                baseline_mean=baseline_mean,
                current_mean=bench.mean,
                delta_pct=delta_pct,
            )
        )
    regressions.sort(key=lambda item: item.delta_pct, reverse=True)
    return regressions


def _render_markdown(result: CampaignResult) -> str:
    lines = [
        f"# Benchmark Campaign: {result.campaign}",
        "",
        f"- Description: {result.description}",
        f"- Commit: `{result.commit}`",
        f"- Worktree dirty: {'yes' if result.worktree_dirty else 'no'}",
        f"- Created: `{result.created_at}`",
        f"- Runtime: `{result.runtime_seconds:.2f}s`",
        f"- Command: `{' '.join(result.command)}`",
        f"- Tests: `{', '.join(result.tests)}`",
        f"- Benchmarks: `{result.benchmark_count}`",
        f"- Warn threshold: `{result.warn_pct:.1f}%`",
        f"- Fail threshold: `{result.fail_pct:.1f}%`",
        "",
    ]
    if result.path_targets or result.artifact_targets or result.operation_targets or result.tags:
        lines.extend(["## Scenario Metadata", ""])
        lines.append(f"- Origin: `{result.origin}`")
        if result.path_targets:
            lines.append(f"- Path targets: `{', '.join(result.path_targets)}`")
        if result.artifact_targets:
            lines.append(f"- Artifact targets: `{', '.join(result.artifact_targets)}`")
        if result.operation_targets:
            lines.append(f"- Operation targets: `{', '.join(result.operation_targets)}`")
        if result.tags:
            lines.append(f"- Tags: `{', '.join(result.tags)}`")
        lines.append("")
    lines.extend(
        [
            "## Slowest Benchmarks",
            "",
            "| Benchmark | Mean (s) | Median (s) | Ops/s | Rounds |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for bench in result.slowest:
        ops = "-" if bench["ops"] is None else f"{bench['ops']:.2f}"
        lines.append(
            f"| `{bench['fullname']}` | {bench['mean']:.6f} | {bench['median']:.6f} | {ops} | {bench['rounds']} |"
        )
    if result.regressions:
        lines.extend(
            [
                "",
                "## Largest Regressions vs Baseline",
                "",
                "| Benchmark | Delta % | Baseline Mean (s) | Current Mean (s) |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for regression in result.regressions[:10]:
            lines.append(
                f"| `{regression['fullname']}` | {regression['delta_pct']:.2f}% | {regression['baseline_mean']:.6f} | {regression['current_mean']:.6f} |"
            )
    if result.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in result.notes])
    return "\n".join(lines) + "\n"


def run_campaign(
    campaign: BenchmarkCampaignEntry,
    *,
    json_out: Path | None,
    markdown_out: Path | None,
    compare_to: Path | None,
    warn_pct: float | None,
    fail_pct: float | None,
) -> CampaignResult:
    if campaign.execution is None or campaign.execution.kind is not ExecutionKind.PYTEST:
        raise ValueError(f"Benchmark campaign {campaign.name!r} must use pytest execution")

    artifact_json = json_out or _default_artifact_path(campaign.name, "json")
    artifact_md = markdown_out or _default_artifact_path(campaign.name, "md")
    artifact_json.parent.mkdir(parents=True, exist_ok=True)
    artifact_md.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"benchmark-{campaign.name}-") as tmpdir:
        raw_json = Path(tmpdir) / "pytest-benchmark.json"
        command = list(
            campaign.execution.pytest_command(
                "-q",
                "--override-ini=addopts=-ra",
                "-n",
                "0",
                "-p",
                "no:randomly",
                "--benchmark-enable",
                f"--benchmark-json={raw_json}",
            )
        )
        start = time.monotonic()
        completed = subprocess.run(command, cwd=ROOT)
        runtime_seconds = time.monotonic() - start
        if completed.returncode != 0 and (not raw_json.exists() or raw_json.stat().st_size == 0):
            raise SystemExit(completed.returncode)
        if not raw_json.exists():
            raise SystemExit(f"Benchmark run for {campaign.name} produced no JSON artifact")
        payload = json.loads(raw_json.read_text())

    benchmarks = [_parse_stats(entry) for entry in payload.get("benchmarks", [])]
    benchmarks.sort(key=lambda item: item.mean, reverse=True)
    baseline_result = _load_campaign_result(compare_to) if compare_to else None
    regressions = _compare_results(benchmarks, baseline_result.benchmarks) if baseline_result else []
    warn_threshold = campaign.warn_pct if warn_pct is None else warn_pct
    fail_threshold = campaign.fail_pct if fail_pct is None else fail_pct
    worst_regression = regressions[0].delta_pct if regressions else None

    result = CampaignResult(
        campaign=campaign.name,
        description=campaign.description,
        commit=_git_output("rev-parse", "HEAD"),
        worktree_dirty=_worktree_dirty(),
        created_at=datetime.now(UTC).isoformat(),
        workspace=str(ROOT),
        command=command,
        tests=list(campaign.tests),
        notes=list(campaign.notes),
        benchmark_count=len(benchmarks),
        runtime_seconds=runtime_seconds,
        exit_code=completed.returncode,
        machine_info=dict(payload.get("machine_info", {})),
        benchmarks=[asdict(bench) for bench in benchmarks],
        slowest=[asdict(bench) for bench in benchmarks[:10]],
        compare_to=str(compare_to) if compare_to else None,
        warn_pct=warn_threshold,
        fail_pct=fail_threshold,
        regressions=[asdict(item) for item in regressions],
        worst_regression_pct=worst_regression,
        **campaign.to_payload(),
    )

    artifact_json.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    artifact_md.write_text(_render_markdown(result))

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if worst_regression is not None and worst_regression > fail_threshold:
        raise SystemExit(
            f"Benchmark regression exceeded fail threshold: {worst_regression:.2f}% > {fail_threshold:.2f}%"
        )
    return result


def render_index() -> str:
    artifact_dir = ROOT / ARTIFACT_DIR
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for json_path in sorted(artifact_dir.glob("*.json")):
        result = _load_campaign_result(json_path)
        md_path = json_path.with_suffix(".md")
        worst = "-" if result.worst_regression_pct is None else f"{result.worst_regression_pct:.2f}%"
        rows.append(
            (
                result.created_at[:10],
                result.campaign,
                result.commit[:12],
                str(result.benchmark_count),
                f"{result.runtime_seconds:.2f}s",
                worst,
                md_path.name if md_path.exists() else "-",
            )
        )
    lines = [
        "# Benchmark Campaign Artifacts",
        "",
        "Use `devtools benchmark-campaign list` to see campaign definitions.",
        "Use `devtools benchmark-campaign run <campaign>` to record a fresh artifact.",
        "Use `devtools benchmark-campaign compare <baseline.json> <candidate.json>` to compare two artifacts.",
        "",
        "| Date | Campaign | Commit | Benchmarks | Runtime | Worst Regression | Markdown |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        date, campaign, commit, count, runtime, worst, markdown = row
        md_link = f"[{markdown}](./{markdown})" if markdown != "-" else "-"
        lines.append(f"| {date} | `{campaign}` | `{commit}` | {count} | {runtime} | {worst} | {md_link} |")
    return "\n".join(lines) + "\n"


def compare_artifacts(baseline: Path, candidate: Path, fail_pct: float) -> int:
    baseline_result = _load_campaign_result(baseline)
    candidate_result = _load_campaign_result(candidate)
    regressions = _compare_results(
        [BenchmarkStat(**entry) for entry in candidate_result.benchmarks],
        baseline_result.benchmarks,
    )
    if not regressions:
        print("No overlapping benchmarks to compare.")
        return 0
    print(f"Comparing {candidate.name} against {baseline.name}:")
    for regression in regressions[:10]:
        print(
            f"  {regression.fullname}: {regression.delta_pct:.2f}% "
            f"({regression.baseline_mean:.6f}s -> {regression.current_mean:.6f}s)"
        )
    worst = regressions[0].delta_pct
    if worst > fail_pct:
        print(f"FAIL: worst regression {worst:.2f}% exceeds {fail_pct:.2f}%")
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List benchmark campaigns")
    subparsers.add_parser("index", help="Regenerate benchmark artifact index markdown")

    run_parser = subparsers.add_parser("run", help="Run a benchmark campaign and write durable artifacts")
    run_parser.add_argument("campaign", choices=sorted(CAMPAIGNS))
    run_parser.add_argument("--json-out", type=Path)
    run_parser.add_argument("--markdown-out", type=Path)
    run_parser.add_argument("--compare-to", type=Path)
    run_parser.add_argument("--warn-pct", type=float)
    run_parser.add_argument("--fail-pct", type=float)

    compare_parser = subparsers.add_parser("compare", help="Compare two existing benchmark artifacts")
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("--fail-pct", type=float, default=DEFAULT_FAIL_PCT)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "list":
        for campaign in CAMPAIGNS.values():
            print(f"{campaign.name}: {campaign.description}")
            for test in campaign.tests:
                print(f"  - {test}")
        return 0
    if args.command == "index":
        path = ROOT / ARTIFACT_DIR / "README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_index())
        print(path)
        return 0
    if args.command == "run":
        run_campaign(
            CAMPAIGNS[args.campaign],
            json_out=args.json_out,
            markdown_out=args.markdown_out,
            compare_to=args.compare_to,
            warn_pct=args.warn_pct,
            fail_pct=args.fail_pct,
        )
        return 0
    if args.command == "compare":
        return compare_artifacts(args.baseline, args.candidate, args.fail_pct)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
