"""Per-package test-suite economics: coverage vs fix-density vs test cost (polylogue-9e5.11).

Background
----------

The repo has ~248k test lines against ~229k product lines by raw line count,
yet a real production defect (embedding staleness) shipped untested despite
that volume. That is a symptom, not a diagnosis: test *volume* says nothing
about whether tests concentrate on risk-bearing substrate or on
mechanically-easy-to-cover surface. This report builds the map that lets a
human or agent tell those apart, per top-level ``polylogue/`` package:

1. **Coverage percent** -- from a ``coverage.py`` JSON report (statement
   coverage; the repo's own ratchet in ``pyproject.toml`` is line-based).
2. **Historical fix-density** -- count of ``fix:``/``fix(scope):``
   conventional-commit subjects (this repo's own convention, see CLAUDE.md's
   Git Protocol) whose commit touched a path under the package. A proxy for
   "how often did this module actually break in a way that needed a fix",
   not a synthetic bug-injection score.
3. **Test wall-time cost exposure** -- from ``testmon``'s own dependency
   database (``.cache/testmon/testmondata``): for every test whose recorded
   dependency fingerprint set includes at least one file under the package,
   sum that test's last-recorded duration. A test touching N packages
   contributes its full duration to each -- this is deliberately a *cost
   exposure* metric ("how much wall-time is on the hook if this package
   changes"), not a time partition, so package totals do not sum to the
   suite total. See the module docstring note below on why an exact,
   double-counting-free wall-time partition is not reconstructable from
   data already on disk.
4. **testmon selection fan-out** -- median, across files with a recorded
   fingerprint in the package, of "how many distinct tests depend on this
   file". Median (not sum/max) because a small number of hub files
   (``polylogue/storage/sqlite/connection.py``,
   ``polylogue/mcp/server_support.py``,
   ``polylogue/schemas/validator_resolution.py``,
   ``polylogue/daemon/status_snapshot.py`` at last measurement) are
   transitively imported by essentially every test and would otherwise
   swamp the per-package number to ~100% for four packages and hide the
   more informative "typical file in this package" fan-out.

Honesty note on wall-time provenance
-------------------------------------

``.cache/verify/runs/*/steps/*/summary.json`` (3,800+ historical verify runs)
only retains the **top 20 slowest** test reports per run
(``devtools/pytest_progress_plugin.py:_SLOW_REPORT_LIMIT``), not a full
per-test duration list -- that full list only ever exists transiently in
``.cache/verify/last-pytest.json`` for the *most recent* invocation, and is
overwritten on the next run. There is therefore no comprehensive historical
per-test wall-time record on disk. ``testmon``'s own database, however,
already tracks one row per currently-known test with its last-recorded
duration plus the full file-dependency graph testmon uses to select tests --
that is what this report uses instead of re-running a fresh full-suite
timing pass (which a full ``devtools verify --all`` would otherwise require).

Coverage still requires an actual instrumented run: this report does not
invoke pytest itself. Point it at a ``coverage json`` report (see
``--coverage-json``); generate one with:

    coverage json --data-file=.cache/coverage/.coverage -o .cache/coverage/coverage.json

after any full/near-full pytest run with ``--cov=polylogue``.

Usage
-----

    python -m devtools.test_economics_report
    python -m devtools.test_economics_report --json
    python -m devtools.test_economics_report --write docs/test-economics.md

Wired into ``devtools lab test-economics`` (see ``command_catalog.py``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_POLYLOGUE_ROOT = _REPO_ROOT / "polylogue"
_DEFAULT_TESTMON_DB = _REPO_ROOT / ".cache" / "testmon" / "testmondata"
_DEFAULT_COVERAGE_JSON = _REPO_ROOT / ".cache" / "coverage" / "coverage.json"
_ROOT_BUCKET = "_root"

# Fix-commit subject match: this repo's conventional-commit convention
# (CLAUDE.md Git Protocol) is `fix:` or `fix(scope):` at the start of the
# subject line. Matches git's own -E (extended regex) grep.
_FIX_GREP_PATTERN = r"^fix(\(|:)"


def discover_packages(polylogue_root: Path = _POLYLOGUE_ROOT) -> list[str]:
    """Every top-level polylogue/ subpackage, plus `_root` for stray top-level .py files."""
    packages = sorted(
        p.name
        for p in polylogue_root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")
    )
    has_root_files = any(p.is_file() and p.suffix == ".py" for p in polylogue_root.iterdir())
    if has_root_files:
        packages.append(_ROOT_BUCKET)
    return packages


def _package_for_relpath(relpath: str, packages: Iterable[str]) -> str | None:
    """Map a 'polylogue/<x>/...' or 'polylogue/<file>.py' path to its package bucket."""
    if not relpath.startswith("polylogue/"):
        return None
    parts = relpath.split("/")
    if len(parts) < 2:
        return None
    head = parts[1]
    if head.endswith(".py") and len(parts) == 2:
        return _ROOT_BUCKET if _ROOT_BUCKET in packages else None
    return head if head in packages else None


@dataclass
class PackageMetrics:
    package: str
    coverage_percent: float | None = None
    coverage_statements: int = 0
    coverage_covered: int = 0
    fix_commits: int = 0
    test_wall_time_exposure_s: float = 0.0
    tests_touching_package: int = 0
    selection_fanout_median: float | None = None
    selection_fanout_max: int = 0
    files_with_fingerprint: int = 0
    quadrant: str = "unclassified"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "coverage_percent": self.coverage_percent,
            "coverage_statements": self.coverage_statements,
            "coverage_covered": self.coverage_covered,
            "fix_commits": self.fix_commits,
            "test_wall_time_exposure_s": round(self.test_wall_time_exposure_s, 2),
            "tests_touching_package": self.tests_touching_package,
            "selection_fanout_median": self.selection_fanout_median,
            "selection_fanout_max": self.selection_fanout_max,
            "files_with_fingerprint": self.files_with_fingerprint,
            "quadrant": self.quadrant,
            "notes": self.notes,
        }


def compute_fix_density(
    packages: list[str], *, repo_root: Path = _REPO_ROOT, polylogue_root: Path = _POLYLOGUE_ROOT
) -> dict[str, int]:
    """Count fix:-prefixed commits touching each package's path, via one `git log` per package.

    One process per package (≈29 invocations) rather than one global log parsed in
    Python, because git's own `-- <path>` restriction is the simplest correct way to
    ask "did this commit touch this subtree" across renames within the tree.

    The `_root` bucket passes each top-level polylogue/*.py file as an explicit,
    non-glob pathspec -- a glob pathspec like `polylogue/*.py` is NOT anchored to
    one path segment in git's default (non-literal) pathspec matching, so `*`
    silently crosses `/` and matches every file in every subpackage too.
    """
    counts: dict[str, int] = {}
    for pkg in packages:
        if pkg == _ROOT_BUCKET:
            pathspec = [f"polylogue/{p.name}" for p in polylogue_root.iterdir() if p.is_file() and p.suffix == ".py"]
        else:
            pathspec = [f"polylogue/{pkg}"]
        cmd = [
            "git",
            "log",
            "--oneline",
            "-E",
            f"--grep={_FIX_GREP_PATTERN}",
            "--",
            *pathspec,
        ]
        result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        counts[pkg] = len(lines)
    return counts


def compute_coverage(packages: list[str], coverage_json_path: Path) -> tuple[dict[str, tuple[int, int]], str | None]:
    """Return {package: (covered_statements, total_statements)} from a coverage.py JSON report.

    Returns (empty dict, warning) if the report is missing.
    """
    if not coverage_json_path.exists():
        return {}, f"no coverage report found at {coverage_json_path}; coverage_percent left null"
    data = json.loads(coverage_json_path.read_text(encoding="utf-8"))
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    files = data.get("files", {})
    for filename, filedata in files.items():
        norm = filename.replace("\\", "/")
        if not norm.startswith("polylogue/"):
            continue
        pkg = _package_for_relpath(norm, packages)
        if pkg is None:
            continue
        summary = filedata.get("summary", {})
        totals[pkg][0] += int(summary.get("covered_lines", 0))
        totals[pkg][1] += int(summary.get("num_statements", 0))
    return {pkg: (v[0], v[1]) for pkg, v in totals.items()}, None


def compute_test_economics(
    packages: list[str], testmon_db_path: Path
) -> tuple[dict[str, tuple[float, int]], dict[str, tuple[float | None, int, int]], str | None]:
    """Query testmon's dependency graph for wall-time exposure and selection fan-out.

    Returns:
      wall_time[pkg] = (summed duration of distinct tests touching the package, test count)
      fanout[pkg] = (median per-file selected-test count, max per-file count, files counted)
      warning if the db is missing.
    """
    if not testmon_db_path.exists():
        return {}, {}, f"no testmon database found at {testmon_db_path}; wall-time/fan-out left null"

    import sqlite3

    con = sqlite3.connect(f"file:{testmon_db_path}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT te.test_name, te.duration, ff.filename
            FROM test_execution te
            JOIN test_execution_file_fp tefp ON tefp.test_execution_id = te.id
            JOIN file_fp ff ON ff.id = tefp.fingerprint_id
            WHERE ff.filename LIKE 'polylogue/%'
            """
        )
        rows = cur.fetchall()

        # wall-time exposure: distinct (test, package) touched
        pkg_tests: dict[str, set[str]] = defaultdict(set)
        test_dur: dict[str, float] = {}
        # per-file distinct test counts, for fan-out
        file_test_counts: dict[str, set[str]] = defaultdict(set)

        for name, dur, filename in rows:
            test_dur[name] = dur
            pkg = _package_for_relpath(filename, packages)
            if pkg is not None:
                pkg_tests[pkg].add(name)
            file_test_counts[filename].add(name)

        wall_time: dict[str, tuple[float, int]] = {}
        for pkg, tests in pkg_tests.items():
            total = sum(test_dur[t] for t in tests)
            wall_time[pkg] = (total, len(tests))

        pkg_file_counts: dict[str, list[int]] = defaultdict(list)
        for filename, testset in file_test_counts.items():
            pkg = _package_for_relpath(filename, packages)
            if pkg is not None:
                pkg_file_counts[pkg].append(len(testset))

        fanout: dict[str, tuple[float | None, int, int]] = {}
        for pkg, counts in pkg_file_counts.items():
            median = statistics.median(counts) if counts else None
            fanout[pkg] = (median, max(counts) if counts else 0, len(counts))

        return wall_time, fanout, None
    finally:
        con.close()


def classify_quadrants(metrics: dict[str, PackageMetrics]) -> None:
    """Assign a quadrant label using median-split thresholds within this package set.

    high fix-density + low coverage -> under-tested substrate
    low fix-density + (high wall-time OR high fan-out) -> over-tested mechanical surface
    everything else -> mixed / does not fit cleanly

    A package with no coverage number is classified "coverage unknown" (optionally
    qualified by fix-density) rather than silently defaulting to well-covered --
    absence of a measurement must never read as a clean bill of health.
    """
    fix_values = [m.fix_commits for m in metrics.values()]
    cov_values = [m.coverage_percent for m in metrics.values() if m.coverage_percent is not None]
    wall_values = [m.test_wall_time_exposure_s for m in metrics.values()]
    fanout_values = [m.selection_fanout_median for m in metrics.values() if m.selection_fanout_median is not None]

    if not fix_values:
        return
    fix_median = statistics.median(fix_values)
    cov_median = statistics.median(cov_values) if cov_values else None
    wall_median = statistics.median(wall_values) if wall_values else 0.0
    fanout_median_all = statistics.median(fanout_values) if fanout_values else 0.0

    for m in metrics.values():
        high_fix = m.fix_commits > fix_median
        high_wall = m.test_wall_time_exposure_s > wall_median
        high_fanout = m.selection_fanout_median is not None and m.selection_fanout_median > fanout_median_all

        if m.coverage_percent is None:
            m.quadrant = "high-fix, coverage unknown" if high_fix else "coverage unknown"
            continue

        assert cov_median is not None  # cov_values is non-empty whenever coverage_percent is set
        low_cov = m.coverage_percent < cov_median

        if high_fix and low_cov:
            m.quadrant = "under-tested substrate"
        elif (not high_fix) and (high_wall or high_fanout):
            m.quadrant = "over-tested mechanical surface"
        elif high_fix and not low_cov:
            m.quadrant = "well-covered risk area"
        elif (not high_fix) and (not high_wall) and (not high_fanout):
            m.quadrant = "low-risk, low-cost (fine as-is)"
        else:
            m.quadrant = "mixed / no clean fit"


def build_report(
    *,
    testmon_db_path: Path = _DEFAULT_TESTMON_DB,
    coverage_json_path: Path = _DEFAULT_COVERAGE_JSON,
    repo_root: Path = _REPO_ROOT,
) -> tuple[dict[str, PackageMetrics], list[str]]:
    warnings: list[str] = []
    packages = discover_packages(repo_root / "polylogue")
    metrics = {pkg: PackageMetrics(package=pkg) for pkg in packages}

    fix_counts = compute_fix_density(packages, repo_root=repo_root, polylogue_root=repo_root / "polylogue")
    for pkg, count in fix_counts.items():
        metrics[pkg].fix_commits = count

    coverage_totals, cov_warning = compute_coverage(packages, coverage_json_path)
    if cov_warning:
        warnings.append(cov_warning)
    for pkg, (covered, total) in coverage_totals.items():
        metrics[pkg].coverage_covered = covered
        metrics[pkg].coverage_statements = total
        metrics[pkg].coverage_percent = round(100.0 * covered / total, 1) if total else None

    wall_time, fanout, tm_warning = compute_test_economics(packages, testmon_db_path)
    if tm_warning:
        warnings.append(tm_warning)
    for pkg, (wall_time_s, touching_count) in wall_time.items():
        metrics[pkg].test_wall_time_exposure_s = wall_time_s
        metrics[pkg].tests_touching_package = touching_count
    for pkg, (median, mx, nfiles) in fanout.items():
        metrics[pkg].selection_fanout_median = median
        metrics[pkg].selection_fanout_max = mx
        metrics[pkg].files_with_fingerprint = nfiles

    classify_quadrants(metrics)
    return metrics, warnings


def render_markdown(metrics: dict[str, PackageMetrics], warnings: list[str]) -> str:
    ordered = sorted(metrics.values(), key=lambda m: (-m.fix_commits, m.package))
    lines: list[str] = []
    lines.append("<!-- Generated by `python -m devtools.test_economics_report --write docs/test-economics.md`. -->")
    lines.append("<!-- Edit devtools/test_economics_report.py, not this file. -->")
    lines.append("")
    lines.append("# Test-suite economics: coverage vs fix-density map")
    lines.append("")
    lines.append(
        "Per-package view of where tests earn their runtime (polylogue-9e5.11). See "
        "`devtools/test_economics_report.py` module docstring for exact metric "
        "definitions and honesty notes on data provenance."
    )
    lines.append("")
    if warnings:
        lines.append("**Warnings from this run:**")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    lines.append(
        "| Package | Coverage % | Fix commits | Wall-time exposure (s) | Tests touching | "
        "Fan-out median/max | Quadrant |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for m in ordered:
        cov = f"{m.coverage_percent:.1f}" if m.coverage_percent is not None else "n/a"
        fanout = (
            f"{m.selection_fanout_median:.0f}/{m.selection_fanout_max}"
            if m.selection_fanout_median is not None
            else "n/a"
        )
        lines.append(
            f"| `{m.package}` | {cov} | {m.fix_commits} | {m.test_wall_time_exposure_s:.1f} | "
            f"{m.tests_touching_package} | {fanout} | {m.quadrant} |"
        )
    lines.append("")
    lines.append(
        "Wall-time exposure double-counts tests that touch multiple packages by design "
        "(cost-exposure, not a partition); fan-out is median/max distinct-test count "
        "per file with a recorded fingerprint in the package, not a package-level sum "
        "(a handful of hub files are imported by ~every test and would otherwise swamp "
        "the package number to ~100%)."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--testmon-db", type=Path, default=_DEFAULT_TESTMON_DB)
    parser.add_argument("--coverage-json", type=Path, default=_DEFAULT_COVERAGE_JSON)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of markdown.")
    parser.add_argument("--write", type=Path, default=None, help="Write the markdown table to this path.")
    args = parser.parse_args(argv)

    metrics, warnings = build_report(
        testmon_db_path=args.testmon_db,
        coverage_json_path=args.coverage_json,
    )

    if args.json:
        payload = {
            "packages": {pkg: m.to_dict() for pkg, m in sorted(metrics.items())},
            "warnings": warnings,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        markdown = render_markdown(metrics, warnings)
        if args.write:
            args.write.write_text(markdown, encoding="utf-8")
            sys.stderr.write(f"wrote {args.write}\n")
        else:
            print(markdown)

    for w in warnings:
        sys.stderr.write(f"warning: {w}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
