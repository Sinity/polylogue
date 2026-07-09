"""9e5.11: pin the quadrant-classification policy in devtools/test_economics_report.py.

classify_quadrants is the "policy" part of the economics report -- the metric
collectors (git log, coverage.py JSON, testmon db) are thin I/O adapters, but
the median-split classification logic is where a subtle bug would silently
misdirect follow-up effort. Tested directly against synthetic PackageMetrics,
not through the full git/coverage/testmon collection pipeline.
"""

from __future__ import annotations

from devtools.test_economics_report import PackageMetrics, classify_quadrants


def test_classify_quadrants_under_tested_substrate() -> None:
    """High fix-density + below-median coverage -> under-tested substrate."""
    metrics = {
        "hot": PackageMetrics(package="hot", fix_commits=100, coverage_percent=50.0),
        "cold": PackageMetrics(package="cold", fix_commits=0, coverage_percent=99.0),
    }
    classify_quadrants(metrics)
    assert metrics["hot"].quadrant == "under-tested substrate"


def test_classify_quadrants_over_tested_mechanical_surface() -> None:
    """Low fix-density + high wall-time or fan-out -> over-tested mechanical surface."""
    metrics = {
        "expensive": PackageMetrics(
            package="expensive",
            fix_commits=0,
            coverage_percent=99.0,
            test_wall_time_exposure_s=1000.0,
        ),
        "cheap": PackageMetrics(
            package="cheap",
            fix_commits=0,
            coverage_percent=99.0,
            test_wall_time_exposure_s=1.0,
        ),
    }
    classify_quadrants(metrics)
    assert metrics["expensive"].quadrant == "over-tested mechanical surface"
    assert metrics["cheap"].quadrant == "low-risk, low-cost (fine as-is)"


def test_classify_quadrants_well_covered_risk_area() -> None:
    """High fix-density + at-or-above-median coverage -> well-covered risk area."""
    metrics = {
        "hot_covered": PackageMetrics(package="hot_covered", fix_commits=100, coverage_percent=99.0),
        "cold": PackageMetrics(package="cold", fix_commits=0, coverage_percent=50.0),
    }
    classify_quadrants(metrics)
    assert metrics["hot_covered"].quadrant == "well-covered risk area"


def test_classify_quadrants_missing_coverage_is_never_silently_clean() -> None:
    """A package with no coverage number must never read as a clean bill of health."""
    metrics = {
        "unmeasured_hot": PackageMetrics(package="unmeasured_hot", fix_commits=100, coverage_percent=None),
        "unmeasured_cold": PackageMetrics(package="unmeasured_cold", fix_commits=0, coverage_percent=None),
        "measured": PackageMetrics(package="measured", fix_commits=1, coverage_percent=80.0),
    }
    classify_quadrants(metrics)
    assert metrics["unmeasured_hot"].quadrant == "high-fix, coverage unknown"
    assert metrics["unmeasured_cold"].quadrant == "coverage unknown"
    assert "unknown" not in metrics["measured"].quadrant


def test_classify_quadrants_empty_metrics_is_a_noop() -> None:
    metrics: dict[str, PackageMetrics] = {}
    classify_quadrants(metrics)
    assert metrics == {}
