"""Tests for the read-surface SLO catalog and ``devtools verify-slos``.

The catalog at ``docs/plans/slo-catalog.yaml`` lists the SLOs that
``devtools/verify_slos.py`` consumes. These tests pin three contracts:

1. The catalog is committed and parses cleanly with entries for at least the
   query/reader/facets/context/cost surfaces.
2. ``verify-slos`` reports zero violations and exits 0 when measured stats
   sit comfortably under budget.
3. ``verify-slos`` reports a violation and exits non-zero when a measured
   surface exceeds its declared budget.

The benchmark subprocess is not executed here — the tests stub
``_run_benchmarks`` so the SLO scoring path runs deterministically.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from devtools import verify_slos

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "docs" / "plans" / "slo-catalog.yaml"

REQUIRED_SURFACES = ("query", "reader", "facets", "context", "cost")


def test_catalog_exists_and_covers_required_surfaces() -> None:
    """The committed SLO catalog must cover the surfaces enumerated in #872."""
    assert CATALOG_PATH.exists(), f"missing SLO catalog at {CATALOG_PATH}"

    surfaces = verify_slos._parse_slo_catalog(CATALOG_PATH.read_text())

    missing = [name for name in REQUIRED_SURFACES if name not in surfaces]
    assert not missing, f"SLO catalog missing required surfaces: {missing}"

    for name in REQUIRED_SURFACES:
        config = surfaces[name]
        bench = config.get("benchmark_test")
        p50 = config.get("p50_ms")
        p95 = config.get("p95_ms")
        assert isinstance(bench, str) and bench, f"surface {name!r} must declare a benchmark_test"
        assert isinstance(p50, int), f"surface {name!r} must declare an integer p50_ms budget"
        assert isinstance(p95, int), f"surface {name!r} must declare an integer p95_ms budget"
        assert p50 > 0
        assert p95 >= p50


def _stub_benchmark_stats(
    surfaces: dict[str, dict[str, object]],
    *,
    mean_s: float,
    stddev_s: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Build a benchmark-stats dict where every surface produces the given mean."""
    stats: dict[str, dict[str, float]] = {}
    for config in surfaces.values():
        node_id = config.get("benchmark_test")
        if isinstance(node_id, str):
            stats[node_id] = {
                "mean": mean_s,
                "median": mean_s,
                "min": mean_s,
                "max": mean_s,
                "stddev": stddev_s,
                "rounds": 5,
            }
    return stats


def test_verify_slos_passes_when_under_budget(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When every surface measures fast enough, verify_slos exits 0."""
    surfaces = verify_slos._parse_slo_catalog(CATALOG_PATH.read_text())
    fast_stats = _stub_benchmark_stats(surfaces, mean_s=0.001)

    monkeypatch.setattr(verify_slos, "_run_benchmarks", lambda _ids: fast_stats)

    rc = verify_slos.main(["--yaml", str(CATALOG_PATH)])
    out = capsys.readouterr().out
    assert rc == 0, f"verify_slos exited non-zero with stdout={out}"
    assert "blocking=False" in out
    assert "VIOLATION" not in out


def test_verify_slos_fails_when_surface_exceeds_budget(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When a measured surface blows its budget, verify_slos exits non-zero."""
    surfaces = verify_slos._parse_slo_catalog(CATALOG_PATH.read_text())
    # 10 seconds is well above every declared p50/p95 budget in the catalog.
    slow_stats = _stub_benchmark_stats(surfaces, mean_s=10.0)

    monkeypatch.setattr(verify_slos, "_run_benchmarks", lambda _ids: slow_stats)

    rc = verify_slos.main(["--yaml", str(CATALOG_PATH)])
    out = capsys.readouterr().out
    assert rc != 0, f"verify_slos should reject over-budget measurements; stdout={out}"
    assert "VIOLATION" in out
    assert "blocking=True" in out


def test_verify_slos_json_reports_violation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """JSON output mirrors the human report and flags the violation."""
    surfaces = verify_slos._parse_slo_catalog(CATALOG_PATH.read_text())
    slow_stats = _stub_benchmark_stats(surfaces, mean_s=10.0)

    monkeypatch.setattr(verify_slos, "_run_benchmarks", lambda _ids: slow_stats)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = verify_slos.main(["--yaml", str(CATALOG_PATH), "--json"])

    payload = json.loads(buffer.getvalue())
    assert rc != 0
    assert payload["blocking"] is True
    assert payload["violations"], "expected violations array to be non-empty"
    surfaces_with_violations = {entry["surface"] for entry in payload["violations"]}
    # All catalog surfaces must trip when fed 10s measurements.
    assert set(REQUIRED_SURFACES).issubset(surfaces_with_violations)
