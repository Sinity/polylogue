"""Tests for the read-surface SLO catalog and ``devtools verify-slos``.

The catalog at ``docs/plans/slo-catalog.yaml`` lists the SLOs that
``devtools/verify_slos.py`` consumes. These tests pin four contracts:

1. The catalog is committed and parses cleanly with entries for at least the
   query/reader/facets/context/cost surfaces.
2. ``verify-slos`` reports zero violations and exits 0 when measured stats
   sit comfortably under budget.
3. ``verify-slos`` reports a violation and exits non-zero when a measured
   surface exceeds its declared budget.
4. ``verify-slos`` treats missing benchmark results as blocking for required
   rows and non-blocking for informational rows.

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


def _write_slo_catalog(tmp_path: Path, body: str) -> Path:
    catalog = tmp_path / "slo-catalog.yaml"
    catalog.write_text(body)
    return catalog


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


def test_required_missing_benchmark_result_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required SLO rows must not pass when their benchmark artifact is absent."""
    catalog = _write_slo_catalog(
        tmp_path,
        """
surfaces:
  reader:
    description: "Reader endpoint"
    benchmark_test: "tests/benchmarks/test_reader_api.py::test_missing_required"
    p50_ms: 100
    p95_ms: 200
    gate: "required"
""",
    )
    monkeypatch.setattr(verify_slos, "_run_benchmarks", lambda _ids: {})

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = verify_slos.main(["--yaml", str(catalog), "--json"])

    payload = json.loads(buffer.getvalue())
    assert rc != 0
    assert payload["blocking"] is True
    assert payload["missing_required"] == [
        {
            "surface": "reader",
            "gate": "required",
            "benchmark_test": "tests/benchmarks/test_reader_api.py::test_missing_required",
            "reason": "no benchmark result for this test",
        }
    ]
    assert payload["uncovered_informational"] == []


def test_informational_missing_benchmark_result_is_visible_but_nonblocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Informational rows can document future SLOs without blocking the gate."""
    catalog = _write_slo_catalog(
        tmp_path,
        """
surfaces:
  cost:
    description: "Cost rollup placeholder"
    benchmark_test: "tests/benchmarks/test_reader_api.py::test_missing_informational"
    p50_ms: 100
    p95_ms: 200
    gate: "informational"
""",
    )
    monkeypatch.setattr(verify_slos, "_run_benchmarks", lambda _ids: {})

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = verify_slos.main(["--yaml", str(catalog), "--json"])

    payload = json.loads(buffer.getvalue())
    assert rc == 0
    assert payload["blocking"] is False
    assert payload["missing_required"] == []
    assert payload["uncovered_informational"] == [
        {
            "surface": "cost",
            "gate": "informational",
            "benchmark_test": "tests/benchmarks/test_reader_api.py::test_missing_informational",
            "reason": "no benchmark result for this test",
        }
    ]


def test_invalid_slo_gate_is_catalog_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown gate values fail the catalog instead of being guessed."""
    catalog = _write_slo_catalog(
        tmp_path,
        """
surfaces:
  query:
    description: "Search endpoint"
    benchmark_test: "tests/benchmarks/test_search_filters.py::test_bench_query"
    p50_ms: 100
    p95_ms: 200
    gate: "aspirational"
""",
    )
    monkeypatch.setattr(verify_slos, "_run_benchmarks", lambda _ids: {})

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = verify_slos.main(["--yaml", str(catalog), "--json"])

    payload = json.loads(buffer.getvalue())
    assert rc != 0
    assert payload["blocking"] is True
    assert payload["catalog_errors"] == [
        "query: invalid gate 'aspirational'; expected one of ['informational', 'required']"
    ]
