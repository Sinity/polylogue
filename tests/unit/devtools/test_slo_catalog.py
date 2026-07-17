"""Tests for the read-surface SLO catalog and ``devtools bench slo``.

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
BENCH_FIXTURE_DIR = REPO_ROOT / "tests" / "data" / "pytest-benchmark"

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
    receipt = payload["workload_receipt"]
    assert receipt["status"] == "failed"
    assert {result["verdict"] for result in receipt["budget_results"]} == {"exceeded"}
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
            "tier": "cheap-local",
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
            "tier": "cheap-local",
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


def test_required_surface_with_malformed_budget_is_catalog_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _write_slo_catalog(
        tmp_path,
        """
surfaces:
  reader:
    description: "Reader endpoint"
    benchmark_test: "tests/benchmarks/test_reader_api.py::test_reader"
    p50_ms: "100"
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
    assert payload["catalog_errors"] == ["reader: required surface must declare integer p50_ms and p95_ms"]


def test_required_surface_with_malformed_benchmark_test_is_catalog_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _write_slo_catalog(
        tmp_path,
        """
surfaces:
  reader:
    description: "Reader endpoint"
    benchmark_test:
      - "tests/benchmarks/test_reader_api.py::test_reader"
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
    assert payload["catalog_errors"] == ["reader: required surface must declare benchmark_test (string)"]


# ---------------------------------------------------------------------------
# Tier filtering (Pack B)
# ---------------------------------------------------------------------------


def test_catalog_required_surfaces_are_cheap_local_tier() -> None:
    """All ``gate: required`` rows must live in the cheap-local tier.

    Promoting a row to ``required`` while leaving it in the ``lab`` tier would
    create a gate that ``devtools verify`` (default loop) cannot reach but that
    still blocks PRs once anyone runs the lab loop. Required-but-lab is a
    contradiction the catalog must forbid by convention.
    """
    surfaces = verify_slos._parse_slo_catalog(CATALOG_PATH.read_text())
    offenders: list[str] = []
    for name, config in surfaces.items():
        gate = config.get("gate", "required")
        tier = config.get("tier", verify_slos.DEFAULT_TIER)
        if gate == "required" and tier != "cheap-local":
            offenders.append(f"{name} (tier={tier!r})")
    assert not offenders, (
        "required SLO rows must be in the cheap-local tier so the default "
        f"verify loop can run them; offenders: {offenders}"
    )


def test_lab_tier_surface_skipped_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lab-tier rows are skipped (and reported) unless --include-lab is set."""
    catalog = _write_slo_catalog(
        tmp_path,
        """
surfaces:
  daemon_lab:
    description: "Lab-only convergence"
    benchmark_test: "tests/benchmarks/test_daemon_convergence.py::test_lab_node"
    p50_ms: 30000
    p95_ms: 60000
    gate: "required"
    tier: "lab"
""",
    )
    captured_ids: dict[str, set[str]] = {"ids": set()}

    def fake_run(ids: set[str]) -> dict[str, dict[str, float]]:
        captured_ids["ids"] = ids
        return {}

    monkeypatch.setattr(verify_slos, "_run_benchmarks", fake_run)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = verify_slos.main(["--yaml", str(catalog), "--json"])

    payload = json.loads(buffer.getvalue())
    assert rc == 0, payload  # lab row is filtered out, so nothing blocks
    assert payload["active_tiers"] == ["cheap-local"]
    assert payload["skipped_tier"] == [
        {
            "surface": "daemon_lab",
            "gate": "required",
            "tier": "lab",
            "reason": "tier 'lab' not in active tiers ['cheap-local']",
        }
    ]
    assert payload["missing_required"] == []
    # No tests collected for execution because lab tier was filtered.
    assert captured_ids["ids"] == set()


def test_lab_tier_surface_runs_with_include_lab(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--include-lab`` adds lab-tier rows to the execution and gating set."""
    catalog = _write_slo_catalog(
        tmp_path,
        """
surfaces:
  daemon_lab:
    description: "Lab-only convergence"
    benchmark_test: "tests/benchmarks/test_daemon_convergence.py::test_lab_node"
    p50_ms: 30000
    p95_ms: 60000
    gate: "required"
    tier: "lab"
""",
    )
    captured_ids: dict[str, set[str]] = {"ids": set()}

    def fake_run(ids: set[str]) -> dict[str, dict[str, float]]:
        captured_ids["ids"] = ids
        return {}

    monkeypatch.setattr(verify_slos, "_run_benchmarks", fake_run)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = verify_slos.main(["--yaml", str(catalog), "--json", "--include-lab"])

    payload = json.loads(buffer.getvalue())
    assert rc != 0, payload  # lab row is now active and its artifact is missing
    assert set(payload["active_tiers"]) == {"cheap-local", "lab"}
    assert payload["skipped_tier"] == []
    assert payload["missing_required"] and payload["missing_required"][0]["surface"] == "daemon_lab"
    assert captured_ids["ids"] == {"tests/benchmarks/test_daemon_convergence.py::test_lab_node"}


def test_invalid_tier_is_catalog_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown tier values must fail the catalog rather than being guessed."""
    catalog = _write_slo_catalog(
        tmp_path,
        """
surfaces:
  query:
    description: "Search endpoint"
    benchmark_test: "tests/benchmarks/test_search_filters.py::test_bench_query"
    p50_ms: 100
    p95_ms: 200
    gate: "required"
    tier: "ci-nightly"
""",
    )
    monkeypatch.setattr(verify_slos, "_run_benchmarks", lambda _ids: {})

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = verify_slos.main(["--yaml", str(catalog), "--json", "--all-tiers"])

    payload = json.loads(buffer.getvalue())
    assert rc != 0
    assert payload["blocking"] is True
    assert payload["catalog_errors"] == ["query: invalid tier 'ci-nightly'; expected one of ['cheap-local', 'lab']"]


# ---------------------------------------------------------------------------
# Real pytest-benchmark JSON fixture parsing (Pack B)
# ---------------------------------------------------------------------------


def test_real_pytest_benchmark_json_fixture_parses() -> None:
    """A committed pytest-benchmark JSON artifact parses through the shared loader.

    Pack B requires at least one real pytest-benchmark artifact to flow through
    the parser, not just hand-built stats dicts. The fixture was captured from
    an actual pytest-benchmark run and lives under tests/data/pytest-benchmark/.
    """
    fixture = BENCH_FIXTURE_DIR / "reader-status.json"
    assert fixture.exists(), f"missing benchmark fixture: {fixture}"

    payload = json.loads(fixture.read_text())
    from devtools.benchmark_results import parse_pytest_benchmark_stats

    stats = parse_pytest_benchmark_stats(payload)
    assert len(stats) >= 1
    by_name = {entry.fullname: entry for entry in stats}
    target = "tests/benchmarks/test_reader_api.py::test_bench_reader_status"
    assert target in by_name
    entry = by_name[target]
    assert entry.mean > 0
    assert entry.median > 0
    assert entry.rounds >= 1


def test_verify_slos_scores_against_real_pytest_benchmark_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real pytest-benchmark JSON drives the SLO scoring path end to end.

    This replaces the stub-only path: we feed the parser the same artifact
    shape pytest-benchmark writes, then assert verify_slos translates seconds
    to milliseconds, applies the budget, and reports pass.
    """
    fixture_payload = json.loads((BENCH_FIXTURE_DIR / "reader-status.json").read_text())
    from devtools.benchmark_results import parse_pytest_benchmark_stats

    parsed_stats = {
        entry.fullname: {
            "mean": entry.mean,
            "median": entry.median,
            "min": entry.minimum,
            "max": entry.maximum,
            "stddev": entry.stddev,
            "rounds": entry.rounds,
        }
        for entry in parse_pytest_benchmark_stats(fixture_payload)
    }
    assert parsed_stats, "fixture must yield at least one parsed stat"

    catalog = _write_slo_catalog(
        tmp_path,
        """
surfaces:
  reader_status:
    description: "Reader status placeholder"
    benchmark_test: "tests/benchmarks/test_reader_api.py::test_bench_reader_status"
    p50_ms: 50
    p95_ms: 100
    gate: "required"
    tier: "cheap-local"
""",
    )
    monkeypatch.setattr(verify_slos, "_run_benchmarks", lambda _ids: parsed_stats)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = verify_slos.main(["--yaml", str(catalog), "--json"])

    payload = json.loads(buffer.getvalue())
    assert rc == 0, payload
    assert payload["passed"], "expected the real-fixture surface to pass"
    measured = payload["passed"][0]
    # fixture median is 0.0011s = 1.1ms; budget is 50ms.
    assert measured["actual_p50_ms"] < measured["target_p50_ms"]
    assert measured["actual_p95_ms"] < measured["target_p95_ms"]
