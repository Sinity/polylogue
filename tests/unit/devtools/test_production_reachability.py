"""Tests for the structured production-route reachability oracle."""

from __future__ import annotations

from pathlib import Path

from devtools.production_reachability import ProductionSeamSpec, check_production_seam

_FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "production_reachability"


def test_wired_fixture_route_is_reachable() -> None:
    report = check_production_seam(
        ProductionSeamSpec(
            test_path="fixture_test.py",
            test_function="test_wired_route",
            production_entrypoint="routes.production_entrypoint",
            tested_symbols=("routes.production_entrypoint",),
            required_symbols=("routes.live_helper",),
            production_namespace="routes",
        ),
        source_root=_FIXTURE_ROOT,
    )

    assert report.ok, report.to_json()


def test_unreachable_tested_symbol_is_a_structured_failure() -> None:
    report = check_production_seam(
        ProductionSeamSpec(
            test_path="fixture_test.py",
            test_function="test_dead_helper",
            production_entrypoint="routes.production_entrypoint",
            tested_symbols=("routes.dead_helper",),
            production_namespace="routes",
        ),
        source_root=_FIXTURE_ROOT,
    )

    assert not report.ok
    assert [violation.to_dict() for violation in report.violations] == [
        {"code": "tested_symbol_unreachable", "symbol": "routes.dead_helper"}
    ]
    assert report.to_dict()["violations"] == [{"code": "tested_symbol_unreachable", "symbol": "routes.dead_helper"}]
