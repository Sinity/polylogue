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


def test_nested_callable_body_does_not_create_a_production_edge() -> None:
    report = check_production_seam(
        ProductionSeamSpec(
            test_path="fixture_test.py",
            test_function="test_nested_route",
            production_entrypoint="routes.route_with_nested",
            tested_symbols=("routes.route_with_nested",),
            required_symbols=("routes.dead_helper",),
            production_namespace="routes",
        ),
        source_root=_FIXTURE_ROOT,
    )

    assert [violation.code for violation in report.violations] == ["required_symbol_unreachable"]


def test_passed_callable_argument_does_not_create_a_test_edge() -> None:
    report = check_production_seam(
        ProductionSeamSpec(
            test_path="fixture_test.py",
            test_function="test_argument_route",
            production_entrypoint="routes.route_accepts_helper",
            tested_symbols=("routes.route_accepts_helper",),
            required_symbols=("routes.dead_helper",),
            production_namespace="routes",
        ),
        source_root=_FIXTURE_ROOT,
    )

    assert [violation.code for violation in report.violations] == ["required_symbol_unreachable"]


def test_signature_calls_do_not_create_a_production_edge() -> None:
    report = check_production_seam(
        ProductionSeamSpec(
            test_path="fixture_test.py",
            test_function="test_signature_route",
            production_entrypoint="routes.route_with_signature_helper",
            tested_symbols=("routes.route_with_signature_helper",),
            required_symbols=("routes.dead_helper",),
            production_namespace="routes",
        ),
        source_root=_FIXTURE_ROOT,
    )

    assert [violation.code for violation in report.violations] == ["required_symbol_unreachable"]


def test_shadowed_import_is_not_resolved_as_a_production_edge() -> None:
    report = check_production_seam(
        ProductionSeamSpec(
            test_path="fixture_test.py",
            test_function="test_shadowed_route",
            production_entrypoint="routes.shadowed_route",
            tested_symbols=("routes.shadowed_route",),
            required_symbols=("routes.live_helper",),
            production_namespace="routes",
        ),
        source_root=_FIXTURE_ROOT,
    )

    assert [violation.code for violation in report.violations] == ["required_symbol_unreachable"]


def test_class_constructor_and_method_are_reachable() -> None:
    report = check_production_seam(
        ProductionSeamSpec(
            test_path="fixture_test.py",
            test_function="test_class_route",
            production_entrypoint="routes.class_route",
            tested_symbols=("routes.class_route",),
            required_symbols=("routes.Runner.run",),
            production_namespace="routes",
        ),
        source_root=_FIXTURE_ROOT,
    )

    assert report.ok, report.to_json()


def test_package_initializer_relative_import_is_resolved() -> None:
    report = check_production_seam(
        ProductionSeamSpec(
            test_path="fixture_test.py",
            test_function="test_package_route",
            production_entrypoint="nestedpkg.package_route",
            tested_symbols=("nestedpkg.package_route",),
            required_symbols=("nestedpkg.child.child_route",),
            production_namespace="nestedpkg",
        ),
        source_root=_FIXTURE_ROOT,
    )

    assert report.ok, report.to_json()
