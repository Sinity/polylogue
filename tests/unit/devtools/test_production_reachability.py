"""Tests for the structured production-route reachability oracle."""

from __future__ import annotations

import ast
from pathlib import Path

from devtools.production_reachability import (
    ProductionSeamSpec,
    _CallGraph,
    _calls_in_function,
    _imports_from_nodes,
    _imports_in_function,
    _parse_modules,
    _scan_function_body,
    check_production_seam,
)

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


def test_declared_fixture_boundary_is_bound_to_test_signature() -> None:
    report = check_production_seam(
        ProductionSeamSpec(
            test_path="fixture_test.py",
            test_function="test_wired_route_with_tmp_path",
            production_entrypoint="routes.production_entrypoint",
            tested_symbols=("routes.production_entrypoint",),
            production_namespace="routes",
            fixture_boundary=("tmp_path",),
        ),
        source_root=_FIXTURE_ROOT,
    )

    assert report.ok, report.to_json()
    assert report.spec.fixture_boundary == ("tmp_path",)


def test_fixture_boundary_rejects_removed_or_renamed_fixture() -> None:
    report = check_production_seam(
        ProductionSeamSpec(
            test_path="fixture_test.py",
            test_function="test_wired_route_with_tmp_path",
            production_entrypoint="routes.production_entrypoint",
            tested_symbols=("routes.production_entrypoint",),
            production_namespace="routes",
            fixture_boundary=("workspace_env",),
        ),
        source_root=_FIXTURE_ROOT,
    )

    assert [violation.to_dict() for violation in report.violations] == [
        {"code": "fixture_boundary_not_declared", "symbol": "workspace_env"}
    ]


def test_fixture_boundary_rejects_path_widening() -> None:
    report = check_production_seam(
        ProductionSeamSpec(
            test_path="fixture_test.py",
            test_function="test_wired_route_with_tmp_path",
            production_entrypoint="routes.production_entrypoint",
            tested_symbols=("routes.production_entrypoint",),
            production_namespace="routes",
            fixture_boundary=("/realm/data",),
        ),
        source_root=_FIXTURE_ROOT,
    )

    assert [violation.to_dict() for violation in report.violations] == [
        {"code": "fixture_boundary_invalid", "symbol": "/realm/data"}
    ]


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


def test_union_traversal_matches_walking_each_root_separately() -> None:
    """``reachable_from_any`` is the union, not the first root's component.

    Consumer reachability walks ten overlapping entrypoints; doing it in one
    traversal is only sound if reachability distributes over union.

    Anti-vacuity: seeding only the first root (or dropping the per-root
    top-level function seeding) loses ``nestedpkg.child.child_route``, which
    nothing in ``routes`` reaches, and the equality below goes red.
    """
    graph = _CallGraph(_parse_modules(_FIXTURE_ROOT, (_FIXTURE_ROOT,)))
    roots = ("routes", "nestedpkg")

    union = graph.reachable_from_any(roots)

    assert union == graph.reachable_from("routes") | graph.reachable_from("nestedpkg")
    assert "routes.live_helper" in union
    assert "nestedpkg.child.child_route" in union
    assert "nestedpkg.child.child_route" not in graph.reachable_from("routes")
    # A repeated root must not change the answer -- the real entrypoint tuple
    # names ``polylogue.cli`` twice.
    assert graph.reachable_from_any(("routes", "nestedpkg", "routes")) == union


def test_one_body_scan_collects_calls_and_imports_without_entering_nested_scopes() -> None:
    """Imports and calls come from a single walk with one traversal rule.

    Anti-vacuity: if ``_BodyScanner`` descended into the nested function, the
    lambda, or the class body, ``dead_helper`` would appear among the calls
    and ``json``/``os``/``sys`` would appear among the imports. If the merged
    scanner lost either collection, one of the two assertions is empty.
    """
    module = ast.parse(
        "def route():\n"
        "    import subprocess\n"
        "    if route:\n"
        "        from pathlib import Path\n"
        "    live_helper()\n"
        "    def nested():\n"
        "        import os\n"
        "        return dead_helper()\n"
        "    handler = lambda: dead_helper()\n"
        "    class Inner:\n"
        "        import sys\n"
        "        def method(self):\n"
        "            return dead_helper()\n"
        "    return nested, handler, Inner\n"
    )
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)

    scan = _scan_function_body(function)

    assert [call.func.id for call in scan.calls if isinstance(call.func, ast.Name)] == ["live_helper"]
    assert sorted(_imports_in_function(function, "fixture")) == ["Path", "subprocess"]
    assert tuple(scan.calls) == _calls_in_function(function)
    assert _imports_from_nodes(scan.imports, "fixture") == _imports_in_function(function, "fixture")
