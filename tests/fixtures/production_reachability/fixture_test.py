"""Test-shaped fixture nodes for the reachability oracle tests."""

from __future__ import annotations

from .nestedpkg import package_route
from .routes import (
    class_route,
    dead_helper,
    production_entrypoint,
    route_accepts_helper,
    route_with_nested,
    shadowed_route,
)


def test_wired_route() -> None:
    assert production_entrypoint() == "live"


def test_dead_helper() -> None:
    assert dead_helper() == "dead"


def test_nested_route() -> None:
    assert route_with_nested() == "live"


def test_argument_route() -> None:
    assert route_accepts_helper(dead_helper) == "live"


def test_shadowed_route() -> None:
    assert shadowed_route() == "shadowed"


def test_class_route() -> None:
    assert class_route() == "live"


def test_package_route() -> None:
    assert package_route() == "child"
