"""Test-shaped fixture nodes for the reachability oracle tests."""

from __future__ import annotations

from .routes import dead_helper, production_entrypoint


def test_wired_route() -> None:
    assert production_entrypoint() == "live"


def test_dead_helper() -> None:
    assert dead_helper() == "dead"
