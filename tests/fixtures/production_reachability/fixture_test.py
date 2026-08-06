"""Test-shaped fixture nodes for the reachability oracle tests."""

from __future__ import annotations

from routes import dead_helper, production_entrypoint


def test_wired_route() -> str:
    return production_entrypoint()


def test_dead_helper() -> str:
    return dead_helper()
