"""Per-test temporary storage stays bounded within one run."""

from __future__ import annotations

import pytest


def test_passing_tests_release_their_tmp_path_at_teardown(pytestconfig: pytest.Config) -> None:
    """The effective policy removes a passing test's tree when it finishes.

    Anti-vacuity: drop ``tmp_path_retention_policy`` from ``pyproject.toml``
    and pytest's default ``all`` keeps every tree until the run ends, which
    this check reports.
    """
    assert pytestconfig.getini("tmp_path_retention_policy") == "failed"
