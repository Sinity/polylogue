"""Per-directory fixtures and hooks for source tests.

Assigns pytest-xdist groups per provider so parametrized Hypothesis
tests (test_looks_like_never_crashes, test_parse_never_crashes)
distribute across workers instead of serializing on one (#1026).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest import Item


def pytest_collection_modifyitems(items: list[Item]) -> None:
    for item in items:
        callspec = getattr(item, "callspec", None)
        provider = callspec.params.get("provider") if callspec is not None else None
        if provider:
            item.add_marker(pytest.mark.xdist_group(name=provider))
