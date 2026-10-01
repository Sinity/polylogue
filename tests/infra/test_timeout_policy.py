"""Runtime policy for explicit pytest timeout markers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.infra.timeout_policy import timeout_marker_error


@pytest.mark.parametrize("value", [None, -1, float("inf"), float("nan"), 901, "30", True])
def test_collection_rejects_invalid_timeout_markers(value: Any) -> None:
    marker = pytest.mark.timeout(value).mark
    assert timeout_marker_error(marker) is not None


@pytest.mark.parametrize("value", [0, 0.1, 30, 120, 900])
def test_collection_accepts_explicit_timeout_markers(value: float) -> None:
    marker = pytest.mark.timeout(value).mark
    assert timeout_marker_error(marker) is None


@pytest.mark.parametrize("value, accepted", [(0, True), (-1, False)])
def test_repository_collection_hook_applies_timeout_policy(value: int, accepted: bool) -> None:
    """Reject invalid markers in collection; zero keeps progressing work cancellable."""
    from tests.conftest import pytest_collection_modifyitems

    item = cast(
        "pytest.Item",
        SimpleNamespace(
            nodeid="tests/example.py::test_case",
            path="tests/example.py",
            get_closest_marker=lambda name: pytest.mark.timeout(value).mark if name == "timeout" else None,
        ),
    )
    config = cast("pytest.Config", SimpleNamespace(getoption=lambda name: None))

    if accepted:
        pytest_collection_modifyitems(config, [item])
    else:
        with pytest.raises(pytest.UsageError):
            pytest_collection_modifyitems(config, [item])
