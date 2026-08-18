"""Runtime policy for explicit pytest timeout markers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.infra.timeout_policy import timeout_marker_error


@pytest.mark.parametrize("value", [None, 0, -1, float("inf"), 901, "30"])
def test_collection_rejects_unbounded_timeout_markers(value: Any) -> None:
    marker = pytest.mark.timeout(value).mark
    assert "0 < seconds <= 900" in (timeout_marker_error(marker) or "")


@pytest.mark.parametrize("value", [0.1, 30, 120, 900])
def test_collection_accepts_bounded_timeout_markers(value: float) -> None:
    marker = pytest.mark.timeout(value).mark
    assert timeout_marker_error(marker) is None


def test_repository_collection_hook_rejects_zero_timeout() -> None:
    """The check is wired into collection, not merely correct in isolation.

    Called directly rather than through `pytester.runpytest_subprocess`. The two
    tests above already cover the predicate for 0, -1, inf, 901 and friends, so
    the only thing a subprocess added was proof of WIRING -- and it bought that
    at the price of a full nested pytest that inherits the managed run's
    environment. Under `devtools verify` that inheritance breaks it outright: the
    outer run sets PYTEST_DISABLE_PLUGIN_AUTOLOAD, the child does not load
    pytest-benchmark, and it then dies on the benchmark flags pyproject's addopts
    still supply.

    Calling the hook proves the same wiring in milliseconds. That pytest invokes
    a hook of this name from tests/conftest.py needs no separate proof -- if it
    did not, the fixtures every other test depends on would not work either.
    """
    from tests.conftest import pytest_collection_modifyitems

    item = cast(
        "pytest.Item",
        SimpleNamespace(
            nodeid="tests/example.py::test_case",
            get_closest_marker=lambda name: pytest.mark.timeout(0).mark if name == "timeout" else None,
        ),
    )

    with pytest.raises(pytest.UsageError, match="0 < seconds <= 900; got 0"):
        pytest_collection_modifyitems([item])
