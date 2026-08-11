"""Runtime policy for explicit pytest timeout markers."""

from __future__ import annotations

from typing import Any

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
