"""Runtime policy for explicit pytest timeout markers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.infra.timeout_policy import timeout_marker_error

pytest_plugins = ("pytester",)

_INVALID_MARKER_CASE = Path(__file__).with_name("timeout_policy_cases") / "invalid_zero_timeout.py"


@pytest.mark.parametrize("value", [None, 0, -1, float("inf"), 901, "30"])
def test_collection_rejects_unbounded_timeout_markers(value: Any) -> None:
    marker = pytest.mark.timeout(value).mark
    assert "0 < seconds <= 900" in (timeout_marker_error(marker) or "")


@pytest.mark.parametrize("value", [0.1, 30, 120, 900])
def test_collection_accepts_bounded_timeout_markers(value: float) -> None:
    marker = pytest.mark.timeout(value).mark
    assert timeout_marker_error(marker) is None


def test_repository_collection_hook_rejects_zero_timeout(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest_subprocess(str(_INVALID_MARKER_CASE), "--collect-only", "-q")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*timeout marker must be finite and within 0 < seconds <= 900; got 0*"])
