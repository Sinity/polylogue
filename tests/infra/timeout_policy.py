"""Runtime validation for explicit pytest timeout markers."""

from __future__ import annotations

import math

import pytest

MAX_EXPLICIT_TEST_TIMEOUT_S = 900.0


def timeout_marker_error(marker: pytest.Mark) -> str | None:
    """Return a diagnostic when a resolved marker disables the test bound."""
    raw_timeout = marker.kwargs.get("timeout")
    if raw_timeout is None and marker.args:
        raw_timeout = marker.args[0]
    if (
        isinstance(raw_timeout, bool)
        or not isinstance(raw_timeout, (int, float))
        or not math.isfinite(float(raw_timeout))
        or float(raw_timeout) <= 0
        or float(raw_timeout) > MAX_EXPLICIT_TEST_TIMEOUT_S
    ):
        return f"timeout marker must be finite and within 0 < seconds <= 900; got {raw_timeout!r}"
    return None
