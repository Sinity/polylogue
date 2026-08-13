"""Collection-only witness for the repository timeout-policy hook."""

from __future__ import annotations

import pytest


@pytest.mark.timeout(0)
def test_zero_timeout_must_not_collect() -> None:
    raise AssertionError("the repository collection hook should reject this node")
