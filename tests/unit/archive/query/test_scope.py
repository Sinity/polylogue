from __future__ import annotations

import pytest

from polylogue.archive.query.scope import (
    ScopeMismatchError,
    SurfaceSpec,
    assert_scope_match,
    request_scope_fingerprint,
    result_scope_fingerprint,
)


def test_pushdown_requires_an_executable_lowerer() -> None:
    with pytest.raises(ValueError, match="without a lowerer"):
        SurfaceSpec("count", "session", True)


def test_scope_fingerprints_have_independent_derivation_inputs() -> None:
    request = request_scope_fingerprint({"session_id": "one"})
    applied = result_scope_fingerprint({"session_ids": ["one"]})
    assert request != applied
    with pytest.raises(ScopeMismatchError):
        assert_scope_match(request, applied)


def test_matching_scope_is_explicitly_accepted() -> None:
    fingerprint = request_scope_fingerprint({"session_id": "one"})
    assert_scope_match(fingerprint, fingerprint)
