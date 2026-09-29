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


def test_independently_observed_equal_scope_uses_the_same_hash_domain() -> None:
    """Distinct request/result hash namespaces made every genuine scope check fail."""
    requested = request_scope_fingerprint({"session_ids": ["one"], "origin": "codex-session"})
    applied = result_scope_fingerprint({"origin": "codex-session", "session_ids": ["one"]})
    assert requested == applied
    assert_scope_match(requested, applied)
    changed = result_scope_fingerprint({"origin": "codex-session", "session_ids": ["two"]})
    with pytest.raises(ScopeMismatchError):
        assert_scope_match(requested, changed)
