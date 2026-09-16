"""polylogue-01fe: one bad origin, one typed answer, on every surface.

The CLI rejected an unknown ``--origin`` with a UsageError naming the valid
origins; MCP now rejects it with ``invalid_argument``; the daemon's
``?origin=`` went straight onto the lenient wire-token normalizer and answered
HTTP 200 with ``total: 0``. A caller who mistyped an origin -- or wrote the
near-miss ``claude-code`` instead of ``claude-code-session`` -- was told "no
data" rather than "no such origin".

The lenient ``Origin`` constructor is a different contract: its job is
normalizing untrusted provider-export tokens to ``unknown-export``, never
gating user input. Conflating the two is what produced the silent empty.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from polylogue.archive.query.spec import QuerySpecError
from polylogue.operations.origin_filters import public_origin_filter_tokens, unknown_origin_filter_tokens


def test_shared_validator_names_unknown_and_near_miss_tokens() -> None:
    """Anti-vacuity: returning ``()`` unconditionally makes this red.

    ``claude-code`` is the near miss the bead names -- a real-looking token
    that is not a public origin -- and it must be rejected, not silently
    filtered to nothing.
    """
    assert unknown_origin_filter_tokens(["bogus-origin", "claude-code"]) == ("bogus-origin", "claude-code")


def test_shared_validator_accepts_every_declared_public_origin() -> None:
    """The gate must not reject tokens the public vocabulary declares.

    Anti-vacuity: a validator built from a hand-copied list instead of
    ``public_origin_filter_tokens`` drifts and turns this red.
    """
    assert unknown_origin_filter_tokens(public_origin_filter_tokens()) == ()


def test_http_query_params_reject_an_unknown_origin_as_a_typed_400() -> None:
    """The daemon's ``?origin=`` answers with the same error class, not an empty page.

    ``QuerySpecError`` carries ``http_status_code=400`` and
    ``daemon_safe_handler`` renders it as the shared QueryErrorPayload 400.

    Anti-vacuity: removing the guard from ``_build_query_spec_params`` makes
    this build a spec and return without raising.
    """
    from polylogue.daemon.http import _build_query_spec_params

    with pytest.raises(QuerySpecError) as excinfo:
        _build_query_spec_params({"origin": ["bogus-origin"]}, None)  # type: ignore[arg-type]
    assert "bogus-origin" in str(excinfo.value)
    assert getattr(excinfo.value, "http_status_code", None) == 400


def test_http_query_params_reject_an_unknown_exclude_origin() -> None:
    """``?exclude_origin=`` is the same input class and gets the same answer.

    An unrecognized exclusion that silently excludes nothing is the same
    "told no data" failure in reverse.
    """
    from polylogue.daemon.http import _build_query_spec_params

    with pytest.raises(QuerySpecError):
        _build_query_spec_params({"exclude_origin": ["bogus-origin"]}, None)  # type: ignore[arg-type]


def test_http_query_params_accept_a_declared_origin() -> None:
    """A valid origin must still build a spec, or the gate is vacuous."""
    from polylogue.daemon.http import DaemonAPIHandler, _build_query_spec_params

    valid = public_origin_filter_tokens()[0]
    # A stub covering exactly the three accessors the builder uses for the
    # params it is not given, so the assertion is about the origin path.
    handler = cast(
        "DaemonAPIHandler",
        SimpleNamespace(
            _get_param=lambda params, key: None,
            _get_bool=lambda params, key: False,
            _get_int=lambda params, key, default=0: default,
        ),
    )
    built = _build_query_spec_params({"origin": [valid]}, handler)
    assert built["origin"] == (valid,)
