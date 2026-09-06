"""The DSL field vocabulary stays bound to the selection it lowers to.

``EXPRESSION_FIELD_REGISTRY`` owns the query tokens the parser accepts, the
completions offer, and the syntax help documents.  ``SessionQuerySpec`` owns
the selection those tokens lower to.  The only link is each entry's
``spec_field`` string, so nothing but this check stops a spec rename from
leaving a token that still completes and still parses and selects nothing.

Anti-vacuity: :func:`test_a_renamed_spec_field_is_reported` renames a spec
field out from under a live registry entry and requires the drift report to
name it.
"""

from __future__ import annotations

import dataclasses

import pytest

from polylogue.archive.query.metadata import EXPRESSION_FIELD_REGISTRY, expression_registry_drift
from polylogue.archive.query.spec import SessionQuerySpec


def test_every_dsl_token_lowers_to_a_real_spec_field() -> None:
    """No shipped query token names a selection field that does not exist."""
    assert expression_registry_drift() == ()


def test_every_dsl_token_declares_where_it_lowers() -> None:
    """A token without a ``spec_field`` has no checkable binding at all."""
    undeclared = sorted(token for token, info in EXPRESSION_FIELD_REGISTRY.items() if not info.get("spec_field"))
    assert undeclared == []


def test_a_renamed_spec_field_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Renaming a spec field out from under a live token is reported as drift."""
    token = "repo"
    lowered = EXPRESSION_FIELD_REGISTRY[token]["spec_field"]
    surviving = tuple(field for field in dataclasses.fields(SessionQuerySpec) if field.name != lowered)
    assert len(surviving) < len(dataclasses.fields(SessionQuerySpec)), f"{lowered!r} is not a spec field"
    monkeypatch.setattr(dataclasses, "fields", lambda obj: surviving if obj is SessionQuerySpec else ())
    reported = expression_registry_drift()
    assert any(token in entry and lowered in entry for entry in reported), reported
