"""Query-field completions are generated from the declaration registry.

``EXPRESSION_FIELD_REGISTRY`` declares the query tokens; ``query_field_candidates``
reads that registry rather than a parallel hand-written completion list. A copied
list is what drifts: a field gains a declaration, the parser accepts it, and it
never appears in a completion because a second inventory was never updated.

The two completion syntaxes partition the vocabulary rather than duplicating it
-- ``compact`` omits the count/numeric fields that only read well in the
``session-boolean`` form -- so coverage is asserted over their union.

Anti-vacuity: :func:`test_a_new_declaration_completes_without_touching_a_list`
declares a field that exists in no completion source and requires it to be
offered anyway; replacing the registry iteration with any literal field list
makes it red.
"""

from __future__ import annotations

import pytest

from polylogue.archive.query.completions import QueryFieldCompletionSyntax, query_field_candidates
from polylogue.archive.query.metadata import EXPRESSION_FIELD_REGISTRY

_SYNTAXES: tuple[QueryFieldCompletionSyntax, ...] = ("compact", "session-boolean")


def _completed_fields(incomplete: str = "") -> set[str]:
    """Every field token the generated completions offer, across both syntaxes."""
    return {candidate.value for syntax in _SYNTAXES for candidate in query_field_candidates(incomplete, syntax=syntax)}


def test_every_declared_field_is_completable() -> None:
    """A declared token nothing completes is a token users cannot discover."""
    assert sorted(set(EXPRESSION_FIELD_REGISTRY) - _completed_fields()) == []


def test_a_new_declaration_completes_without_touching_a_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declaring a field is the whole edit: no completion source names it."""
    field = "synthetic_declared_probe"
    assert field not in EXPRESSION_FIELD_REGISTRY
    monkeypatch.setitem(
        EXPRESSION_FIELD_REGISTRY,
        field,
        {
            "description": "Synthetic declaration used to prove completions are generated",
            "spec_field": "title",
            "negatable": "no",
            "example": f"{field}:value",
        },
    )

    assert field in _completed_fields()


def test_a_new_declaration_is_offered_under_its_own_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefix filtering reads the declaration too, not a static candidate set."""
    field = "synthetic_declared_probe"
    monkeypatch.setitem(
        EXPRESSION_FIELD_REGISTRY,
        field,
        {
            "description": "Synthetic declaration used to prove completions are generated",
            "spec_field": "title",
            "negatable": "no",
            "example": f"{field}:value",
        },
    )

    offered = [candidate for candidate in query_field_candidates("synthetic_") if candidate.value == field]
    assert [candidate.source for candidate in offered] == ["EXPRESSION_FIELD_REGISTRY"]
