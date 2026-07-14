"""Losslessness of ``predicate_from_payload`` against every predicate's own ``to_payload()``.

``polylogue.archive.query.production_evaluator`` depends on this round trip
to reconstruct an executable predicate from a durable ``query:<hash>``
definition without reverse-compiling arbitrary/lossy text. If any predicate
variant's payload shape drifts from its reconstructor (a field renamed on one
side but not the other), these tests fail immediately instead of surfacing as
a silent evaluator misbehavior later.
"""

from __future__ import annotations

import pytest

from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryExistsPredicate,
    QueryFieldPredicate,
    QueryFieldRef,
    QueryLineagePredicate,
    QueryNotPredicate,
    QueryPredicate,
    QuerySemanticPredicate,
    QuerySequenceConstraint,
    QuerySequencePredicate,
    QueryTextPredicate,
    predicate_from_payload,
)

_ROUNDTRIP_CASES: tuple[QueryPredicate, ...] = (
    QueryFieldPredicate(field="origin", values=("codex-session",), op="="),
    QueryFieldPredicate(field="origin", values=("codex-session",), op="=").with_field_ref(
        QueryFieldRef(scope="session", name="origin", source_name="origin")
    ),
    QueryFieldPredicate(field="count", values=("3",), op=">=").with_field_ref(
        QueryFieldRef(scope="unit", name="count", source_name="count", unit="message")
    ),
    QueryNotPredicate(QueryFieldPredicate(field="origin", values=("codex-session",), op="=")),
    QueryBoolPredicate(
        "and",
        (
            QueryFieldPredicate(field="origin", values=("codex-session",), op="="),
            QueryFieldPredicate(field="repo", values=("polylogue",), op="="),
        ),
    ),
    QueryBoolPredicate(
        "or",
        (
            QueryFieldPredicate(field="origin", values=("codex-session",), op="="),
            QueryNotPredicate(QueryFieldPredicate(field="repo", values=("polylogue",), op="=")),
        ),
    ),
    QueryExistsPredicate(unit="block", child=QueryFieldPredicate(field="tool_name", values=("Bash",), op="=")),
    QuerySequencePredicate(action_terms=("plan", "edit", "test")),
    QuerySequencePredicate(
        steps=(
            QueryFieldPredicate(field="action", values=("plan",), op="="),
            QueryFieldPredicate(field="action", values=("edit",), op="="),
        ),
        constraints=(QuerySequenceConstraint(kind="within", within_ms=60_000),),
    ),
    QueryTextPredicate(text="deploy with caveats"),
    QuerySemanticPredicate(text="deploy with caveats"),
    QueryLineagePredicate(seed_session_id="codex-session:abc123"),
)


@pytest.mark.parametrize("predicate", _ROUNDTRIP_CASES, ids=lambda p: type(p).__name__)
def test_payload_roundtrip_is_lossless(predicate: QueryPredicate) -> None:
    payload = predicate.to_payload()
    reconstructed = predicate_from_payload(payload)
    assert reconstructed == predicate
    # The reconstruction must also re-serialize to the identical payload, not
    # merely compare equal as a dataclass (catches asymmetric defaults).
    assert reconstructed.to_payload() == payload


def test_unsupported_predicate_kind_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported predicate payload kind"):
        predicate_from_payload({"kind": "made-up"})


def test_field_predicate_rejects_unsupported_op() -> None:
    with pytest.raises(ValueError, match="unsupported field predicate op"):
        predicate_from_payload({"kind": "field", "field": "origin", "op": "!=", "values": ["x"]})


def test_exists_predicate_rejects_unsupported_unit() -> None:
    with pytest.raises(ValueError, match="unsupported exists unit"):
        predicate_from_payload({"kind": "exists", "unit": "made-up", "child": {"kind": "fts", "text": "x"}})


def test_boolean_predicate_requires_children_list() -> None:
    with pytest.raises(ValueError, match="'children'"):
        predicate_from_payload({"kind": "and", "children": "not-a-list"})


def test_not_predicate_requires_object_child() -> None:
    with pytest.raises(ValueError, match="'child'"):
        predicate_from_payload({"kind": "not", "child": "not-an-object"})
