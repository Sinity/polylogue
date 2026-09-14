"""A judgment recorded through the daemon is the judgment that was elicited.

``mutation.judgment.record`` carries a comparative judgment over a socket, so
the surface that elicits it and the handler that writes the durable row see two
different objects. These tests are the law that the two are the same object:
every field survives the round trip, both verdict shapes survive it, and a body
that lost a field fails loudly rather than writing a weaker row.

Anti-vacuity: drop any key from ``comparative_judgment_wire_form``'s dict, or
stop reading one in ``comparative_judgment_from_wire_form``, and
:func:`test_wire_form_round_trip_preserves_every_field` goes red -- it compares
the whole reconstructed dataclass, not a chosen subset. Coerce an ordering
verdict to a scalar and
:func:`test_ordering_verdict_survives_as_an_ordering` goes red.
"""

from __future__ import annotations

import pytest

from polylogue.analysis.judgment.types import ComparativeJudgment, JudgeIdentity
from polylogue.core.enums import ComparativeVerdict
from polylogue.operations.judgment_wire import (
    comparative_judgment_from_wire_form,
    comparative_judgment_wire_form,
)


def _judge() -> JudgeIdentity:
    return JudgeIdentity(actor_ref="user:local", execution_context_id="cli:compare", role="judge")


def _pairwise() -> ComparativeJudgment:
    return ComparativeJudgment(
        judgment_id="judgment-1",
        items=("ref:left", "ref:right"),
        dimension="quality",
        verdict=ComparativeVerdict.PREFER_LEFT,
        judge=_judge(),
        blinded=True,
        rubric_id="rubric-a",
        rubric_version=3,
        evidence_refs=("session:one", "session:two"),
        elicitation_ref="elicitation-9",
        rationale="left is clearer",
        rationale_visible=True,
        decided_at_ms=1_700_000_000_000,
    )


def test_wire_form_round_trip_preserves_every_field() -> None:
    """The reconstructed judgment equals the elicited one, field for field."""
    original = _pairwise()

    assert comparative_judgment_from_wire_form(comparative_judgment_wire_form(original)) == original


def test_wire_form_is_json_shaped() -> None:
    """Only JSON scalars and containers cross the socket."""
    body = comparative_judgment_wire_form(_pairwise())

    def _json_shaped(value: object) -> bool:
        if isinstance(value, dict):
            return all(isinstance(key, str) and _json_shaped(item) for key, item in value.items())
        if isinstance(value, list):
            return all(_json_shaped(item) for item in value)
        return value is None or isinstance(value, str | int | float | bool)

    assert _json_shaped(body), body
    assert body["verdict"] == "prefer_left"


def test_ordering_verdict_survives_as_an_ordering() -> None:
    """An n-wise ordering must not collapse into a scalar verdict token."""
    original = ComparativeJudgment(
        judgment_id="judgment-2",
        items=("a", "b", "c"),
        dimension="quality",
        verdict=("c", "a", "b"),
        judge=_judge(),
        blinded=False,
        rubric_id="rubric-b",
        rubric_version=1,
    )

    restored = comparative_judgment_from_wire_form(comparative_judgment_wire_form(original))

    assert restored == original
    assert restored.is_ordering


def test_a_body_that_lost_a_required_field_is_refused() -> None:
    """A truncated body fails here, never as a weaker durable row."""
    body = comparative_judgment_wire_form(_pairwise())
    del body["dimension"]

    with pytest.raises(KeyError):
        comparative_judgment_from_wire_form(body)


def test_invariants_are_rechecked_on_the_handler_side() -> None:
    """The dataclass's own invariants gate the durable write, not just the CLI."""
    body = comparative_judgment_wire_form(_pairwise())
    body["items"] = ["ref:left"]

    with pytest.raises(ValueError, match="at least 2 items"):
        comparative_judgment_from_wire_form(body)
