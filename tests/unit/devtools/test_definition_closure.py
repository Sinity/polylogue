"""Mutation-sensitive tests for the definition closure kernel."""

from typing import Any, cast

import pytest

from devtools.definition_closure import (
    ClosurePolicy,
    ClosureStatus,
    Definition,
    EdgeKind,
    EvidenceRef,
    evaluate,
)


def _policy() -> ClosurePolicy:
    return ClosurePolicy(
        "event",
        "polylogue.events.registry",
        (EdgeKind.PRODUCER, EdgeKind.CONSUMER, EdgeKind.LIFECYCLE),
        definitions=(Definition("event:append", (EdgeKind.PRODUCER, EdgeKind.CONSUMER, EdgeKind.LIFECYCLE)),),
        exception_authority="bead:event",
    )


def _evidence(
    *, source: str = "production", route: str | None = None, twin: str | None = None
) -> dict[str, dict[EdgeKind, tuple[EvidenceRef, ...]]]:
    return {
        "event:append": {
            edge: (EvidenceRef(f"src:{edge.value}", source=source, route=route, twin=twin),)
            for edge in (EdgeKind.PRODUCER, EdgeKind.CONSUMER, EdgeKind.LIFECYCLE)
        }
    }


def test_complete_policy_emits_durable_matrix() -> None:
    graph = evaluate((_policy(),), _evidence())
    assert graph.ok
    assert graph.rows[0].status is ClosureStatus.SATISFIED
    payload = graph.to_dict()
    assert payload["inventory_counts"] == {"event": 1}
    rows = payload["rows"]
    assert isinstance(rows, list)
    first_row = cast(dict[str, object], rows[0])
    assert first_row["evidence_refs"] == ["src:producer", "src:consumer", "src:lifecycle"]


def test_missing_edge_names_definition_and_edge() -> None:
    evidence = _evidence()
    del evidence["event:append"][EdgeKind.CONSUMER]
    row = evaluate((_policy(),), evidence).rows[0]
    assert row.status is ClosureStatus.MISSING
    assert row.diagnostic == "event:append: missing edge consumer"


def test_tests_only_bypass_and_divergent_twins_are_not_satisfied() -> None:
    assert evaluate((_policy(),), _evidence(source="tests")).rows[0].status is ClosureStatus.TESTS_ONLY
    assert evaluate((_policy(),), _evidence(route="bypass")).rows[0].status is ClosureStatus.BYPASS
    assert evaluate((_policy(),), _evidence(twin="legacy-operation")).rows[0].status is ClosureStatus.DIVERGENT_TWIN


def test_unavailable_evidence_and_authorized_absence_remain_explicit() -> None:
    policy = _policy()
    unavailable = evaluate((policy,), {}, evidence_available=False).rows[0]
    assert unavailable.status is ClosureStatus.UNAVAILABLE
    absent_policy = ClosurePolicy(
        policy.family,
        policy.authoritative_inventory_ref,
        policy.required_edge_kinds,
        exception_authority="bead:event",
        definitions=policy.definitions,
        intentional_absences={"event:append": "bead:event"},
    )
    absent = evaluate((absent_policy,), {}).rows[0]
    assert absent.status is ClosureStatus.INTENTIONAL_ABSENCE
    assert absent.exception_authority == "bead:event"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"intentional_absences": {"event:unknown": "bead:event"}}, "unknown definition"),
        ({"intentional_absences": {"event:append": ""}}, "cannot be empty"),
        ({"intentional_absences": {"event:append": "bead:event"}, "exception_authority": None}, "exception authority"),
    ],
)
def test_intentional_absence_requires_known_explicit_authority(kwargs: dict[str, object], message: str) -> None:
    base: dict[str, Any] = {
        "family": "event",
        "authoritative_inventory_ref": "polylogue.events.registry",
        "required_edge_kinds": (EdgeKind.PRODUCER,),
        "definitions": (Definition("event:append", (EdgeKind.PRODUCER,)),),
        "exception_authority": "bead:event",
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ClosurePolicy(**base)


def test_matrix_reports_edge_totals_and_unresolved_rows() -> None:
    evidence = _evidence()
    del evidence["event:append"][EdgeKind.CONSUMER]
    payload = evaluate((_policy(),), evidence).to_dict()
    assert payload["required_edge_count"] == 3
    assert payload["actual_edge_count"] == 2
    assert payload["unresolved_rows"] == ["event:append"]


def test_matrix_exposes_authorized_exceptions() -> None:
    policy = ClosurePolicy(
        "event",
        "polylogue.events.registry",
        (EdgeKind.PRODUCER,),
        definitions=(Definition("event:append", (EdgeKind.PRODUCER,)),),
        exception_authority="bead:event",
        intentional_absences={"event:append": "bead:event"},
    )
    payload = evaluate((policy,), {}).to_dict()
    assert payload["exceptions"] == [{"family": "event", "definition_ref": "event:append", "authority": "bead:event"}]


def test_graph_limits_fail_explicitly_instead_of_dropping_rows_or_edges() -> None:
    too_many = tuple(Definition(f"event:{index}", (EdgeKind.PRODUCER,)) for index in range(1025))
    with pytest.raises(ValueError, match="bounded definition limit"):
        evaluate(
            (
                ClosurePolicy(
                    "event",
                    "polylogue.events.registry",
                    (EdgeKind.PRODUCER,),
                    definitions=too_many,
                ),
            )
        )

    policy = ClosurePolicy(
        "event",
        "polylogue.events.registry",
        (EdgeKind.PRODUCER,),
        definitions=(Definition("event:append", (EdgeKind.PRODUCER,)),),
    )
    evidence = {"event:append": {EdgeKind.PRODUCER: tuple(EvidenceRef(str(i)) for i in range(33))}}
    with pytest.raises(ValueError, match="bounded edge limit"):
        evaluate((policy,), evidence)


def test_unknown_evidence_edge_is_not_silently_discarded() -> None:
    with pytest.raises(ValueError, match="unknown evidence edge"):
        evaluate((_policy(),), {"event:append": {"future-edge": ()}})  # type: ignore[dict-item]
