"""Mutation-sensitive tests for the definition closure kernel."""

from typing import cast

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
        definitions=policy.definitions,
        intentional_absences={"event:append": "bead:event"},
    )
    absent = evaluate((absent_policy,), {}).rows[0]
    assert absent.status is ClosureStatus.INTENTIONAL_ABSENCE
    assert absent.exception_authority == "bead:event"
