from __future__ import annotations

import pytest

from polylogue.archive.query.transaction import QueryContinuation, QueryTransactionRequest


def test_query_continuation_round_trips_complete_request_state() -> None:
    request = QueryTransactionRequest(
        operation="query_units",
        arguments={"expression": "actions where tool:Workflow", "repo": "polylogue"},
        page_size=7,
        offset=14,
        projection="compact-action",
        stable_order="session,message,block",
    )
    token = QueryContinuation(request=request, result_ref="result:abc123").encode()

    decoded = QueryContinuation.decode(token)

    assert decoded.result_ref == "result:abc123"
    assert decoded.request == request
    assert decoded.request.next(offset=21).arguments == request.arguments


def test_query_ref_is_logical_and_independent_of_page_offset() -> None:
    first = QueryTransactionRequest("search", {"query": "Workflow"}, page_size=10, offset=0)
    later = first.next(offset=10)

    assert first.query_ref == later.query_ref


def test_query_unit_envelope_can_adopt_the_caller_transaction_identity() -> None:
    """The page protocol does not mint a second identity behind an adapter."""
    from polylogue.archive.query.expression import parse_unit_source_expression
    from polylogue.archive.query.unit_results import QueryUnitRequest, query_unit_envelope

    source = parse_unit_source_expression("actions where tool:Workflow")
    assert source is not None
    transaction = QueryTransactionRequest(
        operation="query_units",
        arguments={"expression": "actions where tool:Workflow", "session_filters": {}},
        page_size=1,
        projection="terminal-unit-envelope",
    )

    # A tiny fake archive keeps this a protocol test: the production behavior
    # exercised is identity ownership, not a duplicate action-query replica.
    class _Archive:
        def query_actions(self, *_args: object, **_kwargs: object) -> list[object]:
            return []

    envelope = query_unit_envelope(
        _Archive(),  # type: ignore[arg-type]
        QueryUnitRequest(expression="actions where tool:Workflow", source=source, limit=1),
        transaction_request=transaction,
    )

    assert envelope.query_ref == transaction.query_ref
    assert envelope.result_ref == "result:" + transaction.query_ref.removeprefix("query:")


@pytest.mark.parametrize("token", ["q1._", "q1.////", "q1.8J+SqQ"])
def test_query_continuation_rejects_malformed_payloads_as_value_errors(token: str) -> None:
    with pytest.raises(ValueError, match="invalid query continuation"):
        QueryContinuation.decode(token)
