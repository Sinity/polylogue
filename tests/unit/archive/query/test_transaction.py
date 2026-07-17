from __future__ import annotations

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
