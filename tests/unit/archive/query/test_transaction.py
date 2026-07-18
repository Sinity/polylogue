from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.query.transaction import (
    QueryContinuation,
    QueryContinuationStaleError,
    QueryTransactionRequest,
    archive_index_epoch,
    validate_continuation_epoch,
)


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
    assert envelope.result_ref == transaction.result_ref


def _make_index_db(path: Path, *, user_version: int = 24) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version = {user_version}")
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, updated_at_ms INTEGER)")
        conn.commit()


def _touch_session(path: Path, *, session_id: str, updated_at_ms: int) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, updated_at_ms) VALUES (?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET updated_at_ms = excluded.updated_at_ms",
            (session_id, updated_at_ms),
        )
        conn.commit()


def test_archive_index_epoch_is_absent_marker_when_index_missing(tmp_path: Path) -> None:
    assert archive_index_epoch(tmp_path / "index.db") == "index:absent"


def test_archive_index_epoch_reflects_schema_version_and_session_watermark(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    _make_index_db(index_db, user_version=24)

    empty_epoch = archive_index_epoch(index_db)
    assert empty_epoch == "index:v24:0:0:0"

    _touch_session(index_db, session_id="claude-code-session:s1", updated_at_ms=1_000)
    after_first_write = archive_index_epoch(index_db)
    assert after_first_write != empty_epoch
    assert after_first_write == "index:v24:1:1:1000"

    _touch_session(index_db, session_id="claude-code-session:s2", updated_at_ms=2_000)
    after_second_write = archive_index_epoch(index_db)
    assert after_second_write != after_first_write
    assert after_second_write == "index:v24:2:2:2000"

    # A later materialization stage backfilling updated_at_ms on an existing
    # session (count/rowid unchanged) still moves the epoch via the watermark.
    _touch_session(index_db, session_id="claude-code-session:s1", updated_at_ms=3_000)
    after_backfill = archive_index_epoch(index_db)
    assert after_backfill == "index:v24:2:2:3000"
    assert after_backfill != after_second_write


def test_result_ref_binds_query_identity_to_declared_archive_epoch() -> None:
    def _request(archive_epoch: str) -> QueryTransactionRequest:
        return QueryTransactionRequest(
            operation="query_units",
            arguments={"expression": "actions where tool:Workflow", "session_filters": {}},
            page_size=10,
            projection="terminal-unit-envelope",
            stable_order="canonical",
            archive_epoch=archive_epoch,
        )

    at_epoch_one = _request("index:v24:1000")
    at_epoch_one_again = _request("index:v24:1000")
    at_epoch_two = _request("index:v24:2000")

    # Same logical query at the same frame -> same result identity.
    assert at_epoch_one.query_ref == at_epoch_two.query_ref  # query identity excludes epoch
    assert at_epoch_one.result_ref == at_epoch_one_again.result_ref
    # The archive moved between the two requests -> different result identity,
    # even though the logical query (query_ref) did not change.
    assert at_epoch_one.result_ref != at_epoch_two.result_ref


def test_continuation_round_trip_preserves_archive_epoch() -> None:
    request = QueryTransactionRequest(
        operation="query_units",
        arguments={"expression": "actions where tool:Workflow", "session_filters": {}},
        page_size=10,
        archive_epoch="index:v24:1000",
    )
    token = QueryContinuation(request=request, result_ref=request.result_ref).encode()

    decoded = QueryContinuation.decode(token)

    assert decoded.request.archive_epoch == "index:v24:1000"


def test_validate_continuation_epoch_accepts_matching_frame(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    _make_index_db(index_db)
    _touch_session(index_db, session_id="s1", updated_at_ms=500)
    current = archive_index_epoch(index_db)

    request = QueryTransactionRequest(operation="query_units", arguments={}, page_size=1, archive_epoch=current)
    validate_continuation_epoch(request, archive_root=tmp_path)  # must not raise


def test_validate_continuation_epoch_rejects_drifted_frame(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    _make_index_db(index_db)
    _touch_session(index_db, session_id="s1", updated_at_ms=500)
    issued_epoch = archive_index_epoch(index_db)

    # The archive admits a new session between the page that issued this
    # continuation and the page that resumes it.
    _touch_session(index_db, session_id="s2", updated_at_ms=999)

    request = QueryTransactionRequest(operation="query_units", arguments={}, page_size=1, archive_epoch=issued_epoch)
    with pytest.raises(QueryContinuationStaleError) as raised:
        validate_continuation_epoch(request, archive_root=tmp_path)
    assert raised.value.code == "query_continuation_stale"
    assert raised.value.issued_epoch == issued_epoch
    assert raised.value.current_epoch == archive_index_epoch(index_db)


def test_validate_continuation_epoch_skips_check_for_unframed_continuation(tmp_path: Path) -> None:
    """A continuation minted before epoch-binding existed (empty epoch) is not rejected."""
    request = QueryTransactionRequest(operation="query_units", arguments={}, page_size=1, archive_epoch="")
    # archive_root does not even exist here; an epoch-aware check would still
    # not raise because there is nothing declared to have drifted from.
    validate_continuation_epoch(request, archive_root=tmp_path / "does-not-exist")


@pytest.mark.parametrize("token", ["q1._", "q1.////", "q1.8J+SqQ"])
def test_query_continuation_rejects_malformed_payloads_as_value_errors(token: str) -> None:
    with pytest.raises(ValueError, match="invalid query continuation"):
        QueryContinuation.decode(token)
