from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.query.transaction import (
    QueryContinuation,
    QueryContinuationStaleError,
    QueryTransactionRequest,
    archive_snapshot_epoch,
    query_units_transaction_request,
    validate_continuation_epoch,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def test_query_continuation_round_trips_complete_request_state() -> None:
    request = QueryTransactionRequest(
        operation="query_units",
        arguments={"expression": "actions where tool:Workflow", "repo": "polylogue"},
        page_size=7,
        offset=14,
        projection="compact-action",
        stable_order="session,message,block",
        archive_epoch="archive:v1:index:v40:7:user:v10:3",
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


def test_query_unit_envelope_fallback_matches_canonical_transaction_identity(tmp_path: Path) -> None:
    """Direct storage and adapter calls share terminal-unit-envelope identity.

    Production dependencies: ``query_unit_envelope`` and the public request
    builder. Removing the fallback builder or its terminal projection makes
    the two routes mint different result references.
    """
    from polylogue.archive.query.expression import parse_unit_source_expression
    from polylogue.archive.query.unit_results import QueryUnitRequest, query_unit_envelope

    source = parse_unit_source_expression("actions where tool:Workflow")
    assert source is not None
    with ArchiveStore(tmp_path) as writer:
        writer.close()
    with ArchiveStore.open_existing(tmp_path) as archive:
        archive.begin_read_snapshot()
        try:
            envelope = query_unit_envelope(
                archive,
                QueryUnitRequest(expression="actions where tool:Workflow", source=source, limit=1),
            )
            canonical = query_units_transaction_request(
                expression="actions where tool:Workflow", session_filters={}, page_size=1
            ).with_archive_epoch(archive_snapshot_epoch(archive))
        finally:
            archive.end_read_snapshot()

    assert envelope.query_ref == canonical.query_ref
    assert envelope.result_ref == canonical.result_ref


def _seed_frame_session(index_db: Path, *, native_id: str, title: str) -> None:
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "INSERT INTO sessions(native_id, origin, title, content_hash) VALUES (?, 'codex-session', ?, zeroblob(32))",
            (native_id, title),
        )


def _snapshot_epoch(root: Path) -> str:
    with ArchiveStore.open_existing(root) as archive:
        archive.begin_read_snapshot()
        try:
            return archive_snapshot_epoch(archive)
        finally:
            archive.end_read_snapshot()


def test_archive_snapshot_epoch_changes_for_nonmaximum_session_rewrite(tmp_path: Path) -> None:
    """A rewrite need not win MAX(updated_at_ms) to invalidate a page."""
    with ArchiveStore(tmp_path) as archive:
        archive.close()
    _seed_frame_session(tmp_path / "index.db", native_id="first", title="first")
    _seed_frame_session(tmp_path / "index.db", native_id="last", title="last")
    issued = _snapshot_epoch(tmp_path)

    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE sessions SET title = 'rewritten first' WHERE native_id = 'first'")

    assert _snapshot_epoch(tmp_path) != issued


def test_archive_snapshot_epoch_changes_for_user_assertion_mutation(tmp_path: Path) -> None:
    """User-tier tag/assertion state is part of a query-unit archive frame."""
    with ArchiveStore(tmp_path) as archive:
        archive.close()
    issued = _snapshot_epoch(tmp_path)
    with sqlite3.connect(tmp_path / "user.db") as conn:
        conn.execute(
            """INSERT INTO assertions(assertion_id, target_ref, kind, created_at_ms, updated_at_ms)
               VALUES ('tag:frame', 'session:codex-session:first', 'tag', 1, 1)"""
        )

    assert _snapshot_epoch(tmp_path) != issued


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
        archive_epoch="archive:v1:index:v40:1000:user:v10:4",
    )
    token = QueryContinuation(request=request, result_ref=request.result_ref).encode()

    decoded = QueryContinuation.decode(token)

    assert decoded.request.archive_epoch == "archive:v1:index:v40:1000:user:v10:4"
    assert decoded.request.continuation_version == 2


def test_validate_continuation_epoch_accepts_matching_snapshot_frame(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as writer:
        writer.close()
    current = _snapshot_epoch(tmp_path)
    request = QueryTransactionRequest(operation="query_units", arguments={}, page_size=1, archive_epoch=current)
    with ArchiveStore.open_existing(tmp_path) as archive:
        archive.begin_read_snapshot()
        try:
            assert validate_continuation_epoch(request, archive=archive) == current
        finally:
            archive.end_read_snapshot()


def test_validate_continuation_epoch_rejects_drifted_snapshot_frame(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as writer:
        writer.close()
    issued_epoch = _snapshot_epoch(tmp_path)
    _seed_frame_session(tmp_path / "index.db", native_id="next", title="next")
    request = QueryTransactionRequest(operation="query_units", arguments={}, page_size=1, archive_epoch=issued_epoch)
    with ArchiveStore.open_existing(tmp_path) as archive:
        archive.begin_read_snapshot()
        try:
            with pytest.raises(QueryContinuationStaleError) as raised:
                validate_continuation_epoch(request, archive=archive)
            current = archive_snapshot_epoch(archive)
        finally:
            archive.end_read_snapshot()
    assert raised.value.code == "query_continuation_stale"
    assert raised.value.issued_epoch == issued_epoch
    assert raised.value.current_epoch == current


def test_only_explicit_q1_tokens_can_bypass_empty_epoch(tmp_path: Path) -> None:
    """Deleting ``archive_epoch`` from a current token cannot restore legacy behavior."""
    request = QueryTransactionRequest(operation="query_units", arguments={}, page_size=1, archive_epoch="frame")
    token = QueryContinuation(request=request, result_ref=request.result_ref).encode()
    raw = json.loads(base64.urlsafe_b64decode(token[3:] + "=" * (-len(token[3:]) % 4)))
    del raw["request"]["archive_epoch"]
    tampered = "q2." + base64.urlsafe_b64encode(json.dumps(raw).encode()).decode().rstrip("=")
    with pytest.raises(ValueError, match="invalid query continuation"):
        QueryContinuation.decode(tampered)

    legacy = dict(raw)
    legacy["v"] = 1
    legacy_token = "q1." + base64.urlsafe_b64encode(json.dumps(legacy).encode()).decode().rstrip("=")
    decoded = QueryContinuation.decode(legacy_token)
    with ArchiveStore(tmp_path) as archive:
        archive.begin_read_snapshot()
        try:
            validate_continuation_epoch(decoded.request, archive=archive)
        finally:
            archive.end_read_snapshot()


@pytest.mark.parametrize("token", ["q1._", "q1.////", "q1.8J+SqQ"])
def test_query_continuation_rejects_malformed_payloads_as_value_errors(token: str) -> None:
    with pytest.raises(ValueError, match="invalid query continuation"):
        QueryContinuation.decode(token)
