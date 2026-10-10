"""One marker carrier owns one assertion transaction and no discarded envelopes."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path

import aiosqlite
import pytest

from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.markers import candidates_for_block, lower_markers
from polylogue.markers.lowering import assertion_id_for_marker
from polylogue.markers.models import MarkerCandidate
from polylogue.storage.accepted_marker_inputs import append_accepted_marker_input, prepare_accepted_marker_input
from polylogue.storage.derived.session.marker_domain import SessionMarkerDerivation
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers import user_write
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL
from tests.infra.identity import archive_block_id, fixture_block_content_identity
from tests.infra.user_tier import connect_measured_user_tier


def _carrier(
    root: Path, count: int, *, trace: list[str] | None = None
) -> tuple[SessionMarkerDerivation, str, tuple[MarkerCandidate, ...]]:
    source_db, user_db = root / "source.db", root / "user.db"
    candidates = tuple(
        candidates_for_block(
            f"message-{position}",
            archive_block_id(f"message-{position}", content_identity=fixture_block_content_identity(position)),
            f"::note: retained candidate {position}",
        )[0]
        for position in range(count)
    )
    records = [asdict(candidate) for candidate in candidates]
    for record, candidate in zip(records, candidates, strict=True):
        assert candidate.assertion_kind is not None
        record["assertion_kind"] = candidate.assertion_kind.value

    async def append() -> None:
        async with aiosqlite.connect(source_db) as source:
            await source.executescript(SOURCE_DDL)
            batch = prepare_accepted_marker_input(
                "synthetic-carrier", [{"session_id": "synthetic-session", "candidates": records}]
            )
            await append_accepted_marker_input(source, batch)
            await source.commit()

    asyncio.run(append())
    with sqlite3.connect(user_db) as user:
        user.executescript(USER_DDL)

    def writer() -> sqlite3.Connection:
        conn = connect_measured(user_db)
        if trace is not None:
            conn.set_trace_callback(trace.append)
        return conn

    adapter = SessionMarkerDerivation(
        lambda: sqlite3.connect(f"file:{source_db}?mode=ro", uri=True),
        lambda: sqlite3.connect(f"file:{user_db}?mode=ro", uri=True),
        writer,
    )
    return adapter, adapter.required_page(object(), cursor=None, limit=1)[0][0], candidates


def test_thousand_marker_assertions_use_one_transaction_and_no_envelopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace: list[str] = []
    adapter, key, _candidates = _carrier(tmp_path, 1000, trace=trace)
    replacement = adapter.compute(object(), key)

    def forbid_envelope(*args: object, **kwargs: object) -> object:
        pytest.fail("marker lowering consumes assertion identities, not envelopes")

    monkeypatch.setattr(user_write, "read_assertion_envelope", forbid_envelope)
    assert adapter.publish(object(), replacement)
    controls = [
        statement.strip().upper()
        for statement in trace
        if statement.strip().upper().startswith(("BEGIN", "COMMIT", "SAVEPOINT", "RELEASE", "ROLLBACK"))
        or "WHERE 0" in statement.upper()
    ]
    assert controls == ["BEGIN IMMEDIATE", "COMMIT"]
    with sqlite3.connect(tmp_path / "user.db") as user:
        assert user.execute("SELECT count(*) FROM assertions").fetchone() == (1000,)
        assert user.execute("SELECT applied_sequence FROM accepted_marker_delivery_cursor").fetchone() == (1,)


@pytest.mark.parametrize("failure", (ValueError, KeyboardInterrupt))
def test_marker_batch_failure_or_cancellation_rolls_back_whole_carrier_and_restarts(
    tmp_path: Path, failure: type[BaseException]
) -> None:
    adapter, key, _candidates = _carrier(tmp_path, 2)
    replacement = adapter.compute(object(), key)
    payload = replacement.payload

    def interrupted() -> Iterator[MarkerCandidate]:
        yield next(payload)
        raise failure("synthetic carrier interruption")

    with pytest.raises(failure):
        adapter.publish(object(), replace(replacement, payload=interrupted()))
    with sqlite3.connect(tmp_path / "user.db") as user:
        assert user.execute("SELECT count(*) FROM assertions").fetchone() == (0,)
        assert user.execute("SELECT count(*) FROM accepted_marker_delivery_cursor").fetchone() == (0,)
    assert adapter.publish(object(), adapter.compute(object(), key))
    with sqlite3.connect(tmp_path / "user.db") as user:
        assert user.execute("SELECT count(*) FROM assertions").fetchone() == (2,)
        assert user.execute("SELECT applied_sequence FROM accepted_marker_delivery_cursor").fetchone() == (1,)


def test_marker_batch_preserves_operator_rows_and_terminal_judgments(tmp_path: Path) -> None:
    adapter, key, candidates = _carrier(tmp_path, 3)
    with closing(connect_measured_user_tier(tmp_path / "user.db")) as user:
        human_id = assertion_id_for_marker(candidates[0])
        assert human_id is not None
        user_write.upsert_assertion(
            user,
            assertion_id=human_id,
            target_ref="session:operator",
            kind=AssertionKind.NOTE,
            body_text="operator-owned",
            now_ms=1,
        )
        (judged_id,) = lower_markers(user, (candidates[1],), now_ms=2)
        user_write.mark_assertion_status(user, judged_id, AssertionStatus.REJECTED, now_ms=3)
        before = user.execute("SELECT * FROM assertions ORDER BY assertion_id").fetchall()
        user.commit()
    assert adapter.publish(object(), adapter.compute(object(), key))
    with sqlite3.connect(tmp_path / "user.db") as user:
        after = user.execute(
            "SELECT * FROM assertions WHERE assertion_id IN (?, ?) ORDER BY assertion_id", (human_id, judged_id)
        ).fetchall()
        assert [tuple(row) for row in before] == after
        assert user.execute(
            "SELECT status, author_kind, context_policy_json FROM assertions WHERE assertion_id=?",
            (assertion_id_for_marker(candidates[2]),),
        ).fetchone() == ("candidate", "agent", '{"inject":false,"promotion_required":true}')


def test_failed_nested_assertion_batch_preserves_prior_caller_work_and_retires_handle(tmp_path: Path) -> None:
    conn = connect_measured_user_tier(tmp_path / "user.db")
    try:
        user_write.upsert_assertion(conn, assertion_id="prior", target_ref="session:prior", kind=AssertionKind.NOTE)
        with pytest.raises(ValueError, match="synthetic batch failure"):
            with user_write.assertion_write_batch(conn) as writer:
                writer.upsert(assertion_id="batch-only", target_ref="session:batch", kind=AssertionKind.NOTE)
                raise ValueError("synthetic batch failure")
        assert conn.in_transaction
        assert user_write.read_assertion_envelope(conn, "prior") is not None
        assert user_write.read_assertion_envelope(conn, "batch-only") is None
        with pytest.raises(RuntimeError, match="owning batch transaction"):
            writer.upsert(assertion_id="escaped", target_ref="session:escaped", kind=AssertionKind.NOTE)
    finally:
        conn.rollback()
        conn.close()
