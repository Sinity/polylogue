from __future__ import annotations

import sqlite3

import pytest

from polylogue.archive.context_models import ContextImage, ContextOmission, ContextSegment, ContextSpec
from polylogue.context.compiler import context_snapshot_record_from_image
from polylogue.core.refs import EvidenceRef
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.context_delivery_write import (
    list_context_deliveries,
    read_context_delivery,
    write_context_delivery,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(USER_DDL)
    conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER]}")
    return conn


def _image(*, markdown: str = "exact reviewed context") -> ContextImage:
    evidence = EvidenceRef(session_id="codex-session:source", message_id="m1")
    segment = ContextSegment(
        segment_id="assertion:a1",
        kind="assertion",
        title="Reviewed note",
        markdown=markdown,
        evidence_refs=(evidence,),
        assertion_refs=("assertion:a1",),
        caveats=("snapshot evidence",),
        token_estimate=5,
    )
    return ContextImage(
        spec=ContextSpec(seed_refs=("assertion:a1",), read_views=()),
        segments=(segment,),
        evidence_refs=(evidence,),
        assertion_refs=("assertion:a1",),
        omitted=(ContextOmission(ref="assertion:a2", reason="budget", detail="lower rank"),),
        caveats=("snapshot evidence",),
        token_estimate=5,
    )


def test_context_delivery_round_trips_exact_image_and_identity() -> None:
    conn = _conn()
    image = _image()
    record = context_snapshot_record_from_image(image, boundary="session-start", run_ref="run:r1")

    written = write_context_delivery(
        conn,
        image=image,
        record=record,
        recipient_ref="agent:codex-main",
        delivered_by_ref="user:local",
        delivered_at_ms=123,
    )
    replay = write_context_delivery(
        conn,
        image=image,
        record=record,
        recipient_ref="agent:codex-main",
        delivered_by_ref="user:local",
        delivered_at_ms=123,
    )

    assert written.context_image == image
    assert written.recipient_ref == "agent:codex-main"
    assert written.delivered_by_ref == "user:local"
    assert written.boundary == "session-start"
    assert written.segment_refs == ("assertion:a1",)
    assert written.evidence_refs == ("codex-session:source::m1",)
    assert written.assertion_refs == ("assertion:a1",)
    assert written.omissions[0]["detail"] == "lower rank"
    assert replay.outcome == "idempotent"
    assert read_context_delivery(conn, record.snapshot_ref) == written
    assert [item.snapshot_ref for item in list_context_deliveries(conn, recipient_ref="agent:codex-main").items] == [
        written.snapshot_ref
    ]
    assert [item.snapshot_ref for item in list_context_deliveries(conn, assertion_ref="assertion:a1").items] == [
        written.snapshot_ref
    ]


@pytest.mark.parametrize(
    ("field", "kwargs", "match"),
    [
        ("recipient", {"recipient_ref": "agent:other"}, "recipient_ref"),
        ("actor", {"delivered_by_ref": "agent:runtime"}, "delivered_by_ref"),
        ("timestamp", {"delivered_at_ms": 124}, "delivered_at_ms"),
    ],
)
def test_context_delivery_refuses_same_ref_identity_drift(field: str, kwargs: dict[str, object], match: str) -> None:
    del field
    conn = _conn()
    image = _image()
    record = context_snapshot_record_from_image(image, boundary="session-start", run_ref="run:r1")
    base: dict[str, object] = {
        "image": image,
        "record": record,
        "recipient_ref": "agent:codex-main",
        "delivered_by_ref": "user:local",
        "delivered_at_ms": 123,
    }
    write_context_delivery(conn, **base)  # type: ignore[arg-type]
    base.update(kwargs)

    with pytest.raises(ValueError, match=match):
        write_context_delivery(conn, **base)  # type: ignore[arg-type]


def test_context_delivery_refuses_image_or_record_drift() -> None:
    conn = _conn()
    image = _image()
    record = context_snapshot_record_from_image(image, boundary="session-start")

    with pytest.raises(ValueError, match="digest"):
        write_context_delivery(
            conn,
            image=_image(markdown="mutated"),
            record=record,
            recipient_ref="agent:codex-main",
            delivered_by_ref="user:local",
        )
    with pytest.raises(ValueError, match="boundary"):
        write_context_delivery(
            conn,
            image=image,
            record=record.model_copy(update={"boundary": " "}),
            recipient_ref="agent:codex-main",
            delivered_by_ref="user:local",
        )


@pytest.mark.parametrize(
    ("recipient", "actor", "match"),
    [
        ("not-a-ref", "user:local", "object ref"),
        ("agent:codex-main", "session:not-an-actor", "delivered_by_ref"),
    ],
)
def test_context_delivery_validates_recipient_and_actor_refs(recipient: str, actor: str, match: str) -> None:
    conn = _conn()
    image = _image()
    record = context_snapshot_record_from_image(image, boundary="session-start")
    with pytest.raises(ValueError, match=match):
        write_context_delivery(
            conn,
            image=image,
            record=record,
            recipient_ref=recipient,
            delivered_by_ref=actor,
        )


@pytest.mark.parametrize("operation", ["write", "read", "list"])
@pytest.mark.parametrize("execute_failure", [False, True])
def test_context_delivery_retains_original_failed_statement_until_creator_retry(
    monkeypatch: pytest.MonkeyPatch, operation: str, execute_failure: bool
) -> None:
    from builtins import BaseExceptionGroup
    from typing import Any

    from polylogue.storage.io_phase_metrics import _MeasuredConnection, connect_measured, live_connection_cursors
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError, NativeSQLCustodyOwner
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    connection = connect_measured(":memory:")
    assert isinstance(connection, _MeasuredConnection)
    owner = NativeSQLCustodyOwner(connection)
    selected: list[ControlledCursor] = []
    completed: list[str] = []
    primary = OSError("synthetic context statement failure")
    image = _image()
    record = context_snapshot_record_from_image(image, boundary="session-start", run_ref="run:r1")
    try:
        connection.executescript(USER_DDL)
        if operation != "write":
            write_context_delivery(
                connection,
                image=image,
                record=record,
                recipient_ref="agent:codex-main",
                delivered_by_ref="user:local",
                delivered_at_ms=123,
            )
            connection.commit()
        owner.retain_settlement_callback(lambda: completed.append("settled"))
        make_cursor = connection.cursor

        class ContextCursor(ControlledCursor):
            def execute(self, sql: str, parameters: Any = (), /) -> ContextCursor:
                matches = (
                    sql.lstrip().startswith("INSERT INTO context_deliveries")
                    if operation == "write"
                    else sql.lstrip().startswith("SELECT snapshot_ref, recipient_ref")
                    if operation == "read"
                    # A paged list read opens with its exact total.
                    else sql.lstrip().startswith("SELECT COUNT(*) FROM context_deliveries")
                )
                if matches:
                    selected.append(self)
                    self.allow_cleanup.clear()
                    if execute_failure:
                        raise primary
                super().execute(sql, parameters)
                return self

        def cursor() -> sqlite3.Cursor:
            return make_cursor(factory=ContextCursor)

        monkeypatch.setattr(connection, "cursor", cursor)
        with pytest.raises(BaseExceptionGroup if execute_failure else OSError) as failure:
            if operation == "write":
                write_context_delivery(
                    connection,
                    image=image,
                    record=record,
                    recipient_ref="agent:codex-main",
                    delivered_by_ref="user:local",
                    delivered_at_ms=123,
                )
            elif operation == "read":
                read_context_delivery(connection, record.snapshot_ref)
            else:
                list_context_deliveries(connection, recipient_ref="agent:codex-main")
        assert len(selected) == 1
        statement = selected[0]
        assert statement.close_attempts == 1
        assert statement in live_connection_cursors(connection)
        assert any(cursor is statement for cursor in connection._unsettled_native_cursors.values())
        if execute_failure:
            assert isinstance(failure.value, BaseExceptionGroup)
            assert any(error is primary for error in failure.value.exceptions)
        else:
            assert failure.value is statement.cleanup_failure
        with pytest.raises(NativeConnectionSettlementError) as unsettled:
            owner.close()
        assert unsettled.value.owner is owner and owner.connection is connection
        assert owner.close_required and not owner._settled and completed == []
        assert statement.close_attempts == 2
        statement.allow_cleanup.set()
        owner.close()
        assert statement.close_attempts == 3
        assert owner.connection is None and owner._settled and completed == ["settled"]
        assert not connection._unsettled_native_cursors
    finally:
        for statement in selected:
            statement.allow_cleanup.set()
        owner.close()
