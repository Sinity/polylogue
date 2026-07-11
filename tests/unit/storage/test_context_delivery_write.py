from __future__ import annotations

import sqlite3

import pytest

from polylogue.context.compiler import (
    ContextImage,
    ContextOmission,
    ContextSegment,
    ContextSpec,
    context_snapshot_record_from_image,
)
from polylogue.core.refs import EvidenceRef
from polylogue.storage.sqlite.archive_tiers.context_delivery_write import (
    list_context_deliveries,
    read_context_delivery,
    write_context_delivery,
)
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL, USER_SCHEMA_VERSION


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(USER_DDL)
    conn.execute(f"PRAGMA user_version = {USER_SCHEMA_VERSION}")
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
    assert list_context_deliveries(conn, recipient_ref="agent:codex-main") == [written]
    assert list_context_deliveries(conn, assertion_ref="assertion:a1") == [written]


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
