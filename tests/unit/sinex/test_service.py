"""Production publication-service retry, restart, gating, and status behavior."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

from polylogue.sinex.models import PublicationMode, PublicationReceipt, ReceiptState
from polylogue.sinex.service import PublicationService
from polylogue.sinex.transport import LocalReferenceTransport
from tests.unit.sinex._fixtures import MutableClock, publication_payload


class LostReceiptTransport:
    """Persist remotely, then lose the first response before local receipt commit."""

    def __init__(self) -> None:
        self.remote = LocalReferenceTransport()
        self.lose_first = True

    async def publish_revision(self, **kwargs: object) -> PublicationReceipt:
        receipt = await self.remote.publish_revision(**kwargs)  # type: ignore[arg-type]
        if self.lose_first:
            self.lose_first = False
            raise ConnectionError("authorization=Bearer never-persist-me")
        return receipt


class UnsafeDetailTransport:
    async def publish_revision(
        self, *, request_id: str, manifest_bytes: bytes, segment_bytes: Mapping[str, bytes]
    ) -> PublicationReceipt:
        return PublicationReceipt(
            request_id=request_id,
            state=ReceiptState.RAW_ACCEPTED,
            detail="authorization=Bearer bearer-secret token=top-secret endpoint=local",
        )


def test_lost_receipt_restart_reuses_request_id_without_duplicate_remote_record(
    workspace_env: dict[str, Path],
) -> None:
    db = workspace_env["archive_root"] / "source.db"
    clock = MutableClock(10_000)
    transport = LostReceiptTransport()
    first_process = PublicationService(db, PublicationMode.PRIMARY, transport, clock=clock, base_retry_ms=10)
    obligation = first_process.stage_payload(publication_payload(object_id="claude-code-session:crash"))
    assert obligation is not None
    first = first_process.drain_once()
    assert first.transport_failures == 1
    assert first_process.projection_blocked([obligation.object_id])
    assert transport.remote.call_count(obligation.request_id) == 1

    clock.advance(100)
    restarted = PublicationService(db, PublicationMode.PRIMARY, transport, clock=clock, base_retry_ms=10)
    assert restarted.drain_once().confirmed == 1
    assert not restarted.projection_blocked([obligation.object_id])
    assert transport.remote.call_count(obligation.request_id) == 1


def test_retry_debt_rejection_and_mode_specific_gating(workspace_env: dict[str, Path]) -> None:
    db = workspace_env["archive_root"] / "source.db"
    clock = MutableClock(20_000)

    raw_transport = LocalReferenceTransport(
        fault_fn=lambda _request_id, attempt: ReceiptState.RAW_ACCEPTED if attempt == 1 else None
    )
    raw_service = PublicationService(db, PublicationMode.PRIMARY, raw_transport, clock=clock, base_retry_ms=10)
    raw_service.stage_payload(publication_payload(object_id="claude-code-session:raw"))
    assert raw_service.drain_once(object_ids=["claude-code-session:raw"]).deferred == 1
    assert raw_service.projection_blocked(["claude-code-session:raw"])
    clock.advance(100)
    assert raw_service.drain_once(object_ids=["claude-code-session:raw"]).confirmed == 1

    debt = PublicationService(
        db,
        PublicationMode.PRIMARY,
        LocalReferenceTransport(fault_fn=lambda _r, _n: ReceiptState.DURABLE_DEBT),
        clock=clock,
    )
    debt.stage_payload(publication_payload(object_id="claude-code-session:debt"))
    assert debt.drain_once(object_ids=["claude-code-session:debt"]).durable_debt == 1
    assert not debt.projection_blocked(["claude-code-session:debt"])
    assert debt.lag(object_ids=["claude-code-session:debt"]) == 1

    rejected = PublicationService(
        db,
        PublicationMode.PRIMARY,
        LocalReferenceTransport(fault_fn=lambda _r, _n: ReceiptState.REJECTED),
        clock=clock,
    )
    rejected.stage_payload(publication_payload(object_id="claude-code-session:rejected"))
    rejected.drain_once(object_ids=["claude-code-session:rejected"])
    assert rejected.projection_blocked(["claude-code-session:rejected"])

    mirror = PublicationService(
        db,
        PublicationMode.MIRROR,
        LocalReferenceTransport(fault_fn=lambda _r, _n: ReceiptState.REJECTED),
        clock=clock,
    )
    mirror.stage_payload(publication_payload(object_id="claude-code-session:mirror"))
    mirror.drain_once(object_ids=["claude-code-session:mirror"])
    assert not mirror.projection_blocked(["claude-code-session:mirror"])
    assert mirror.status().blocking == 0
    assert mirror.lag(object_ids=["claude-code-session:mirror"]) == 1


def test_corrupt_payload_is_retry_debt_and_does_not_abort_bounded_batch(
    workspace_env: dict[str, Path],
) -> None:
    db = workspace_env["archive_root"] / "source.db"
    clock = MutableClock(30_000)
    service = PublicationService(db, PublicationMode.MIRROR, LocalReferenceTransport(), clock=clock, max_batch=2)
    service.stage_payload(publication_payload(object_id="claude-code-session:bad"))
    service.stage_payload(publication_payload(object_id="claude-code-session:good"))
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE sinex_publication_segments SET segment_bytes=X'FF' WHERE object_id=?",
        ("claude-code-session:bad",),
    )
    conn.commit()
    conn.close()

    summary = service.drain_once(object_ids=["claude-code-session:bad", "claude-code-session:good"], limit=999)
    assert summary.attempted == 2
    assert summary.confirmed == 1
    assert summary.payload_failures == 1
    assert summary.transport_failures == 0


def test_status_redacts_receipt_details_and_off_mode_is_zero_work(
    workspace_env: dict[str, Path], tmp_path: Path
) -> None:
    db = workspace_env["archive_root"] / "source.db"
    clock = MutableClock(40_000)
    service = PublicationService(db, PublicationMode.MIRROR, UnsafeDetailTransport(), clock=clock)
    service.stage_payload(publication_payload(object_id="claude-code-session:secret"))
    service.drain_once(object_ids=["claude-code-session:secret"])
    conn = sqlite3.connect(db)
    detail = conn.execute(
        "SELECT receipt_detail FROM sinex_publication_receipts WHERE object_id=?",
        ("claude-code-session:secret",),
    ).fetchone()[0]
    conn.close()
    assert "top-secret" not in detail
    assert "bearer-secret" not in detail
    assert "<redacted>" in detail
    assert "top-secret" not in json.dumps(service.status().as_dict())

    nonexistent = tmp_path / "off-does-not-exist.db"
    off = PublicationService(nonexistent, PublicationMode.OFF)
    assert off.stage_payload(publication_payload(object_id="claude-code-session:off")) is None
    assert off.drain_once().attempted == 0
    assert off.status().total == 0
    assert not nonexistent.exists()


def test_compat_retry_reports_lag_only_for_staged_subjects(workspace_env: dict[str, Path]) -> None:
    db = workspace_env["archive_root"] / "source.db"
    service = PublicationService(db, PublicationMode.MIRROR, LocalReferenceTransport())
    selected_payload = publication_payload(object_id="claude-code-session:selected")
    other_payload = publication_payload(object_id="claude-code-session:other", revision_id="other")
    selected = service.stage_payload(selected_payload)
    service.stage_payload(other_payload)
    assert selected is not None

    summary = asyncio.run(
        service.retry_pending([(selected, selected_payload.manifest_bytes, selected_payload.segment_bytes)])
    )

    assert summary.confirmed == 1
    assert summary.remaining_lag == 0
    assert service.lag() == 1
