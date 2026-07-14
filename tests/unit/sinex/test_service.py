"""PublicationService: mode gating, durability, and receipt-barrier semantics.

These tests exercise the real ``sinex_publication_obligations`` table via a
real source.db (``workspace_env``) and the real ``LocalReferenceTransport``
contract double -- only network I/O is out of scope here, per the package
docstring's documented cross-repo blocker.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.sinex.models import ObligationStatus, PublicationMode, ReceiptState
from polylogue.sinex.service import PublicationService
from polylogue.sinex.transport import LocalReferenceTransport


def _obligation_table_rows(source_db: Path) -> int:
    conn = sqlite3.connect(source_db)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM sinex_publication_obligations").fetchone()[0])
    finally:
        conn.close()


def test_off_mode_creates_no_obligation_and_requires_no_transport(workspace_env: dict[str, Path]) -> None:
    source_db = workspace_env["archive_root"] / "source.db"
    service = PublicationService(source_db_path=source_db, mode=PublicationMode.OFF, transport=None)

    obligation = service.stage(
        object_id="claude-code-session:s1",
        protocol_version="polylogue.material-protocol/v1",
        revision_id="rev-1",
        manifest_digest="digest-1",
    )

    assert obligation is None
    assert _obligation_table_rows(source_db) == 0


def test_non_off_mode_requires_a_transport(workspace_env: dict[str, Path]) -> None:
    source_db = workspace_env["archive_root"] / "source.db"
    with pytest.raises(ValueError, match="requires a transport"):
        PublicationService(source_db_path=source_db, mode=PublicationMode.MIRROR, transport=None)


async def test_mirror_mode_stages_obligation_in_caller_supplied_transaction(workspace_env: dict[str, Path]) -> None:
    """The obligation must be visible in the SAME transaction the caller

    controls (design: "the same durable source-tier transaction that records
    the acquired/normalized revision"), and rolling that transaction back
    must roll the obligation back too -- an obligation orphaned from its
    revision commit is worse than no obligation.
    """
    source_db = workspace_env["archive_root"] / "source.db"
    transport = LocalReferenceTransport()
    service = PublicationService(source_db_path=source_db, mode=PublicationMode.MIRROR, transport=transport)

    conn = sqlite3.connect(source_db, timeout=30.0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        obligation = service.stage(
            object_id="claude-code-session:s1",
            protocol_version="polylogue.material-protocol/v1",
            revision_id="rev-1",
            manifest_digest="digest-1",
            conn=conn,
        )
        assert obligation is not None
        # Visible inside the same uncommitted transaction.
        assert conn.execute("SELECT COUNT(*) FROM sinex_publication_obligations").fetchone()[0] == 1
        conn.rollback()
    finally:
        conn.close()

    assert _obligation_table_rows(source_db) == 0


async def test_primary_mode_advances_projection_only_on_confirmed_receipt(workspace_env: dict[str, Path]) -> None:
    source_db = workspace_env["archive_root"] / "source.db"
    transport = LocalReferenceTransport()
    service = PublicationService(source_db_path=source_db, mode=PublicationMode.PRIMARY, transport=transport)
    confirmed_calls: list[str] = []

    obligation = await service.publish(
        object_id="claude-code-session:s1",
        protocol_version="polylogue.material-protocol/v1",
        revision_id="rev-1",
        manifest_digest="digest-1",
        manifest_bytes=b"{}",
        segment_bytes={"head": b"{}"},
        on_confirmed=lambda o: confirmed_calls.append(o.object_id),
    )

    assert obligation is not None
    assert obligation.status is ObligationStatus.CONFIRMED
    assert confirmed_calls == ["claude-code-session:s1"]
    assert service.lag() == 0


async def test_primary_mode_does_not_advance_projection_on_raw_accepted(workspace_env: dict[str, Path]) -> None:
    """A bare RAW_ACCEPTED (in-memory accept, not a durable Sinex receipt)

    must leave the obligation pending and must NOT fire on_confirmed -- this
    is the exact failure mode r6d.11 exists to prevent (mpsc/NATS-publish
    acceptance mistaken for a durable commit).
    """
    source_db = workspace_env["archive_root"] / "source.db"
    transport = LocalReferenceTransport(fault_fn=lambda _rid, _attempt: ReceiptState.RAW_ACCEPTED)
    service = PublicationService(source_db_path=source_db, mode=PublicationMode.PRIMARY, transport=transport)
    confirmed_calls: list[str] = []

    obligation = await service.publish(
        object_id="claude-code-session:s1",
        protocol_version="polylogue.material-protocol/v1",
        revision_id="rev-1",
        manifest_digest="digest-1",
        manifest_bytes=b"{}",
        segment_bytes={},
        on_confirmed=lambda o: confirmed_calls.append(o.object_id),
    )

    assert obligation is not None
    assert obligation.status is ObligationStatus.PENDING
    assert confirmed_calls == []
    assert service.lag() == 1


async def test_rejected_receipt_marks_obligation_rejected_and_never_confirms(workspace_env: dict[str, Path]) -> None:
    source_db = workspace_env["archive_root"] / "source.db"
    transport = LocalReferenceTransport(fault_fn=lambda _rid, _attempt: ReceiptState.REJECTED)
    service = PublicationService(source_db_path=source_db, mode=PublicationMode.MIRROR, transport=transport)
    confirmed_calls: list[str] = []

    obligation = await service.publish(
        object_id="claude-code-session:s1",
        protocol_version="polylogue.material-protocol/v1",
        revision_id="rev-1",
        manifest_digest="digest-1",
        manifest_bytes=b"{}",
        segment_bytes={},
        on_confirmed=lambda o: confirmed_calls.append(o.object_id),
    )

    assert obligation is not None
    assert obligation.status is ObligationStatus.REJECTED
    assert confirmed_calls == []
    # rejected is terminal: not in the retryable "lag" set.
    assert service.lag() == 0


async def test_durable_debt_unlocks_progress_but_is_distinct_from_confirmed(workspace_env: dict[str, Path]) -> None:
    source_db = workspace_env["archive_root"] / "source.db"
    transport = LocalReferenceTransport(fault_fn=lambda _rid, _attempt: ReceiptState.DURABLE_DEBT)
    service = PublicationService(source_db_path=source_db, mode=PublicationMode.MIRROR, transport=transport)
    confirmed_calls: list[str] = []

    obligation = await service.publish(
        object_id="claude-code-session:s1",
        protocol_version="polylogue.material-protocol/v1",
        revision_id="rev-1",
        manifest_digest="digest-1",
        manifest_bytes=b"{}",
        segment_bytes={},
        on_confirmed=lambda o: confirmed_calls.append(o.object_id),
    )

    assert obligation is not None
    assert obligation.status is ObligationStatus.DURABLE_DEBT
    # DurableDebt IS a documented unlocking outcome (r6d.11) -- on_confirmed
    # fires -- but the persisted status stays distinguishable from a clean
    # PersistedConfirmed so a mirror-mode operator can see the exact lag.
    assert confirmed_calls == ["claude-code-session:s1"]


async def test_retry_pending_eventually_confirms_and_reports_zero_remaining_lag(
    workspace_env: dict[str, Path],
) -> None:
    source_db = workspace_env["archive_root"] / "source.db"
    transport = LocalReferenceTransport(
        fault_fn=lambda _rid, attempt_number: ReceiptState.RAW_ACCEPTED if attempt_number == 1 else None
    )
    service = PublicationService(source_db_path=source_db, mode=PublicationMode.MIRROR, transport=transport)

    first_attempt = await service.publish(
        object_id="claude-code-session:s1",
        protocol_version="polylogue.material-protocol/v1",
        revision_id="rev-1",
        manifest_digest="digest-1",
        manifest_bytes=b"{}",
        segment_bytes={},
    )
    assert first_attempt is not None
    assert first_attempt.status is ObligationStatus.PENDING
    assert service.lag() == 1

    summary = await service.retry_pending([(first_attempt, b"{}", {})])

    assert summary.attempted == 1
    assert summary.confirmed == 1
    assert summary.remaining_lag == 0
    assert service.lag() == 0
    assert transport.call_count("claude-code-session:s1|polylogue.material-protocol/v1|rev-1|digest-1") == 2


async def test_retry_pending_is_a_true_no_op_in_off_mode(workspace_env: dict[str, Path]) -> None:
    source_db = workspace_env["archive_root"] / "source.db"
    service = PublicationService(source_db_path=source_db, mode=PublicationMode.OFF, transport=None)

    summary = await service.retry_pending([])

    assert summary.attempted == 0
    assert summary.confirmed == 0


def test_obligation_survives_a_process_restart_between_commit_and_transport_attempt(
    workspace_env: dict[str, Path],
) -> None:
    """Killpoint: a crash after the durable local commit but before the

    transport attempt runs must not lose the obligation. Simulated here by
    staging with one ``PublicationService``/connection (committed and
    closed, standing in for "process A commits, then dies"), then reading it
    back with a brand-new ``PublicationService`` instance that never saw the
    first one in memory -- the only thing connecting them is the durable
    source.db row.
    """
    source_db = workspace_env["archive_root"] / "source.db"
    transport_a = LocalReferenceTransport()
    service_a = PublicationService(source_db_path=source_db, mode=PublicationMode.MIRROR, transport=transport_a)
    staged = service_a.stage(
        object_id="claude-code-session:s1",
        protocol_version="polylogue.material-protocol/v1",
        revision_id="rev-1",
        manifest_digest="digest-1",
    )
    assert staged is not None
    del service_a, transport_a  # "process A" is gone; nothing but the DB row remains

    transport_b = LocalReferenceTransport()
    service_b = PublicationService(source_db_path=source_db, mode=PublicationMode.MIRROR, transport=transport_b)
    resumed = service_b.pending()

    assert len(resumed) == 1
    assert resumed[0].object_id == "claude-code-session:s1"
    assert resumed[0].status is ObligationStatus.PENDING
    assert resumed[0].attempt_count == 0
    assert transport_b.call_count() == 0  # resuming did not fabricate a phantom attempt


def test_obligation_ledger_does_not_depend_on_ops_db(workspace_env: dict[str, Path]) -> None:
    """ops.db is disposable diagnostics only; deleting it must not touch the

    durable source.db obligation (design: "ops.db/convergence debt may
    mirror attempts, latency, and diagnostics only. It is disposable and can
    never be the sole outbox or recovery authority.").
    """
    source_db = workspace_env["archive_root"] / "source.db"
    ops_db = workspace_env["archive_root"] / "ops.db"
    transport = LocalReferenceTransport()
    service = PublicationService(source_db_path=source_db, mode=PublicationMode.MIRROR, transport=transport)
    service.stage(
        object_id="claude-code-session:s1",
        protocol_version="polylogue.material-protocol/v1",
        revision_id="rev-1",
        manifest_digest="digest-1",
    )

    ops_db.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(f"{ops_db}{suffix}").unlink(missing_ok=True)

    assert service.lag() == 1
    assert len(service.pending()) == 1


async def test_stage_retry_after_confirmation_does_not_reopen_the_obligation(
    workspace_env: dict[str, Path],
) -> None:
    """Restaging the SAME revision after it already confirmed must return the

    settled row unchanged, not reset it to pending -- proves duplicate
    delivery (e.g. a re-run ingest pass for a revision already published)
    cannot resurrect a closed obligation.
    """
    source_db = workspace_env["archive_root"] / "source.db"
    transport = LocalReferenceTransport()
    service = PublicationService(source_db_path=source_db, mode=PublicationMode.MIRROR, transport=transport)

    confirmed = await service.publish(
        object_id="claude-code-session:s1",
        protocol_version="polylogue.material-protocol/v1",
        revision_id="rev-1",
        manifest_digest="digest-1",
        manifest_bytes=b"{}",
        segment_bytes={},
    )
    assert confirmed is not None
    assert confirmed.status is ObligationStatus.CONFIRMED

    restaged = service.stage(
        object_id="claude-code-session:s1",
        protocol_version="polylogue.material-protocol/v1",
        revision_id="rev-1",
        manifest_digest="digest-1",
    )
    assert restaged == confirmed
    assert _obligation_table_rows(source_db) == 1
