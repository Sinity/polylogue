"""Orchestrates the durable publication obligation against a transport.

``PublicationService`` is the seam between (a) the durable ``source.db``
obligation ledger, which must survive process crashes, and (b) an injected
:class:`~polylogue.sinex.transport.SinexTransport`, which may fail, be slow,
or (in ``off`` mode) not exist at all. It intentionally does not know how to
build ``SessionMaterial``/manifest/segment bytes -- callers supply those
(from :mod:`polylogue.material_protocol.v1`), keeping this module's only job
"is this revision durably staged, and has it been confirmed".
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.sinex import obligations as obligations_store
from polylogue.sinex.models import ObligationStatus, PublicationMode, PublicationObligation, ReceiptState
from polylogue.sinex.transport import SinexTransport

_DEFAULT_CLOCK: Callable[[], int] = lambda: int(time.time() * 1000)  # noqa: E731


@dataclass(frozen=True, slots=True)
class DrainSummary:
    """Outcome of one drain pass over pending/retryable obligations."""

    attempted: int = 0
    confirmed: int = 0
    durable_debt: int = 0
    rejected: int = 0
    remaining_lag: int = 0


@dataclass
class PublicationService:
    """Stage and drain Sinex publication obligations for one archive."""

    source_db_path: Path
    mode: PublicationMode
    transport: SinexTransport | None = None
    clock: Callable[[], int] = field(default=_DEFAULT_CLOCK)

    def __post_init__(self) -> None:
        if self.mode is not PublicationMode.OFF and self.transport is None:
            raise ValueError(f"mode={self.mode.value} requires a transport")

    def stage(
        self,
        *,
        object_id: str,
        protocol_version: str,
        revision_id: str,
        manifest_digest: str,
        conn: sqlite3.Connection | None = None,
    ) -> PublicationObligation | None:
        """Create (or return the existing) durable obligation for a revision.

        Returns ``None`` in off mode without touching ``source.db`` at all --
        "off: today's Polylogue source/user tiers and blobs are canonical; no
        Sinex dependency or hidden network work" (design). When ``conn`` is
        supplied, the obligation is written into the CALLER's open
        transaction (the "same durable source-tier transaction" requirement);
        otherwise this method opens and commits its own transaction.
        """
        if self.mode is PublicationMode.OFF:
            return None
        now_ms = self.clock()
        if conn is not None:
            return obligations_store.record_obligation(
                conn,
                object_id=object_id,
                protocol_version=protocol_version,
                revision_id=revision_id,
                manifest_digest=manifest_digest,
                mode=self.mode,
                now_ms=now_ms,
            )
        owned_conn = sqlite3.connect(self.source_db_path, timeout=30.0)
        owned_conn.execute("PRAGMA busy_timeout = 30000")
        try:
            owned_conn.execute("BEGIN IMMEDIATE")
            obligation = obligations_store.record_obligation(
                owned_conn,
                object_id=object_id,
                protocol_version=protocol_version,
                revision_id=revision_id,
                manifest_digest=manifest_digest,
                mode=self.mode,
                now_ms=now_ms,
            )
            owned_conn.commit()
            return obligation
        except Exception:
            owned_conn.rollback()
            raise
        finally:
            owned_conn.close()

    async def publish(
        self,
        *,
        object_id: str,
        protocol_version: str,
        revision_id: str,
        manifest_digest: str,
        manifest_bytes: bytes,
        segment_bytes: Mapping[str, bytes],
        conn: sqlite3.Connection | None = None,
        on_confirmed: Callable[[PublicationObligation], None] | None = None,
    ) -> PublicationObligation | None:
        """Stage the obligation, then attempt exactly one publish.

        In ``primary`` mode ``on_confirmed`` is the ONLY sanctioned way a
        caller may advance a local projection for this revision -- it fires
        if and only if the transport receipt's
        :meth:`~polylogue.sinex.models.ReceiptState.unlocks_progress` is
        true, never on a bare send. Returns ``None`` in off mode.
        """
        obligation = self.stage(
            object_id=object_id,
            protocol_version=protocol_version,
            revision_id=revision_id,
            manifest_digest=manifest_digest,
            conn=conn,
        )
        if obligation is None:
            return None
        return await self._attempt(obligation, manifest_bytes, segment_bytes, on_confirmed=on_confirmed)

    async def _attempt(
        self,
        obligation: PublicationObligation,
        manifest_bytes: bytes,
        segment_bytes: Mapping[str, bytes],
        *,
        on_confirmed: Callable[[PublicationObligation], None] | None,
    ) -> PublicationObligation:
        assert self.transport is not None  # off mode never reaches here
        publishing_conn = sqlite3.connect(self.source_db_path, timeout=30.0)
        publishing_conn.execute("PRAGMA busy_timeout = 30000")
        try:
            publishing_conn.execute("BEGIN IMMEDIATE")
            obligations_store.mark_publishing(publishing_conn, obligation, now_ms=self.clock())
            publishing_conn.commit()
        except Exception:
            publishing_conn.rollback()
            raise
        finally:
            publishing_conn.close()
        receipt = await self.transport.publish_revision(
            request_id=obligation.request_id,
            manifest_bytes=manifest_bytes,
            segment_bytes=segment_bytes,
        )
        if receipt.state is ReceiptState.REJECTED:
            new_status = ObligationStatus.REJECTED
        elif receipt.state.unlocks_progress():
            new_status = (
                ObligationStatus.CONFIRMED
                if receipt.state is ReceiptState.PERSISTED_CONFIRMED
                else ObligationStatus.DURABLE_DEBT
            )
        else:
            # RAW_ACCEPTED or any other non-unlocking state: still pending,
            # still retryable. Never a silent success.
            new_status = ObligationStatus.PENDING
        conn = sqlite3.connect(self.source_db_path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            conn.execute("BEGIN IMMEDIATE")
            updated = obligations_store.mark_attempt(
                conn,
                obligation,
                status=new_status,
                receipt_state=receipt.state,
                error=receipt.detail if receipt.state is ReceiptState.REJECTED else None,
                now_ms=self.clock(),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        if receipt.state.unlocks_progress() and on_confirmed is not None:
            on_confirmed(updated)
        return updated

    def pending(self) -> tuple[PublicationObligation, ...]:
        """List every non-terminal obligation (mirror-mode lag report)."""
        conn = sqlite3.connect(f"file:{self.source_db_path}?mode=ro", uri=True)
        try:
            return obligations_store.list_obligations(
                conn,
                statuses=(ObligationStatus.PENDING, ObligationStatus.PUBLISHING, ObligationStatus.DURABLE_DEBT),
            )
        finally:
            conn.close()

    def lag(self) -> int:
        """Exact count of obligations awaiting a confirming receipt."""
        return len(self.pending())

    async def retry_pending(
        self,
        staged: Sequence[tuple[PublicationObligation, bytes, Mapping[str, bytes]]],
        *,
        on_confirmed: Callable[[PublicationObligation], None] | None = None,
    ) -> DrainSummary:
        """Redrive previously-staged obligations the caller re-supplies bytes for.

        The obligation ledger deliberately does not store manifest/segment
        bytes (that is the material store's job, not this ledger's); a
        caller resolves each pending obligation back to its material bytes
        and passes the pairs here. Off mode returns an all-zero summary
        without any transport calls.
        """
        if self.mode is PublicationMode.OFF:
            return DrainSummary()
        confirmed = 0
        debt = 0
        rejected = 0
        for obligation, manifest_bytes, segment_bytes in staged:
            updated = await self._attempt(obligation, manifest_bytes, segment_bytes, on_confirmed=on_confirmed)
            if updated.status is ObligationStatus.CONFIRMED:
                confirmed += 1
            elif updated.status is ObligationStatus.DURABLE_DEBT:
                debt += 1
            elif updated.status is ObligationStatus.REJECTED:
                rejected += 1
        return DrainSummary(
            attempted=len(staged),
            confirmed=confirmed,
            durable_debt=debt,
            rejected=rejected,
            remaining_lag=self.lag(),
        )


__all__ = ["DrainSummary", "PublicationService"]
