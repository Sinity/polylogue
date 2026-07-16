"""Supervised, bounded draining of the durable Sinex publication outbox."""

from __future__ import annotations

import asyncio
import re
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.sinex import obligations as obligations_store
from polylogue.sinex.models import (
    ObligationStatus,
    PublicationMode,
    PublicationObligation,
    PublicationPayload,
    PublicationReceipt,
    PublicationStatus,
    ReceiptState,
)
from polylogue.sinex.transport import SinexTransport, SinexTransportUnavailableError

_RETRYABLE_STATUSES = (
    ObligationStatus.PENDING,
    ObligationStatus.PUBLISHING,
    ObligationStatus.DURABLE_DEBT,
)

_DEFAULT_CLOCK: Callable[[], int] = lambda: int(time.time() * 1000)  # noqa: E731
_SAFE_CODE_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")
_SECRET_RE = re.compile(r"(?i)(token|secret|password|authorization|api[-_]?key)\s*[:=]\s*\S+")


@dataclass(frozen=True, slots=True)
class DrainSummary:
    """Outcome of one bounded drain pass."""

    attempted: int = 0
    confirmed: int = 0
    durable_debt: int = 0
    rejected: int = 0
    deferred: int = 0
    transport_failures: int = 0
    remaining_lag: int = 0


@dataclass
class PublicationService:
    """Stage and drain publication obligations for one source.db.

    SQLite mutations occur only on the calling daemon thread.  When a sync
    convergence pass is already running inside an asyncio loop, only the
    transport coroutine is run in a short-lived worker thread; that thread
    never receives a database connection.
    """

    source_db_path: Path
    mode: PublicationMode
    transport: SinexTransport | None = None
    clock: Callable[[], int] = field(default=_DEFAULT_CLOCK)
    max_batch: int = 16
    attempt_timeout_s: float = 30.0
    base_retry_ms: int = 1_000
    max_retry_ms: int = 5 * 60_000
    publishing_lease_ms: int = 60_000
    durable_debt_retry_ms: int = 15 * 60_000

    def __post_init__(self) -> None:
        self.mode = PublicationMode.from_string(self.mode)
        if self.mode is not PublicationMode.OFF and self.transport is None:
            raise SinexTransportUnavailableError(
                f"mode={self.mode.value} requires an injected configured Sinex transport"
            )
        if self.max_batch < 1:
            raise ValueError("max_batch must be positive")
        if self.attempt_timeout_s <= 0:
            raise ValueError("attempt_timeout_s must be positive")

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            conn = sqlite3.connect(f"file:{self.source_db_path}?mode=ro", uri=True, timeout=30.0)
        else:
            conn = sqlite3.connect(self.source_db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _safe_code(value: str) -> str:
        return _SAFE_CODE_RE.sub("_", value)[:128]

    @staticmethod
    def _safe_detail(value: str) -> str:
        return _SECRET_RE.sub(r"\1=<redacted>", value.replace("\x00", ""))[:256]

    def _retry_at(self, obligation: PublicationObligation, now_ms: int) -> int:
        exponent = min(obligation.attempt_count, 12)
        delay = int(self.base_retry_ms) * (2**exponent)
        return int(now_ms + min(int(self.max_retry_ms), delay))

    def stage(
        self,
        *,
        object_id: str,
        protocol_version: str,
        revision_id: str,
        manifest_digest: str,
        conn: sqlite3.Connection | None = None,
    ) -> PublicationObligation | None:
        """Compatibility metadata stage; production ingest uses stage_payload."""
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
        owned = self._connect()
        try:
            owned.execute("BEGIN IMMEDIATE")
            obligation = obligations_store.record_obligation(
                owned,
                object_id=object_id,
                protocol_version=protocol_version,
                revision_id=revision_id,
                manifest_digest=manifest_digest,
                mode=self.mode,
                now_ms=now_ms,
            )
            owned.commit()
            return obligation
        except Exception:
            owned.rollback()
            raise
        finally:
            owned.close()

    def stage_payload(
        self,
        payload: PublicationPayload,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> PublicationObligation | None:
        """Durably stage exact bytes; off mode performs no database work."""
        if self.mode is PublicationMode.OFF:
            return None
        now_ms = self.clock()
        if conn is not None:
            return obligations_store.stage_payload(conn, payload=payload, mode=self.mode, now_ms=now_ms)
        owned = self._connect()
        try:
            owned.execute("BEGIN IMMEDIATE")
            obligation = obligations_store.stage_payload(owned, payload=payload, mode=self.mode, now_ms=now_ms)
            owned.commit()
            return obligation
        except Exception:
            owned.rollback()
            raise
        finally:
            owned.close()

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
        """Stage exact bytes and make one bounded transport attempt."""
        if conn is not None:
            raise ValueError(
                "publish cannot invoke transport inside an uncommitted caller transaction; "
                "use stage_payload(..., conn=conn), commit, then drain_once()"
            )
        payload = PublicationPayload(
            object_id=object_id,
            protocol_version=protocol_version,
            revision_id=revision_id,
            manifest_digest=manifest_digest,
            manifest_bytes=manifest_bytes,
            segments=tuple(sorted((str(name), bytes(value)) for name, value in segment_bytes.items())),
        )
        obligation = self.stage_payload(payload)
        if obligation is None:
            return None
        return await self._attempt_async(obligation, payload, on_confirmed=on_confirmed)

    def _lease(self, obligation: PublicationObligation) -> PublicationObligation:
        now_ms = self.clock()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            leased = obligations_store.mark_publishing(
                conn,
                obligation,
                now_ms=now_ms,
                lease_until_ms=now_ms + self.publishing_lease_ms,
            )
            conn.commit()
            return leased
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _persist_outcome(
        self,
        obligation: PublicationObligation,
        *,
        receipt: PublicationReceipt | None,
        error_code: str | None,
    ) -> PublicationObligation:
        now_ms = self.clock()
        if receipt is None:
            status = ObligationStatus.PENDING
            next_attempt_at_ms = self._retry_at(obligation, now_ms)
        elif receipt.state is ReceiptState.REJECTED:
            status = ObligationStatus.REJECTED
            next_attempt_at_ms = None
        elif receipt.state is ReceiptState.PERSISTED_CONFIRMED:
            status = ObligationStatus.CONFIRMED
            next_attempt_at_ms = None
        elif receipt.state.unlocks_progress():
            status = ObligationStatus.DURABLE_DEBT
            next_attempt_at_ms = now_ms + self.durable_debt_retry_ms
        else:
            status = ObligationStatus.PENDING
            next_attempt_at_ms = self._retry_at(obligation, now_ms)
        safe_receipt = None
        if receipt is not None:
            safe_receipt = PublicationReceipt(
                request_id=receipt.request_id,
                state=receipt.state,
                detail=self._safe_detail(receipt.detail),
            )
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            updated = obligations_store.mark_attempt(
                conn,
                obligation,
                status=status,
                receipt=safe_receipt,
                error_code=self._safe_code(error_code) if error_code else None,
                now_ms=now_ms,
                next_attempt_at_ms=next_attempt_at_ms,
            )
            conn.commit()
            return updated
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    async def _attempt_async(
        self,
        obligation: PublicationObligation,
        payload: PublicationPayload,
        *,
        on_confirmed: Callable[[PublicationObligation], None] | None = None,
    ) -> PublicationObligation:
        transport = self.transport
        assert transport is not None
        leased = self._lease(obligation)
        if leased.status is not ObligationStatus.PUBLISHING:
            return leased
        receipt: PublicationReceipt | None = None
        error_code: str | None = None
        try:
            receipt = await asyncio.wait_for(
                transport.publish_revision(
                    request_id=leased.request_id,
                    manifest_bytes=payload.manifest_bytes,
                    segment_bytes=payload.segment_bytes,
                ),
                timeout=self.attempt_timeout_s,
            )
            if receipt.request_id != leased.request_id:
                raise ValueError("transport returned a receipt for a different request_id")
        except TimeoutError:
            error_code = "transport_timeout"
        except Exception as exc:
            error_code = f"transport_exception:{type(exc).__name__}"
        updated = self._persist_outcome(leased, receipt=receipt, error_code=error_code)
        if updated.progress_unlocked and on_confirmed is not None:
            on_confirmed(updated)
        return updated

    def _run_transport_sync(self, obligation: PublicationObligation, payload: PublicationPayload) -> PublicationReceipt:
        transport = self.transport
        assert transport is not None

        async def invoke() -> PublicationReceipt:
            return await asyncio.wait_for(
                transport.publish_revision(
                    request_id=obligation.request_id,
                    manifest_bytes=payload.manifest_bytes,
                    segment_bytes=payload.segment_bytes,
                ),
                timeout=self.attempt_timeout_s,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(invoke())
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="sinex-transport") as executor:
            return executor.submit(asyncio.run, invoke()).result()

    def _attempt_sync(self, obligation: PublicationObligation, payload: PublicationPayload) -> PublicationObligation:
        leased = self._lease(obligation)
        if leased.status is not ObligationStatus.PUBLISHING:
            return leased
        receipt: PublicationReceipt | None = None
        error_code: str | None = None
        try:
            receipt = self._run_transport_sync(leased, payload)
            if receipt.request_id != leased.request_id:
                raise ValueError("transport returned a receipt for a different request_id")
        except TimeoutError:
            error_code = "transport_timeout"
        except Exception as exc:
            error_code = f"transport_exception:{type(exc).__name__}"
        return self._persist_outcome(leased, receipt=receipt, error_code=error_code)

    def pending(self, *, object_ids: Sequence[str] | None = None) -> tuple[PublicationObligation, ...]:
        if self.mode is PublicationMode.OFF:
            return ()
        conn = self._connect(readonly=True)
        try:
            return obligations_store.list_obligations(
                conn,
                statuses=_RETRYABLE_STATUSES,
                object_ids=object_ids,
            )
        finally:
            conn.close()

    def lag(self, *, object_ids: Sequence[str] | None = None) -> int:
        """Exact unresolved count, including rejected terminal failures."""
        if self.mode is PublicationMode.OFF:
            return 0
        conn = self._connect(readonly=True)
        try:
            clauses = ["status != 'confirmed'"]
            params: list[object] = []
            if object_ids is not None:
                ids = tuple(dict.fromkeys(str(value) for value in object_ids if value))
                if not ids:
                    return 0
                clauses.append(f"object_id IN ({','.join('?' for _ in ids)})")
                params.extend(ids)
            row = conn.execute(
                f"SELECT COUNT(*) FROM sinex_publication_obligations WHERE {' AND '.join(clauses)}",
                params,
            ).fetchone()
            return int(row[0]) if row is not None else 0
        finally:
            conn.close()

    def has_due_work(self, object_ids: Sequence[str]) -> bool:
        if self.mode is PublicationMode.OFF or not object_ids:
            return False
        conn = self._connect(readonly=True)
        try:
            return bool(
                obligations_store.list_obligations(
                    conn,
                    statuses=_RETRYABLE_STATUSES,
                    object_ids=object_ids,
                    due_at_ms=self.clock(),
                    limit=1,
                )
            )
        finally:
            conn.close()

    def unresolved_object_ids(self, object_ids: Sequence[str]) -> set[str]:
        """Return selected objects with any exact revision not fully confirmed."""
        if self.mode is PublicationMode.OFF or not object_ids:
            return set()
        ids = tuple(dict.fromkeys(str(value) for value in object_ids if value))
        if not ids:
            return set()
        conn = self._connect(readonly=True)
        try:
            rows = conn.execute(
                f"""
                SELECT DISTINCT object_id
                FROM sinex_publication_obligations
                WHERE object_id IN ({",".join("?" for _ in ids)})
                  AND status != 'confirmed'
                """,
                ids,
            ).fetchall()
            return {str(row[0]) for row in rows}
        finally:
            conn.close()

    def blocking_object_ids(self, object_ids: Sequence[str]) -> set[str]:
        """Return objects whose newest accepted revision lacks an allowed receipt.

        Historical revisions remain queryable as lag/debt, but cannot re-block a
        newer confirmed revision.  ``rowid`` is a deterministic tie-breaker for
        multiple accepted revisions staged in the same millisecond.
        """
        if self.mode is not PublicationMode.PRIMARY or not object_ids:
            return set()
        ids = tuple(dict.fromkeys(str(value) for value in object_ids if value))
        if not ids:
            return set()
        conn = self._connect(readonly=True)
        try:
            rows = conn.execute(
                f"""
                WITH ranked AS (
                    SELECT object_id, last_receipt_state,
                           ROW_NUMBER() OVER (
                               PARTITION BY object_id
                               ORDER BY created_at_ms DESC, rowid DESC
                           ) AS revision_rank
                    FROM sinex_publication_obligations
                    WHERE object_id IN ({",".join("?" for _ in ids)})
                )
                SELECT object_id
                FROM ranked
                WHERE revision_rank = 1
                  AND (last_receipt_state IS NULL
                       OR last_receipt_state NOT IN (
                           'persisted_confirmed', 'durable_debt', 'spool_accepted_lossless'
                       ))
                """,
                ids,
            ).fetchall()
            return {str(row[0]) for row in rows}
        finally:
            conn.close()

    def projection_blocked(self, object_ids: Sequence[str]) -> bool:
        """Whether a selected newest primary revision lacks an allowed receipt."""
        return bool(self.blocking_object_ids(object_ids))

    def drain_once(
        self,
        *,
        object_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> DrainSummary:
        """Drain at most ``limit`` due rows and persist every outcome."""
        if self.mode is PublicationMode.OFF:
            return DrainSummary()
        bounded_limit = min(limit or self.max_batch, self.max_batch)
        now_ms = self.clock()
        conn = self._connect(readonly=True)
        try:
            due = obligations_store.list_obligations(
                conn,
                statuses=_RETRYABLE_STATUSES,
                object_ids=object_ids,
                due_at_ms=now_ms,
                limit=bounded_limit,
            )
        finally:
            conn.close()
        confirmed = debt = rejected = deferred = failures = 0
        for obligation in due:
            payload_conn = self._connect(readonly=True)
            try:
                payload = obligations_store.load_payload(payload_conn, obligation)
            except Exception as exc:
                leased = self._lease(obligation)
                if leased.status is ObligationStatus.PUBLISHING:
                    updated = self._persist_outcome(
                        leased,
                        receipt=None,
                        error_code=f"payload_load:{type(exc).__name__}",
                    )
                else:
                    updated = leased
            else:
                updated = self._attempt_sync(obligation, payload)
            finally:
                payload_conn.close()
            if updated.status is ObligationStatus.CONFIRMED:
                confirmed += 1
            elif updated.status is ObligationStatus.DURABLE_DEBT:
                debt += 1
            elif updated.status is ObligationStatus.REJECTED:
                rejected += 1
            elif updated.last_error:
                failures += 1
            else:
                deferred += 1
        return DrainSummary(
            attempted=len(due),
            confirmed=confirmed,
            durable_debt=debt,
            rejected=rejected,
            deferred=deferred,
            transport_failures=failures,
            remaining_lag=self.lag(object_ids=object_ids),
        )

    async def retry_pending(
        self,
        staged: Sequence[tuple[PublicationObligation, bytes, Mapping[str, bytes]]] | None = None,
        *,
        on_confirmed: Callable[[PublicationObligation], None] | None = None,
    ) -> DrainSummary:
        """Compatibility async redrive; durable bytes are authoritative when omitted."""
        if self.mode is PublicationMode.OFF:
            return DrainSummary()
        if staged is None:
            return await asyncio.to_thread(self.drain_once)
        confirmed = debt = rejected = deferred = failures = 0
        for obligation, manifest_bytes, segments in staged[: self.max_batch]:
            payload = PublicationPayload(
                object_id=obligation.object_id,
                protocol_version=obligation.protocol_version,
                revision_id=obligation.revision_id,
                manifest_digest=obligation.manifest_digest,
                manifest_bytes=manifest_bytes,
                segments=tuple(sorted((str(name), bytes(value)) for name, value in segments.items())),
            )
            updated = await self._attempt_async(obligation, payload, on_confirmed=on_confirmed)
            if updated.status is ObligationStatus.CONFIRMED:
                confirmed += 1
            elif updated.status is ObligationStatus.DURABLE_DEBT:
                debt += 1
            elif updated.status is ObligationStatus.REJECTED:
                rejected += 1
            elif updated.last_error:
                failures += 1
            else:
                deferred += 1
        return DrainSummary(
            attempted=min(len(staged), self.max_batch),
            confirmed=confirmed,
            durable_debt=debt,
            rejected=rejected,
            deferred=deferred,
            transport_failures=failures,
            remaining_lag=self.lag(),
        )

    def reset_retryable(self, *, include_rejected: bool = False) -> int:
        if self.mode is PublicationMode.OFF:
            return 0
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            count = obligations_store.reset_retryable(
                conn,
                now_ms=self.clock(),
                include_rejected=include_rejected,
            )
            conn.commit()
            return count
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def status(self) -> PublicationStatus:
        """Return bounded, secret-safe status without payload or raw detail."""
        if self.mode is PublicationMode.OFF:
            return PublicationStatus(mode=self.mode)
        now_ms = self.clock()
        conn = self._connect(readonly=True)
        try:
            rows = conn.execute("SELECT status, COUNT(*) FROM sinex_publication_obligations GROUP BY status").fetchall()
            counts = {str(row[0]): int(row[1]) for row in rows}
            due_row = conn.execute(
                """
                SELECT COUNT(*) FROM sinex_publication_obligations
                WHERE status IN ('pending', 'publishing', 'durable_debt')
                  AND COALESCE(next_attempt_at_ms, created_at_ms) <= ?
                """,
                (now_ms,),
            ).fetchone()
            blocking_row = conn.execute(
                """
                WITH ranked AS (
                    SELECT last_receipt_state,
                           ROW_NUMBER() OVER (
                               PARTITION BY object_id
                               ORDER BY created_at_ms DESC, rowid DESC
                           ) AS revision_rank
                    FROM sinex_publication_obligations
                )
                SELECT COUNT(*) FROM ranked
                WHERE revision_rank = 1
                  AND (last_receipt_state IS NULL
                       OR last_receipt_state NOT IN (
                           'persisted_confirmed', 'durable_debt', 'spool_accepted_lossless'
                       ))
                """
            ).fetchone()
            oldest_row = conn.execute(
                """
                SELECT MIN(created_at_ms) FROM sinex_publication_obligations
                WHERE status != 'confirmed'
                """
            ).fetchone()
            recent = conn.execute(
                """
                SELECT receipt_state, error_code
                FROM sinex_publication_receipts
                ORDER BY received_at_ms DESC, attempt_number DESC LIMIT 1
                """
            ).fetchone()
            oldest = int(oldest_row[0]) if oldest_row is not None and oldest_row[0] is not None else None
            receipt_state = ReceiptState(str(recent[0])) if recent is not None and recent[0] is not None else None
            error_code = str(recent[1]) if recent is not None and recent[1] is not None else None
            total = sum(counts.values())
            active_lag = total - counts.get("confirmed", 0)
            return PublicationStatus(
                mode=self.mode,
                total=total,
                pending=counts.get("pending", 0),
                publishing=counts.get("publishing", 0),
                confirmed=counts.get("confirmed", 0),
                durable_debt=counts.get("durable_debt", 0),
                rejected=counts.get("rejected", 0),
                retry_due=int(due_row[0]) if due_row is not None else 0,
                blocking=(
                    int(blocking_row[0]) if self.mode is PublicationMode.PRIMARY and blocking_row is not None else 0
                ),
                active_lag=active_lag,
                oldest_active_age_ms=max(0, now_ms - oldest) if oldest is not None else None,
                last_receipt_state=receipt_state,
                last_error_code=error_code,
            )
        finally:
            conn.close()


__all__ = ["DrainSummary", "PublicationService"]
