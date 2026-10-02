"""Async adapter to the sole source-tier raw-admission writer."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterable

import aiosqlite

from polylogue.core.enums import Origin, Provider
from polylogue.storage.sqlite.archive_tiers.common import require_vocabulary
from polylogue.storage.sqlite.archive_tiers.raw_admission import (
    RawAdmissionExecution,
    RawAdmissionPlan,
    execute_raw_admission_plan_sync,
)
from polylogue.storage.sqlite.archive_tiers.source_items import (
    FrozenSourceManifest,
    SourceItemMemberDisposition,
    complete_source_item_enumeration,
    record_source_item_member_disposition,
)
from polylogue.storage.sqlite.archive_tiers.source_items import (
    publish_acquired_zip_input as publish_acquired_zip_input_sync,
)


def _apply_plan(owner: aiosqlite.Connection, plan: RawAdmissionPlan, transaction_depth: int) -> RawAdmissionExecution:
    """Run one complete admission on the connection's actual SQLite worker."""
    conn = owner._conn
    inserted = conn.execute("SELECT 1 FROM raw_sessions WHERE raw_id=?", (plan.raw_id,)).fetchone() is None
    result = execute_raw_admission_plan_sync(conn, plan, manage_transaction=transaction_depth == 0)
    return RawAdmissionExecution(result=result, inserted=inserted)


async def execute_raw_admission_plan_async(
    conn: aiosqlite.Connection,
    plan: RawAdmissionPlan,
    transaction_depth: int,
) -> RawAdmissionExecution:
    """Queue the canonical writer without transporting its native connection.

    Identity, profile and container receipts, renewed observations and conflict
    rollback have one implementation. Cancellation propagates only after the
    actual queued operation settles, so callers cannot reuse or close the
    connection while admission still owns it.
    """
    require_vocabulary(plan.request.origin, Origin, field="origin")
    if plan.request.capture_mode is not None:
        require_vocabulary(plan.request.capture_mode, Provider, field="capture_mode")
    # Import lazily: the backend loads its query adapters during construction.
    from polylogue.storage.sqlite.async_sqlite import _await_settled

    task: asyncio.Task[RawAdmissionExecution] = asyncio.ensure_future(
        conn._execute(_apply_plan, conn, plan, transaction_depth)  # type: ignore[no-untyped-call]
    )
    await _await_settled(task)
    return task.result()


__all__ = ["execute_raw_admission_plan_async"]


def _publish_input(owner: aiosqlite.Connection, manifest: FrozenSourceManifest, observed_at_ms: int, depth: int) -> str:
    conn = owner._conn
    conn.execute("SAVEPOINT acquired_zip_input")
    try:
        item = publish_acquired_zip_input_sync(conn, manifest, observed_at_ms=observed_at_ms)
    except BaseException:
        conn.execute("ROLLBACK TO acquired_zip_input")
        conn.execute("RELEASE acquired_zip_input")
        raise
    conn.execute("RELEASE acquired_zip_input")
    if depth == 0:
        conn.commit()
    return item


async def publish_acquired_zip_input(
    conn: aiosqlite.Connection,
    manifest: FrozenSourceManifest,
    *,
    observed_at_ms: int,
    transaction_depth: int,
) -> str:
    from polylogue.storage.sqlite.async_sqlite import _await_settled

    task: asyncio.Task[str] = asyncio.ensure_future(
        conn._execute(_publish_input, conn, manifest, observed_at_ms, transaction_depth)  # type: ignore[no-untyped-call]
    )
    await _await_settled(task)
    return task.result()


def _record_input_disposition(
    owner: aiosqlite.Connection,
    generation: str,
    item: str,
    ordinal: int,
    name: str,
    disposition: str,
    diagnostic: str | None,
    observed_at_ms: int,
    depth: int,
) -> None:
    conn = owner._conn
    conn.execute("SAVEPOINT acquired_zip_disposition")
    try:
        record_source_item_member_disposition(
            conn,
            source_generation_id=generation,
            source_item_id=item,
            entry_ordinal=ordinal,
            member_name=name,
            disposition=SourceItemMemberDisposition(disposition),
            diagnostic=diagnostic or "",
            observed_at_ms=observed_at_ms,
        )
    except BaseException:
        conn.execute("ROLLBACK TO acquired_zip_disposition")
        conn.execute("RELEASE acquired_zip_disposition")
        raise
    conn.execute("RELEASE acquired_zip_disposition")
    if depth == 0:
        conn.commit()


async def record_acquired_zip_disposition(
    conn: aiosqlite.Connection,
    *,
    source_generation_id: str,
    source_item_id: str,
    entry_ordinal: int,
    member_name: str,
    disposition: str,
    diagnostic: str | None,
    observed_at_ms: int,
    transaction_depth: int,
) -> None:
    from polylogue.storage.sqlite.async_sqlite import _await_settled

    task: asyncio.Task[None] = asyncio.ensure_future(
        conn._execute(  # type: ignore[no-untyped-call]
            _record_input_disposition,
            conn,
            source_generation_id,
            source_item_id,
            entry_ordinal,
            member_name,
            disposition,
            diagnostic,
            observed_at_ms,
            transaction_depth,
        )
    )
    await _await_settled(task)
    task.result()


def _complete_input(
    owner: aiosqlite.Connection,
    generation: str,
    item: str,
    fingerprint: str,
    coordinates: Iterable[str],
    member_count: int,
    observed_at_ms: int,
    depth: int,
    check_stop: Callable[[], None],
) -> str:
    conn = owner._conn
    conn.execute("SAVEPOINT acquired_zip_completion")
    try:
        result = complete_source_item_enumeration(
            conn,
            source_generation_id=generation,
            source_item_id=item,
            enumeration_fingerprint=fingerprint,
            record_coordinates=coordinates,
            member_ordinals=range(member_count),
            member_count=member_count,
            enumerated_at_ms=observed_at_ms,
            check_stop=check_stop,
        )
    except BaseException:
        conn.execute("ROLLBACK TO acquired_zip_completion")
        conn.execute("RELEASE acquired_zip_completion")
        raise
    conn.execute("RELEASE acquired_zip_completion")
    if depth == 0:
        conn.commit()
    return result


async def complete_acquired_zip_input(
    conn: aiosqlite.Connection,
    *,
    source_generation_id: str,
    source_item_id: str,
    enumeration_fingerprint: str,
    record_coordinates: Iterable[str],
    member_count: int,
    observed_at_ms: int,
    transaction_depth: int,
) -> str:
    from polylogue.storage.sqlite.async_sqlite import _await_settled

    stopped = threading.Event()

    def checkpoint() -> None:
        if stopped.is_set():
            raise asyncio.CancelledError()

    task: asyncio.Task[str] = asyncio.ensure_future(
        conn._execute(  # type: ignore[no-untyped-call]
            _complete_input,
            conn,
            source_generation_id,
            source_item_id,
            enumeration_fingerprint,
            record_coordinates,
            member_count,
            observed_at_ms,
            transaction_depth,
            checkpoint,
        )
    )
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        stopped.set()
        await _await_settled(task)
        raise
    await _await_settled(task)
    return task.result()
