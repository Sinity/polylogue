"""Expose original-owner SQL settlement without exporting SQLite backends."""

from __future__ import annotations

from polylogue.core.sql_settlement import AsyncSQLCustodyOwner, SQLCustodyOwner
from polylogue.storage.sqlite.async_sqlite import retained_write_backends_on_current_thread
from polylogue.storage.sqlite.connection_profile import (
    retained_native_settlement_owners_on_current_thread,
    settle_cached_connections_on_current_thread,
)
from polylogue.storage.sqlite.reference_seal import retained_reference_seals_on_current_thread
from polylogue.storage.sqlite.write_lease import WriteLease, retained_sql_owners_on_current_thread


def retained_sync_sql_owners() -> tuple[SQLCustodyOwner, ...]:
    owners = (
        *retained_sql_owners_on_current_thread(),
        *retained_reference_seals_on_current_thread(),
        *retained_native_settlement_owners_on_current_thread(),
    )
    return tuple({id(owner): owner for owner in owners}.values())


def retained_async_sql_owners(*, lease: object | None = None) -> tuple[AsyncSQLCustodyOwner, ...]:
    if lease is not None and not isinstance(lease, WriteLease):
        raise TypeError("SQL settlement requires the actual write lease")
    return retained_write_backends_on_current_thread(lease=lease)


def settle_cached_sql(custody: object) -> None:
    settle_cached_connections_on_current_thread(custody)
