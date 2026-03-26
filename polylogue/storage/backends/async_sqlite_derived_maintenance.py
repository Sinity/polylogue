"""Derived maintenance lineage writes for the async SQLite backend."""

from __future__ import annotations

from polylogue.storage.backends.queries import maintenance_runs as maintenance_runs_q
from polylogue.storage.store import MaintenanceRunRecord


class SQLiteDerivedMaintenanceMixin:
    """Derived maintenance methods for ``SQLiteBackend``."""

    async def record_maintenance_run(
        self,
        record: MaintenanceRunRecord,
    ) -> None:
        """Persist one maintenance lineage record."""
        async with self._get_connection() as conn:
            await maintenance_runs_q.record_maintenance_run(
                conn,
                record,
                self._transaction_depth,
            )


__all__ = ["SQLiteDerivedMaintenanceMixin"]
