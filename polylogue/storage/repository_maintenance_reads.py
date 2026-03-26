"""Readiness/status/maintenance read methods for the repository."""

from __future__ import annotations


class RepositoryMaintenanceReadMixin:
    async def get_action_event_read_model_status(self) -> dict[str, int | bool]:
        return await self.queries.get_action_event_read_model_status()

    async def get_session_product_status(self) -> dict[str, int | bool]:
        return await self.queries.get_session_product_status()

    async def list_maintenance_runs(self, *, limit: int = 20):
        return await self.queries.list_maintenance_runs(limit=limit)


__all__ = ["RepositoryMaintenanceReadMixin"]
