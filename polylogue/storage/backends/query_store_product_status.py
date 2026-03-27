"""Derived product/action read-model status band for SQLiteQueryStore."""

from __future__ import annotations


class SQLiteQueryStoreProductStatusMixin:
    async def get_action_event_read_model_status(self) -> dict[str, int | bool]:
        from polylogue.storage.action_event_status import action_event_read_model_status_async

        async with self._connection_factory() as conn:
            return await action_event_read_model_status_async(conn)

    async def get_session_product_status(self) -> dict[str, int | bool]:
        from polylogue.storage.session_product_status import session_product_status_async

        async with self._connection_factory() as conn:
            return await session_product_status_async(conn)


__all__ = ["SQLiteQueryStoreProductStatusMixin"]
