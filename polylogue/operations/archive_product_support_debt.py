"""Debt-product mixin (simplified — governance lineage removed)."""

from __future__ import annotations

from polylogue.archive_products import (
    ArchiveDebtProduct,
    ArchiveDebtProductQuery,
)
from polylogue.storage.backends.connection import connection_context
from polylogue.storage.repair import collect_archive_debt_statuses_sync


class ArchiveProductDebtMixin:
    async def list_archive_debt_products(
        self,
        query: ArchiveDebtProductQuery | None = None,
    ) -> list[ArchiveDebtProduct]:
        request = query or ArchiveDebtProductQuery()
        with connection_context(self.config.db_path) as conn:
            statuses = collect_archive_debt_statuses_sync(conn)
        products = [
            ArchiveDebtProduct.from_status(status)
            for status in statuses.values()
        ]
        products.sort(key=lambda product: (product.category, product.debt_name))
        if request.category:
            products = [product for product in products if product.category == request.category]
        if request.only_actionable:
            products = [product for product in products if not product.healthy]
        if request.offset:
            products = products[request.offset:]
        if request.limit is not None:
            products = products[:request.limit]
        return products


__all__ = ["ArchiveProductDebtMixin"]
