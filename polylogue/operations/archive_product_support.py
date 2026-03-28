"""Small public root for archive-product support mixins."""

from __future__ import annotations

from polylogue.operations.archive_product_support_analytics import ArchiveProductAggregateMixin
from polylogue.operations.archive_product_support_debt import ArchiveProductDebtMixin
from polylogue.operations.archive_product_support_session import ArchiveProductSessionMixin


class ArchiveProductMixin(
    ArchiveProductSessionMixin,
    ArchiveProductAggregateMixin,
    ArchiveProductDebtMixin,
):
    """Versioned archive-product retrieval methods."""


__all__ = ["ArchiveProductMixin"]
