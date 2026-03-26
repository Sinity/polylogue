"""Governance and status archive-product workflow helpers."""

from __future__ import annotations

from polylogue.archive_products import (
    ArchiveDebtProductQuery,
    MaintenanceRunProductQuery,
)
from polylogue.cli.types import AppEnv
from polylogue.sync_bridge import run_coroutine_sync


def get_products_status(env: AppEnv) -> tuple[dict[str, int | bool], list[object]]:
    status = run_coroutine_sync(env.operations.get_session_product_status())
    debt_items = run_coroutine_sync(
        env.operations.list_archive_debt_products(ArchiveDebtProductQuery())
    )
    return status, debt_items


def list_maintenance_run_products(env: AppEnv, *, limit: int) -> list[object]:
    return run_coroutine_sync(
        env.operations.list_maintenance_run_products(
            MaintenanceRunProductQuery(limit=limit)
        )
    )


def list_archive_debt_products(
    env: AppEnv,
    *,
    category: str | None,
    actionable_only: bool,
    limit: int,
    offset: int,
) -> list[object]:
    return run_coroutine_sync(
        env.operations.list_archive_debt_products(
            ArchiveDebtProductQuery(
                category=category,
                only_actionable=actionable_only,
                limit=limit,
                offset=offset,
            )
        )
    )


__all__ = [
    "get_products_status",
    "list_archive_debt_products",
    "list_maintenance_run_products",
]
