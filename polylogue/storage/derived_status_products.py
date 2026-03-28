"""Derived-model statuses for session products and archive read models."""

from __future__ import annotations

from polylogue.maintenance_models import DerivedModelStatus
from polylogue.storage.derived_status_products_actions import build_action_statuses
from polylogue.storage.derived_status_products_aggregates import build_aggregate_statuses
from polylogue.storage.derived_status_products_profiles import build_profile_statuses
from polylogue.storage.derived_status_products_timelines import build_timeline_statuses


def build_archive_product_statuses(metrics: dict[str, int | bool]) -> dict[str, DerivedModelStatus]:
    return {
        **build_action_statuses(metrics),
        **build_profile_statuses(metrics),
        **build_timeline_statuses(metrics),
        **build_aggregate_statuses(metrics),
    }


__all__ = ["build_archive_product_statuses"]
