"""Embedding readiness snapshot for daemon status surfaces."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import polylogue.config as polylogue_config
from polylogue.logging import get_logger
from polylogue.storage.embeddings.status_payload import embedding_status_payload

logger = get_logger(__name__)


def _defaults(*, enabled: bool, config_enabled: bool, has_key: bool, model: str, dimension: int) -> dict[str, object]:
    return {
        "embedding_enabled": enabled,
        "embedding_config_enabled": config_enabled,
        "embedding_has_voyage_key": has_key,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "embedding_status": "empty",
        "embedding_freshness_status": "empty",
        "embedding_retrieval_ready": False,
        "embedding_pending_count": 0,
        "embedding_pending_message_count": 0,
        "embedding_pending_message_count_exact": False,
        "embedding_stale_count": 0,
        "embedding_coverage_percent": 0.0,
        "embedding_failure_count": 0,
        "embedding_failure_details": [],
        "embedding_estimated_cost_usd": 0.0,
        "embedding_latest_catchup_run": None,
        "embedding_latest_material_catchup_run": None,
    }


def embedding_readiness_info(db_file: Path, *, detail: bool = False) -> dict[str, object]:
    """Query embedding tables for bounded daemon status visibility."""

    cfg = polylogue_config.load_polylogue_config()
    config_enabled = bool(cfg.embedding_enabled)
    has_key = cfg.voyage_api_key is not None
    enabled = config_enabled and has_key
    model = cfg.embedding_model
    dimension = cfg.embedding_dimension
    if not db_file.exists() and not db_file.with_name("index.db").exists():
        return _defaults(
            enabled=enabled,
            config_enabled=config_enabled,
            has_key=has_key,
            model=model,
            dimension=dimension,
        )

    try:
        payload = embedding_status_payload(
            SimpleNamespace(config=SimpleNamespace(db_path=db_file)),
            include_retrieval_bands=False,
            include_detail=detail,
        )
    except (sqlite3.Error, OSError) as exc:
        # _defaults() reports embedding_status="empty" / retrieval_ready=False
        # / pending counts of 0 — identical to a genuinely fresh archive with
        # no embeddings yet. Log loudly so a transient DB error doesn't read
        # as "nothing to embed" (polylogue-cpf.4).
        logger.warning("embedding readiness query failed for %s: %s", db_file, exc, exc_info=True)
        return _defaults(
            enabled=enabled,
            config_enabled=config_enabled,
            has_key=has_key,
            model=model,
            dimension=dimension,
        )

    return {
        "embedding_enabled": enabled,
        "embedding_config_enabled": config_enabled,
        "embedding_has_voyage_key": has_key,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "embedding_status": payload["status"],
        "embedding_freshness_status": payload["freshness_status"],
        "embedding_retrieval_ready": payload["retrieval_ready"],
        "embedding_pending_count": payload["pending_sessions"],
        "embedding_pending_message_count": payload["pending_messages"],
        "embedding_pending_message_count_exact": payload["pending_messages_exact"],
        "embedding_stale_count": payload["stale_messages"],
        "embedding_coverage_percent": payload["embedding_coverage_percent"],
        "embedding_failure_count": payload["failure_count"],
        "embedding_failure_details": payload["failure_details"],
        "embedding_estimated_cost_usd": payload["total_estimated_cost_usd"],
        "embedding_latest_catchup_run": payload["latest_catchup_run"],
        "embedding_latest_material_catchup_run": payload["latest_material_catchup_run"],
    }


__all__ = ["embedding_readiness_info"]
