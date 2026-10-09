"""Embedding readiness snapshot for daemon status surfaces."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import polylogue.config as polylogue_config
from polylogue.core.status_error_privacy import redact_status_error
from polylogue.logging import WARNING, emit
from polylogue.storage.embeddings.status_payload import EmbeddingCatchupRunPayload, embedding_status_payload


def _defaults(
    *, enabled: bool, config_enabled: bool, has_key: bool, model: str, dimension: int, unreadable: bool = False
) -> dict[str, object]:
    return {
        "embedding_enabled": enabled,
        "embedding_config_enabled": config_enabled,
        "embedding_has_voyage_key": has_key,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "embedding_status": "unknown" if unreadable else "empty",
        "embedding_freshness_status": "unknown" if unreadable else "empty",
        "embedding_unmeasurable_reason": "readiness_unreadable" if unreadable else None,
        "embedding_retrieval_ready": False,
        "embedding_pending_count": None if unreadable else 0,
        "embedding_pending_message_count": None if unreadable else 0,
        "embedding_pending_message_count_exact": False,
        "embedding_stale_count": None if unreadable else 0,
        "embedding_coverage_percent": None if unreadable else 0.0,
        "embedding_failure_count": None if unreadable else 0,
        "embedding_terminal_failure_count": None if unreadable else 0,
        "embedding_retryable_failure_count": None if unreadable else 0,
        "embedding_failure_details": [],
        "embedding_estimated_cost_usd": None if unreadable else 0.0,
        "embedding_latest_catchup_run": None,
        "embedding_latest_material_catchup_run": None,
    }


def _private_run(run: EmbeddingCatchupRunPayload | None) -> EmbeddingCatchupRunPayload | None:
    if run is None:
        return None
    reason = run["stop_reason"]
    return {**run, "stop_reason": redact_status_error(reason) if reason is not None else None}


def embedding_readiness_info(db_file: Path, *, detail: bool = False) -> dict[str, object]:
    """Query embedding tables for bounded daemon status visibility."""

    cfg = polylogue_config.load_polylogue_config()
    from polylogue.storage.archive_identity import ArchiveLocation

    config_enabled = bool(cfg.embedding_enabled)
    has_key = cfg.voyage_api_key is not None
    enabled = config_enabled and has_key
    model = cfg.embedding_model
    dimension = cfg.embedding_dimension
    index_db = ArchiveLocation.resolve(db_file.parent).active_index_path
    if not db_file.exists() and not index_db.exists():
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
        emit(
            "daemon.embed.readiness_query_failed",
            level=WARNING,
            outcome="degraded",
            reason="readiness_unreadable",
            path=db_file,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return _defaults(
            enabled=enabled,
            config_enabled=config_enabled,
            has_key=has_key,
            model=model,
            dimension=dimension,
            unreadable=True,
        )

    return {
        "embedding_enabled": enabled,
        "embedding_config_enabled": config_enabled,
        "embedding_has_voyage_key": has_key,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "embedding_status": payload["status"],
        "embedding_freshness_status": payload["freshness_status"],
        "embedding_unmeasurable_reason": payload["coverage_unmeasurable_reason"],
        "embedding_retrieval_ready": payload["retrieval_ready"],
        "embedding_pending_count": payload["pending_sessions"],
        "embedding_pending_message_count": payload["pending_messages"],
        "embedding_pending_message_count_exact": payload["pending_messages_exact"],
        "embedding_stale_count": payload["stale_messages"],
        "embedding_coverage_percent": payload["embedding_coverage_percent"],
        "embedding_failure_count": payload["failure_count"],
        "embedding_terminal_failure_count": payload["terminal_failure_count"],
        "embedding_retryable_failure_count": payload["retryable_failure_count"],
        "embedding_failure_details": [
            {**detail, "error_message": redact_status_error(detail["error_message"])}
            for detail in payload["failure_details"]
        ],
        "embedding_estimated_cost_usd": payload["total_estimated_cost_usd"],
        "embedding_latest_catchup_run": _private_run(payload["latest_catchup_run"]),
        "embedding_latest_material_catchup_run": _private_run(payload["latest_material_catchup_run"]),
    }


__all__ = ["embedding_readiness_info"]
