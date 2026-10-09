"""Embedding readiness snapshot for daemon status surfaces."""

from __future__ import annotations

from pathlib import Path

import polylogue.config as polylogue_config
from polylogue.core.status_error_privacy import redact_status_error
from polylogue.logging import WARNING, emit
from polylogue.operations.embedding_readiness import EmbeddingReadinessUnavailableError, read_embedding_readiness
from polylogue.storage.embeddings.status_payload import EmbeddingCatchupRunPayload


def embedding_readiness_settings() -> dict[str, object]:
    """Read current embedding policy without collecting archive measurements."""
    cfg = polylogue_config.load_polylogue_config()
    config_enabled = bool(cfg.embedding_enabled)
    has_key = cfg.voyage_api_key is not None
    return {
        "embedding_enabled": config_enabled and has_key,
        "embedding_config_enabled": config_enabled,
        "embedding_has_voyage_key": has_key,
        "embedding_model": cfg.embedding_model,
        "embedding_dimension": cfg.embedding_dimension,
    }


def _defaults(settings: dict[str, object], *, unreadable: bool = False) -> dict[str, object]:
    return {
        **settings,
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

    settings = embedding_readiness_settings()
    from polylogue.storage.archive_identity import ArchiveLocation

    index_db = ArchiveLocation.resolve(db_file.parent).active_index_path
    if not db_file.exists() and not index_db.exists():
        return _defaults(settings)

    try:
        payload = read_embedding_readiness(db_file, detail=detail)
    except EmbeddingReadinessUnavailableError as exc:
        emit(
            "daemon.embed.readiness_query_failed",
            level=WARNING,
            outcome="degraded",
            reason="readiness_unreadable",
            path=db_file,
            error_type=type(exc.__cause__).__name__,
            error_detail=str(exc.__cause__),
        )
        return _defaults(settings, unreadable=True)

    return {
        **settings,
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


__all__ = ["embedding_readiness_info", "embedding_readiness_settings"]
