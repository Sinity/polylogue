"""Profile/status derived-model builders."""

from __future__ import annotations

from polylogue.maintenance_models import DerivedModelStatus
from polylogue.storage.derived_status_support import pending_rows
from polylogue.storage.store import SESSION_PRODUCT_MATERIALIZER_VERSION


def build_profile_fts_status(
    metrics: dict[str, int | bool],
    *,
    key_prefix: str,
    name: str,
    label: str,
) -> DerivedModelStatus:
    ready_key = f"{key_prefix}_ready"
    rows_key = f"{key_prefix}_rows"
    duplicate_key = f"{key_prefix}_duplicates"
    return DerivedModelStatus(
        name=name,
        ready=bool(metrics[ready_key]),
        detail=(
            f"{label} ready ({metrics[rows_key]:,}/{metrics['profile_rows']:,} rows)"
            if bool(metrics[ready_key])
            else (
                f"{label} pending ({metrics[rows_key]:,}/{metrics['profile_rows']:,} rows, "
                f"duplicates {metrics[duplicate_key]:,})"
            )
        ),
        source_rows=int(metrics["profile_rows"]),
        materialized_rows=int(metrics[rows_key]),
        pending_rows=pending_rows(int(metrics["profile_rows"]), int(metrics[rows_key])),
        stale_rows=int(metrics[duplicate_key]),
    )


def build_profile_statuses(metrics: dict[str, int | bool]) -> dict[str, DerivedModelStatus]:
    return {
        "session_profile_rows": DerivedModelStatus(
            name="session_profile_rows",
            ready=bool(metrics["profile_rows_ready"]),
            detail=(
                f"Session-profile rows ready ({metrics['profile_rows']:,}/{metrics['total_conversations']:,} conversations)"
                if bool(metrics["profile_rows_ready"])
                else f"Session-profile rows pending ({metrics['profile_rows']:,}/{metrics['total_conversations']:,} conversations)"
            ),
            source_documents=int(metrics["total_conversations"]),
            materialized_documents=int(metrics["profile_rows"]),
            pending_documents=int(metrics["missing_profile_rows"]),
            stale_rows=int(metrics["stale_profile_rows"]),
            orphan_rows=int(metrics["orphan_profile_rows"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=bool(int(metrics["stale_profile_rows"]) == 0 and int(metrics["orphan_profile_rows"]) == 0),
        ),
        "session_profile_merged_fts": build_profile_fts_status(
            metrics,
            key_prefix="profile_merged_fts",
            name="session_profile_merged_fts",
            label="Session-profile merged FTS",
        ),
        "session_profile_evidence_fts": build_profile_fts_status(
            metrics,
            key_prefix="profile_evidence_fts",
            name="session_profile_evidence_fts",
            label="Session-profile evidence FTS",
        ),
        "session_profile_inference_fts": build_profile_fts_status(
            metrics,
            key_prefix="profile_inference_fts",
            name="session_profile_inference_fts",
            label="Session-profile inference FTS",
        ),
        "session_profile_enrichment_fts": build_profile_fts_status(
            metrics,
            key_prefix="profile_enrichment_fts",
            name="session_profile_enrichment_fts",
            label="Session-profile enrichment FTS",
        ),
    }


__all__ = ["build_profile_statuses"]
