"""Archive/data health checks."""

from __future__ import annotations

from typing import Any, cast

from polylogue.logging import get_logger

from .config import Config
from .health_cache import load_cached_report, write_cache
from .health_models import HealthCheck, HealthReport, VerifyStatus
from .lib.provider_identity import CORE_SCHEMA_PROVIDERS
from .sources.drive_client import default_credentials_path, default_token_path
from .storage.backends.connection import connection_context, open_connection
from .storage.embedding_stats import read_embedding_stats_sync
from .storage.index import index_status

logger = get_logger(__name__)


def run_archive_health(config: Config, *, deep: bool = False) -> HealthReport:
    """Run comprehensive system health and data verification checks."""
    checks: list[HealthCheck] = []

    checks.append(HealthCheck("config", VerifyStatus.OK, summary="Zero-config (XDG paths)"))

    for path_name in ("archive_root", "render_root"):
        path = getattr(config, path_name)
        if path.exists():
            checks.append(HealthCheck(path_name, VerifyStatus.OK, summary=str(path)))
        else:
            checks.append(HealthCheck(path_name, VerifyStatus.WARNING, summary=f"Missing {path}"))

    db_error: str | None = None
    try:
        with open_connection(None) as conn:
            checks.append(HealthCheck("database", VerifyStatus.OK, summary="DB reachable"))
            if deep:
                integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                checks.append(
                    HealthCheck(
                        "sqlite_integrity",
                        VerifyStatus.OK if integrity == "ok" else VerifyStatus.ERROR,
                        summary=integrity,
                    )
                )
    except Exception as exc:
        db_error = str(exc)
        checks.append(HealthCheck("database", VerifyStatus.ERROR, summary=f"DB error: {db_error}"))

    if db_error is None:
        idx = index_status()
        if idx["exists"]:
            checks.append(
                HealthCheck(
                    "index",
                    VerifyStatus.OK,
                    count=int(cast(Any, idx["count"])),
                    summary=f"messages indexed: {idx['count']}",
                )
            )
        else:
            checks.append(HealthCheck("index", VerifyStatus.WARNING, summary="index not built"))
    else:
        checks.append(
            HealthCheck(
                "index",
                VerifyStatus.WARNING,
                summary=f"Skipped: database unavailable ({db_error})",
            )
        )

    if db_error is None:
        with connection_context(None) as conn:
            orphan_cids = conn.execute(
                """
                SELECT DISTINCT m.conversation_id FROM messages m
                WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = m.conversation_id)
                """
            ).fetchall()
            if orphan_cids:
                placeholders = ",".join("?" for _ in orphan_cids)
                orphan_count = conn.execute(
                    f"SELECT COUNT(*) FROM messages WHERE conversation_id IN ({placeholders})",
                    [row[0] for row in orphan_cids],
                ).fetchone()[0]
            else:
                orphan_count = 0
            checks.append(
                HealthCheck(
                    "orphaned_messages",
                    VerifyStatus.OK if orphan_count == 0 else VerifyStatus.ERROR,
                    count=orphan_count,
                    summary=(
                        "No orphaned messages"
                        if orphan_count == 0
                        else f"{orphan_count:,} orphaned messages"
                    ),
                )
            )

            dup_conv = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT conversation_id FROM conversations GROUP BY conversation_id HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
            checks.append(
                HealthCheck(
                    "duplicate_conversations",
                    VerifyStatus.OK if dup_conv == 0 else VerifyStatus.ERROR,
                    count=dup_conv,
                    summary=(
                        "No duplicates"
                        if dup_conv == 0
                        else f"{dup_conv} duplicate conversation IDs"
                    ),
                )
            )

            empty_conv = conn.execute(
                """
                SELECT COUNT(*) FROM conversations c
                WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.conversation_id)
                """
            ).fetchone()[0]
            checks.append(
                HealthCheck(
                    "empty_conversations",
                    VerifyStatus.OK if empty_conv == 0 else VerifyStatus.WARNING,
                    count=empty_conv,
                    summary=(
                        "No empty conversations"
                        if empty_conv == 0
                        else f"{empty_conv} conversation(s) with no messages"
                    ),
                )
            )

            provider_rows = conn.execute(
                """
                SELECT provider_name, COUNT(*) AS count
                FROM conversations
                GROUP BY provider_name
                ORDER BY count DESC, provider_name ASC
                """
            ).fetchall()
            provider_breakdown = {
                str(row["provider_name"]): int(row["count"])
                for row in provider_rows
            }
            checks.append(
                HealthCheck(
                    "provider_distribution",
                    VerifyStatus.OK,
                    count=sum(provider_breakdown.values()),
                    summary=f"{len(provider_breakdown)} provider(s) represented",
                    breakdown=provider_breakdown,
                )
            )

            try:
                fts_exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
                ).fetchone()

                if fts_exists:
                    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                    fts_count = conn.execute(
                        "SELECT COUNT(*) FROM messages_fts_docsize"
                    ).fetchone()[0]

                    if msg_count == fts_count:
                        checks.append(
                            HealthCheck(
                                "fts_sync",
                                VerifyStatus.OK,
                                count=fts_count,
                                summary=f"FTS index in sync ({fts_count:,} messages indexed)",
                            )
                        )
                    else:
                        checks.append(
                            HealthCheck(
                                "fts_sync",
                                VerifyStatus.WARNING,
                                count=abs(msg_count - fts_count),
                                summary=(
                                    f"FTS out of sync: {msg_count:,} messages vs "
                                    f"{fts_count:,} indexed"
                                ),
                            )
                        )
                else:
                    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                    checks.append(
                        HealthCheck(
                            "fts_sync",
                            VerifyStatus.WARNING,
                            count=msg_count,
                            summary=f"FTS index not built ({msg_count:,} messages not indexed)",
                    )
                )
            except Exception as exc:
                checks.append(
                    HealthCheck(
                        "fts_sync",
                        VerifyStatus.ERROR,
                        summary=f"FTS check failed: {exc}",
                    )
                )

            total_conversations = sum(provider_breakdown.values())
            try:
                embedding_stats = read_embedding_stats_sync(conn)
                embedded_conversations = embedding_stats.embedded_conversations
                embedded_messages = embedding_stats.embedded_messages
                pending_conversations = embedding_stats.pending_conversations
                embedding_coverage = (
                    (embedded_conversations / total_conversations) * 100 if total_conversations else 0.0
                )

                if total_conversations == 0:
                    checks.append(
                        HealthCheck(
                            "embedding_coverage",
                            VerifyStatus.OK,
                            summary="No conversations to embed",
                        )
                    )
                elif embedded_conversations == 0 and embedded_messages == 0 and pending_conversations == 0:
                    checks.append(
                        HealthCheck(
                            "embedding_coverage",
                            VerifyStatus.WARNING,
                            count=0,
                            summary=f"Embeddings not built (0/{total_conversations:,} conversations embedded)",
                        )
                    )
                elif pending_conversations > 0:
                    checks.append(
                        HealthCheck(
                            "embedding_coverage",
                            VerifyStatus.WARNING,
                            count=pending_conversations,
                            summary=(
                                f"Embeddings partial ({embedded_conversations:,}/{total_conversations:,} conversations, "
                                f"{embedded_messages:,} messages, pending {pending_conversations:,}, "
                                f"coverage {embedding_coverage:.1f}%)"
                            ),
                        )
                    )
                else:
                    checks.append(
                        HealthCheck(
                            "embedding_coverage",
                            VerifyStatus.OK,
                            count=embedded_conversations,
                            summary=(
                                f"Embeddings ready ({embedded_conversations:,}/{total_conversations:,} conversations, "
                                f"{embedded_messages:,} messages, coverage {embedding_coverage:.1f}%)"
                            ),
                        )
                    )

                if embedded_messages == 0:
                    checks.append(
                        HealthCheck(
                            "embedding_freshness",
                            VerifyStatus.OK,
                            summary="No embedded messages to assess freshness",
                        )
                    )
                elif embedding_stats.stale_messages > 0 or embedding_stats.messages_missing_provenance > 0:
                    model_summary = ", ".join(
                        f"{name} ({count})"
                        for name, count in embedding_stats.model_counts.items()
                    ) or "unknown"
                    checks.append(
                        HealthCheck(
                            "embedding_freshness",
                            VerifyStatus.WARNING,
                            count=embedding_stats.stale_messages,
                            summary=(
                                f"Embeddings stale ({embedding_stats.stale_messages:,} stale, "
                                f"{embedding_stats.messages_missing_provenance:,} missing provenance, "
                                f"models: {model_summary})"
                            ),
                        )
                    )
                else:
                    model_summary = ", ".join(
                        f"{name} ({count})"
                        for name, count in embedding_stats.model_counts.items()
                    ) or "unknown"
                    checks.append(
                        HealthCheck(
                            "embedding_freshness",
                            VerifyStatus.OK,
                            count=embedded_messages,
                            summary=(
                                f"Embeddings fresh ({embedded_messages:,} messages, "
                                f"models: {model_summary}, latest {embedding_stats.newest_embedded_at or 'unknown'})"
                            ),
                        )
                    )
            except Exception as exc:
                checks.append(
                    HealthCheck(
                        "embedding_coverage",
                        VerifyStatus.ERROR,
                        summary=f"Embedding coverage check failed: {exc}",
                    )
                )
                checks.append(
                    HealthCheck(
                        "embedding_freshness",
                        VerifyStatus.ERROR,
                        summary=f"Embedding freshness check failed: {exc}",
                    )
                )

    for source in config.sources:
        if source.folder:
            cred_path = default_credentials_path(config.drive_config)
            token_path = default_token_path(config.drive_config)
            cred_status = VerifyStatus.OK if cred_path.exists() else VerifyStatus.WARNING
            token_status = VerifyStatus.OK if token_path.exists() else VerifyStatus.WARNING
            checks.append(
                HealthCheck(
                    f"source:{source.name}",
                    cred_status,
                    summary=f"drive folder '{source.folder}' credentials: {cred_path}",
                )
            )
            checks.append(
                HealthCheck(
                    f"source:{source.name}:token",
                    token_status,
                    summary=f"drive token: {token_path}",
                )
            )
        else:
            if source.path and source.path.exists():
                checks.append(
                    HealthCheck(f"source:{source.name}", VerifyStatus.OK, summary=str(source.path))
                )
            else:
                checks.append(
                    HealthCheck(
                        f"source:{source.name}",
                        VerifyStatus.WARNING,
                        summary=f"missing path: {source.path}",
                    )
                )

    try:
        from .schemas.registry import SchemaRegistry

        registry = SchemaRegistry()
        known_providers = list(CORE_SCHEMA_PROVIDERS)
        available = registry.list_providers()
        missing = [provider for provider in known_providers if provider not in available]

        if missing:
            checks.append(
                HealthCheck(
                    "schemas_missing",
                    VerifyStatus.WARNING,
                    count=len(missing),
                    summary=f"Missing schemas for: {', '.join(missing)}",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    "schemas_coverage",
                    VerifyStatus.OK,
                    count=len(available),
                    summary=f"All {len(available)} provider schemas present",
                )
            )

        stale_providers = []
        for provider in available:
            age = registry.get_schema_age_days(provider)
            if age is not None and age > 30:
                stale_providers.append(f"{provider} ({age}d)")
        if stale_providers:
            checks.append(
                HealthCheck(
                    "schemas_freshness",
                    VerifyStatus.WARNING,
                    count=len(stale_providers),
                    summary=f"Stale schemas (>30d): {', '.join(stale_providers)}",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    "schemas_freshness",
                    VerifyStatus.OK,
                    summary="All schemas current",
                )
            )
    except Exception as exc:
        checks.append(
            HealthCheck("schemas", VerifyStatus.WARNING, summary=f"Schema check failed: {exc}")
        )

    report = HealthReport(checks=checks)
    write_cache(config.archive_root, report)
    return report


def get_health(config: Config, *, deep: bool = False) -> HealthReport:
    """Get an archive health report, using cache when valid."""
    if not deep:
        cached_report = load_cached_report(config.archive_root)
        if cached_report is not None:
            return cached_report

    report = run_archive_health(config, deep=deep)
    report.cached = False
    report.age_seconds = 0
    return report
