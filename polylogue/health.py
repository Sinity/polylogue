"""Health checks, verification, and repair operations."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

from .config import Config
from .lib.log import get_logger
from .sources.drive_client import default_credentials_path, default_token_path
from .storage.backends.connection import connection_context, open_connection
from .storage.index import index_status

LOGGER = get_logger(__name__)
HEALTH_TTL_SECONDS = 600


class VerifyStatus(str, Enum):
    """Status levels for health and verification checks."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


@dataclass
class HealthCheck:
    """Result of a single health or verification check."""

    name: str
    status: VerifyStatus
    count: int = 0
    detail: str = ""
    breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "count": self.count,
            "detail": self.detail,
            "breakdown": self.breakdown,
        }


@dataclass
class HealthReport:
    """Comprehensive health and verification report."""

    checks: list[HealthCheck]
    summary: dict[str, int]
    timestamp: int = field(default_factory=lambda: int(time.time()))
    cached: bool = False
    age_seconds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cached": self.cached,
            "age_seconds": self.age_seconds,
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
        }


def _cache_path(archive_root: Path) -> Path:
    return archive_root / "health.json"


def _load_cached(archive_root: Path) -> dict[str, Any] | None:
    path = _cache_path(archive_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        LOGGER.warning("Failed to load health cache: %s", exc)
    return None


def _write_cache(archive_root: Path, report: HealthReport | dict[str, Any]) -> None:
    path = _cache_path(archive_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict() if isinstance(report, HealthReport) else report
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("Failed to write health cache: %s", exc)


def run_health(config: Config, *, deep: bool = False) -> HealthReport:
    """Run comprehensive system health and data verification checks.

    Args:
        config: Application configuration
        deep: If True, run PRAGMA integrity_check (slow on large databases).
              Skipped by default since it reads every page in the DB file.
    """
    checks: list[HealthCheck] = []

    # 1. Environment & Paths
    checks.append(HealthCheck("config", VerifyStatus.OK, detail="Zero-config (XDG paths)"))

    for path_name in ("archive_root", "render_root"):
        path = getattr(config, path_name)
        if path.exists():
            checks.append(HealthCheck(path_name, VerifyStatus.OK, detail=str(path)))
        else:
            checks.append(HealthCheck(path_name, VerifyStatus.WARNING, detail=f"Missing {path}"))

    # 2. Database Reachability (& optional Integrity)
    try:
        with open_connection(None) as conn:
            checks.append(HealthCheck("database", VerifyStatus.OK, detail="DB reachable"))
            if deep:
                integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                checks.append(
                    HealthCheck(
                        "sqlite_integrity",
                        VerifyStatus.OK if integrity == "ok" else VerifyStatus.ERROR,
                        detail=integrity,
                    )
                )
    except Exception as exc:
        checks.append(HealthCheck("database", VerifyStatus.ERROR, detail=f"DB error: {exc}"))

    # 3. Search Index Status
    idx = index_status()
    if idx["exists"]:
        checks.append(
            HealthCheck(
                "index", VerifyStatus.OK, count=int(cast(Any, idx["count"])), detail=f"messages indexed: {idx['count']}"
            )
        )
    else:
        checks.append(HealthCheck("index", VerifyStatus.WARNING, detail="index not built"))

    # 4. Data Quality (from former verify.py)
    with connection_context(None) as conn:
        # Orphaned messages — two-step approach for performance.
        # Step 1: find distinct orphan conversation_ids (fast via index on small conversations table).
        # Step 2: count messages only for those IDs (skipped entirely when no orphans exist).
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
                detail="No orphaned messages" if orphan_count == 0 else f"{orphan_count:,} orphaned messages",
            )
        )

        # Duplicate conversations
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
                detail="No duplicates" if dup_conv == 0 else f"{dup_conv} duplicate conversation IDs",
            )
        )

        # Empty conversations (conversations with no messages)
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
                detail="No empty conversations" if empty_conv == 0 else f"{empty_conv} conversation(s) with no messages",
            )
        )

        # FTS sync check (verify messages_fts table exists and is in sync)
        try:
            # Check if messages_fts table exists
            fts_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()

            if fts_exists:
                # Count messages in both tables
                # Note: COUNT(*) on the FTS virtual table is extremely slow (15s+ on large DBs).
                # Use the backing docsize table instead, which has one row per indexed document.
                msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]

                if msg_count == fts_count:
                    checks.append(
                        HealthCheck(
                            "fts_sync",
                            VerifyStatus.OK,
                            count=fts_count,
                            detail=f"FTS index in sync ({fts_count:,} messages indexed)",
                        )
                    )
                else:
                    checks.append(
                        HealthCheck(
                            "fts_sync",
                            VerifyStatus.WARNING,
                            count=abs(msg_count - fts_count),
                            detail=f"FTS out of sync: {msg_count:,} messages vs {fts_count:,} indexed",
                        )
                    )
            else:
                # FTS table doesn't exist
                msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                checks.append(
                    HealthCheck(
                        "fts_sync",
                        VerifyStatus.WARNING,
                        count=msg_count,
                        detail=f"FTS index not built ({msg_count:,} messages not indexed)",
                    )
                )
        except Exception as exc:
            checks.append(
                HealthCheck(
                    "fts_sync",
                    VerifyStatus.ERROR,
                    detail=f"FTS check failed: {exc}",
                )
            )

    # 5. Source Accessibility
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
                    detail=f"drive folder '{source.folder}' credentials: {cred_path}",
                )
            )
            checks.append(
                HealthCheck(
                    f"source:{source.name}:token",
                    token_status,
                    detail=f"drive token: {token_path}",
                )
            )
        else:
            if source.path and source.path.exists():
                checks.append(HealthCheck(f"source:{source.name}", VerifyStatus.OK, detail=str(source.path)))
            else:
                checks.append(
                    HealthCheck(f"source:{source.name}", VerifyStatus.WARNING, detail=f"missing path: {source.path}")
                )

    # 6. Schema Health
    try:
        from .schemas.registry import SchemaRegistry

        registry = SchemaRegistry()
        known_providers = ["chatgpt", "claude-ai", "claude-code", "codex", "gemini"]
        available = registry.list_providers()
        missing = [p for p in known_providers if p not in available]

        if missing:
            checks.append(
                HealthCheck(
                    "schemas_missing",
                    VerifyStatus.WARNING,
                    count=len(missing),
                    detail=f"Missing schemas for: {', '.join(missing)}",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    "schemas_coverage",
                    VerifyStatus.OK,
                    count=len(available),
                    detail=f"All {len(available)} provider schemas present",
                )
            )

        # Check schema age (warn if >30 days)
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
                    detail=f"Stale schemas (>30d): {', '.join(stale_providers)}",
                )
            )
        else:
            checks.append(
                HealthCheck("schemas_freshness", VerifyStatus.OK, detail="All schemas current")
            )
    except Exception as exc:
        checks.append(
            HealthCheck("schemas", VerifyStatus.WARNING, detail=f"Schema check failed: {exc}")
        )

    # Build summary
    summary = {"ok": 0, "warning": 0, "error": 0}
    for check in checks:
        summary[check.status.value] += 1

    report = HealthReport(checks=checks, summary=summary)
    _write_cache(config.archive_root, report)
    return report


def get_health(config: Config, *, deep: bool = False) -> HealthReport:
    """Get health report, using cache if valid.

    Args:
        config: Application configuration
        deep: If True, skip cache and run PRAGMA integrity_check (slow).
    """
    if not deep:
        cached_data = _load_cached(config.archive_root)
        now = int(time.time())
        if cached_data:
            ts = cached_data.get("timestamp", 0)
            if (now - ts) < HEALTH_TTL_SECONDS:
                try:
                    checks = [
                        HealthCheck(
                            name=c["name"],
                            status=VerifyStatus(c["status"]),
                            count=c.get("count", 0),
                            detail=c.get("detail", ""),
                            breakdown=c.get("breakdown", {}),
                        )
                        for c in cached_data.get("checks", [])
                    ]
                except (KeyError, ValueError):
                    # Corrupted cache — fall through to fresh health check
                    pass
                else:
                    return HealthReport(
                        checks=checks,
                        summary=cached_data.get("summary", {}),
                        timestamp=ts,
                        cached=True,
                        age_seconds=now - ts,
                    )

    report = run_health(config, deep=deep)
    report.cached = False
    report.age_seconds = 0
    return report


def cached_health_summary(archive_root: Path) -> str:
    """Get a concise summary of the cached health state."""
    cached_data = _load_cached(archive_root)
    if not cached_data:
        return "not run"
    try:
        ts = int(cached_data.get("timestamp", 0))
    except (TypeError, ValueError):
        return "unknown"
    age = int(time.time()) - ts
    summary = cached_data.get("summary", {})
    if not summary:
        return f"cached {age}s ago"

    parts = []
    for s in (VerifyStatus.OK, VerifyStatus.WARNING, VerifyStatus.ERROR):
        if count := summary.get(s.value):
            parts.append(f"{s.value}={count}")
    return f"cached {age}s ago ({', '.join(parts)})"


@dataclass
class RepairResult:
    """Result of a repair operation."""

    name: str
    repaired_count: int
    success: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repaired_count": self.repaired_count,
            "success": self.success,
            "detail": self.detail,
        }


def _run_repair(
    name: str,
    count_sql: str,
    action_sql: str | None,
    dry_run: bool,
    conn: sqlite3.Connection,
) -> RepairResult:
    """Generic repair framework for data cleanup operations.

    Args:
        name: Name of the repair (used in logs and results)
        count_sql: SQL query that returns COUNT(*) to identify affected rows
        action_sql: SQL query to execute the repair (optional for dry-run-only repairs)
        dry_run: If True, count only; if False, execute action_sql
        conn: Database connection

    Returns:
        RepairResult with count and status
    """
    try:
        # Get count of affected rows
        count = conn.execute(count_sql).fetchone()[0]

        if dry_run:
            # Dry-run: just report count
            return RepairResult(
                name=name,
                repaired_count=count,
                success=True,
                detail=f"Would: {count} rows affected" if count else "Would: No issues found",
            )

        # Execute repair
        if action_sql:
            result = conn.execute(action_sql)
            conn.commit()
            return RepairResult(
                name=name,
                repaired_count=result.rowcount,
                success=True,
                detail=f"Repaired {result.rowcount} rows" if result.rowcount else "No repairs needed",
            )

        return RepairResult(
            name=name,
            repaired_count=0,
            success=True,
            detail="No action SQL provided",
        )
    except Exception as exc:
        return RepairResult(
            name=name,
            repaired_count=0,
            success=False,
            detail=f"Repair failed: {exc}",
        )


def repair_orphaned_messages(config: Config, dry_run: bool = False) -> RepairResult:
    """Delete messages that reference non-existent conversations."""
    with connection_context(None) as conn:
        # Two-step count for performance (avoids full table scan on 1.5M+ rows)
        orphan_cids = conn.execute(
            """
            SELECT DISTINCT conversation_id FROM messages
            WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = messages.conversation_id)
            """
        ).fetchall()

        if not orphan_cids:
            return RepairResult(
                name="orphaned_messages",
                repaired_count=0,
                success=True,
                detail="No orphaned messages found",
            )

        placeholders = ",".join("?" for _ in orphan_cids)
        count_sql = f"SELECT COUNT(*) FROM messages WHERE conversation_id IN ({placeholders})"

        # Manually execute the count query for this case
        try:
            count = conn.execute(count_sql, [row[0] for row in orphan_cids]).fetchone()[0]
            if dry_run:
                return RepairResult(
                    name="orphaned_messages",
                    repaired_count=count,
                    success=True,
                    detail=f"Would: Delete {count} orphaned messages" if count else "Would: No orphaned messages found",
                )

            result = conn.execute(
                f"DELETE FROM messages WHERE conversation_id IN ({placeholders})",
                [row[0] for row in orphan_cids],
            )
            conn.commit()
            return RepairResult(
                name="orphaned_messages",
                repaired_count=result.rowcount,
                success=True,
                detail=f"Deleted {result.rowcount} orphaned messages" if result.rowcount else "No orphaned messages found",
            )
        except Exception as exc:
            return RepairResult(
                name="orphaned_messages",
                repaired_count=0,
                success=False,
                detail=f"Failed to delete orphaned messages: {exc}",
            )


def repair_empty_conversations(config: Config, dry_run: bool = False) -> RepairResult:
    """Delete conversations that have no messages."""
    with connection_context(None) as conn:
        return _run_repair(
            name="empty_conversations",
            count_sql="SELECT COUNT(*) FROM conversations c WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.conversation_id)",
            action_sql="DELETE FROM conversations WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = conversations.conversation_id)",
            dry_run=dry_run,
            conn=conn,
        )


def repair_dangling_fts(config: Config, dry_run: bool = False) -> RepairResult:
    """Rebuild FTS index entries that are out of sync with messages table."""
    try:
        with connection_context(None) as conn:
            # Check if FTS table exists
            fts_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()

            if not fts_exists:
                return RepairResult(
                    name="dangling_fts",
                    repaired_count=0,
                    success=True,
                    detail="FTS table does not exist, skipping",
                )

            if dry_run:
                # Fast estimate: compare row counts
                # Use docsize backing table — COUNT(*) on FTS virtual table is 15s+ on large DBs
                msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
                diff = abs(msg_count - fts_count)

                if diff == 0:
                    return RepairResult(
                        name="dangling_fts",
                        repaired_count=0,
                        success=True,
                        detail="FTS index in sync",
                    )

                return RepairResult(
                    name="dangling_fts",
                    repaired_count=diff,
                    success=True,
                    detail=f"Would: FTS sync: {msg_count:,} messages vs {fts_count:,} indexed ({diff:,} difference)",
                )

            # Delete FTS entries that don't have corresponding messages
            result = conn.execute(
                """
                DELETE FROM messages_fts
                WHERE rowid IN (
                    SELECT f.rowid FROM messages_fts f
                    WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.rowid = f.rowid)
                )
                """
            )
            deleted = result.rowcount

            # Insert missing entries into FTS
            inserted = conn.execute(
                """
                INSERT INTO messages_fts (rowid, message_id, conversation_id, content)
                SELECT m.rowid, m.message_id, m.conversation_id, m.text FROM messages m
                WHERE NOT EXISTS (SELECT 1 FROM messages_fts f WHERE f.rowid = m.rowid)
                """
            ).rowcount

            conn.commit()

            total = deleted + inserted
            return RepairResult(
                name="dangling_fts",
                repaired_count=total,
                success=True,
                detail=f"FTS sync: deleted {deleted} orphaned, added {inserted} missing entries",
            )
    except Exception as exc:
        return RepairResult(
            name="dangling_fts",
            repaired_count=0,
            success=False,
            detail=f"Failed to repair FTS index: {exc}",
        )


def repair_orphaned_attachments(config: Config, dry_run: bool = False) -> RepairResult:
    """Delete attachments that are not referenced by any message or have orphaned refs."""
    try:
        with connection_context(None) as conn:
            if dry_run:
                # Count distinct orphaned refs (a single ref can be orphaned on both axes)
                orphaned_refs = conn.execute(
                    """
                    SELECT COUNT(*) FROM attachment_refs ar
                    WHERE (ar.message_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = ar.message_id))
                       OR NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = ar.conversation_id)
                    """
                ).fetchone()[0]

                atts_deleted = conn.execute(
                    """
                    SELECT COUNT(*) FROM attachments a
                    WHERE NOT EXISTS (SELECT 1 FROM attachment_refs ar WHERE ar.attachment_id = a.attachment_id)
                    """
                ).fetchone()[0]

                total = orphaned_refs + atts_deleted
                return RepairResult(
                    name="orphaned_attachments",
                    repaired_count=total,
                    success=True,
                    detail=f"Would: Clean {orphaned_refs} orphaned refs, {atts_deleted} unreferenced attachments",
                )

            # First, delete attachment_refs that point to non-existent messages
            ref_result = conn.execute(
                """
                DELETE FROM attachment_refs
                WHERE message_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = attachment_refs.message_id)
                """
            )
            refs_deleted = ref_result.rowcount

            # Delete attachment_refs that point to non-existent conversations
            conv_ref_result = conn.execute(
                """
                DELETE FROM attachment_refs
                WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = attachment_refs.conversation_id)
                """
            )
            conv_refs_deleted = conv_ref_result.rowcount

            # Delete attachments that have no remaining refs
            att_result = conn.execute(
                """
                DELETE FROM attachments
                WHERE NOT EXISTS (SELECT 1 FROM attachment_refs ar WHERE ar.attachment_id = attachments.attachment_id)
                """
            )
            atts_deleted = att_result.rowcount

            conn.commit()

            total = refs_deleted + conv_refs_deleted + atts_deleted
            return RepairResult(
                name="orphaned_attachments",
                repaired_count=total,
                success=True,
                detail=f"Cleaned {refs_deleted} orphaned refs, {conv_refs_deleted} conv refs, {atts_deleted} attachments",
            )
    except Exception as exc:
        return RepairResult(
            name="orphaned_attachments",
            repaired_count=0,
            success=False,
            detail=f"Failed to clean orphaned attachments: {exc}",
        )


def repair_wal_checkpoint(config: Config, dry_run: bool = False) -> RepairResult:
    """Force WAL checkpoint to resolve busy pages and reclaim WAL space."""
    try:
        if dry_run:
            # All PRAGMA wal_checkpoint modes actually perform a checkpoint.
            # For true dry-run, inspect the WAL file on disk instead.
            from polylogue.storage.backends.connection import default_db_path

            db_path = default_db_path()
            wal_path = Path(str(db_path) + "-wal")
            if wal_path.exists():
                wal_size = wal_path.stat().st_size
                pages_estimate = wal_size // 4096
                return RepairResult(
                    name="wal_checkpoint",
                    repaired_count=pages_estimate,
                    success=True,
                    detail=f"Would: WAL checkpoint (~{pages_estimate} pages, {wal_size:,} bytes)",
                )
            return RepairResult(
                name="wal_checkpoint",
                repaired_count=0,
                success=True,
                detail="Would: No WAL file present, nothing to checkpoint",
            )

        with connection_context(None) as conn:
            result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            row = result.fetchone()
            # wal_checkpoint returns (busy, log, checkpointed)
            busy, log, checkpointed = row[0], row[1], row[2]
            if busy:
                return RepairResult(
                    name="wal_checkpoint",
                    repaired_count=0,
                    success=False,
                    detail=f"WAL checkpoint had busy pages: {busy} busy, {log} log, {checkpointed} checkpointed",
                )
            return RepairResult(
                name="wal_checkpoint",
                repaired_count=checkpointed if checkpointed > 0 else 0,
                success=True,
                detail=f"WAL checkpoint complete: {checkpointed} pages checkpointed",
            )
    except Exception as exc:
        return RepairResult(
            name="wal_checkpoint",
            repaired_count=0,
            success=False,
            detail=f"WAL checkpoint failed: {exc}",
        )


def repair_unknown_roles(config: Config, dry_run: bool = False) -> RepairResult:
    """Reclassify 'unknown' role messages for claude-code conversations.

    Claude-code sessions had record types (progress, file-history-snapshot, etc.)
    stored as role='unknown' before the parser was fixed. Uses json_extract on
    provider_meta to map: progress/result → 'tool', system/summary/snapshot → 'system'.
    """
    try:
        with connection_context(None) as conn:
            count = conn.execute(
                """SELECT COUNT(*) FROM messages m
                   JOIN conversations c ON c.conversation_id = m.conversation_id
                   WHERE m.role = 'unknown' AND c.provider_name = 'claude-code'"""
            ).fetchone()[0]

            if count == 0:
                return RepairResult(
                    name="unknown_roles",
                    repaired_count=0,
                    success=True,
                    detail="No unknown-role messages found in claude-code conversations",
                )

            if dry_run:
                return RepairResult(
                    name="unknown_roles",
                    repaired_count=count,
                    success=True,
                    detail=f"Would: Reclassify {count:,} unknown → tool/system in claude-code conversations",
                )

            # Use json_extract to map record type → correct role
            result = conn.execute(
                """UPDATE messages SET role = CASE
                       WHEN json_extract(provider_meta, '$.raw.type')
                           IN ('summary', 'system', 'file-history-snapshot', 'queue-operation')
                           THEN 'system'
                       ELSE 'tool'
                   END
                   WHERE role = 'unknown'
                   AND conversation_id IN (
                       SELECT conversation_id FROM conversations WHERE provider_name = 'claude-code'
                   )"""
            )
            conn.commit()
            return RepairResult(
                name="unknown_roles",
                repaired_count=result.rowcount,
                success=True,
                detail=f"Reclassified {result.rowcount:,} unknown → tool/system in claude-code conversations",
            )
    except Exception as exc:
        return RepairResult(
            name="unknown_roles",
            repaired_count=0,
            success=False,
            detail=f"Failed to repair unknown roles: {exc}",
        )


def run_all_repairs(config: Config, dry_run: bool = False) -> list[RepairResult]:
    """Run all repair operations and return results.

    Args:
        config: Configuration object
        dry_run: If True, show what would be repaired without making changes
    """
    return [
        repair_orphaned_messages(config, dry_run=dry_run),
        repair_empty_conversations(config, dry_run=dry_run),
        repair_dangling_fts(config, dry_run=dry_run),
        repair_orphaned_attachments(config, dry_run=dry_run),
        repair_unknown_roles(config, dry_run=dry_run),
        repair_wal_checkpoint(config, dry_run=dry_run),
    ]


__all__ = [
    "get_health",
    "run_health",
    "HealthCheck",
    "HealthReport",
    "VerifyStatus",
    "cached_health_summary",
    "RepairResult",
    "repair_orphaned_messages",
    "repair_empty_conversations",
    "repair_dangling_fts",
    "repair_orphaned_attachments",
    "repair_wal_checkpoint",
    "run_all_repairs",
]
