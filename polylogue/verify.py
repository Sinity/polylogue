"""Data quality verification for polylogue conversations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from polylogue.core.json import loads
from polylogue.db import connection_context


class VerifyStatus(str, Enum):
    """Status levels for verification checks."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


@dataclass
class VerifyResult:
    """Result of a single verification check."""
    name: str
    status: VerifyStatus
    count: int = 0
    detail: str = ""
    breakdown: dict[str, int] = field(default_factory=dict)


@dataclass
class VerifyReport:
    """Full verification report."""
    checks: list[VerifyResult]
    summary: dict[str, int]  # {ok: N, warning: N, error: N}

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "count": c.count,
                    "detail": c.detail,
                    "breakdown": c.breakdown,
                }
                for c in self.checks
            ],
            "summary": self.summary,
        }


def verify_data(verbose: bool = False) -> VerifyReport:
    """Run comprehensive data quality verification."""
    checks: list[VerifyResult] = []

    with connection_context(None) as conn:
        # 1. Basic counts
        conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        att_count = conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
        checks.append(VerifyResult(
            name="record_counts",
            status=VerifyStatus.OK,
            count=conv_count + msg_count + att_count,
            detail=f"{conv_count:,} conversations, {msg_count:,} messages, {att_count:,} attachments",
        ))

        # 2. Provider distribution
        provider_rows = conn.execute(
            "SELECT provider_name, COUNT(*) FROM conversations GROUP BY provider_name"
        ).fetchall()
        provider_counts = {row[0]: row[1] for row in provider_rows}
        checks.append(VerifyResult(
            name="provider_distribution",
            status=VerifyStatus.OK,
            count=len(provider_counts),
            detail=", ".join(f"{k}: {v:,}" for k, v in sorted(provider_counts.items(), key=lambda x: -x[1])),
            breakdown=provider_counts,
        ))

        # 3. Orphaned messages (messages referencing non-existent conversations)
        orphan_count = conn.execute("""
            SELECT COUNT(*) FROM messages m
            WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = m.conversation_id)
        """).fetchone()[0]
        checks.append(VerifyResult(
            name="orphaned_messages",
            status=VerifyStatus.OK if orphan_count == 0 else VerifyStatus.ERROR,
            count=orphan_count,
            detail="No orphaned messages" if orphan_count == 0 else f"{orphan_count:,} orphaned messages",
        ))

        # 4. Empty conversations (no messages)
        empty_rows = conn.execute("""
            SELECT c.provider_name, COUNT(*)
            FROM conversations c
            WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.conversation_id)
            GROUP BY c.provider_name
        """).fetchall()
        empty_breakdown = {row[0]: row[1] for row in empty_rows}
        empty_total = sum(empty_breakdown.values())
        # Empty conversations are often stubs/rollouts, so warning not error
        checks.append(VerifyResult(
            name="empty_conversations",
            status=VerifyStatus.OK if empty_total == 0 else VerifyStatus.WARNING,
            count=empty_total,
            detail="No empty conversations" if empty_total == 0 else f"{empty_total:,} empty conversations",
            breakdown=empty_breakdown,
        ))

        # 5. Duplicate conversation IDs
        dup_conv = conn.execute("""
            SELECT conversation_id, COUNT(*) as cnt
            FROM conversations
            GROUP BY conversation_id
            HAVING cnt > 1
        """).fetchall()
        checks.append(VerifyResult(
            name="duplicate_conversations",
            status=VerifyStatus.OK if len(dup_conv) == 0 else VerifyStatus.ERROR,
            count=len(dup_conv),
            detail="No duplicates" if len(dup_conv) == 0 else f"{len(dup_conv)} duplicate conversation IDs",
        ))

        # 6. Duplicate message IDs
        dup_msg = conn.execute("""
            SELECT message_id, COUNT(*) as cnt
            FROM messages
            GROUP BY message_id
            HAVING cnt > 1
        """).fetchall()
        checks.append(VerifyResult(
            name="duplicate_messages",
            status=VerifyStatus.OK if len(dup_msg) == 0 else VerifyStatus.ERROR,
            count=len(dup_msg),
            detail="No duplicates" if len(dup_msg) == 0 else f"{len(dup_msg)} duplicate message IDs",
        ))

        # 7. Messages with empty text (expected for tool calls, etc.)
        empty_text_rows = conn.execute("""
            SELECT c.provider_name, COUNT(*)
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE m.text IS NULL OR m.text = ''
            GROUP BY c.provider_name
        """).fetchall()
        empty_text_breakdown = {row[0]: row[1] for row in empty_text_rows}
        empty_text_total = sum(empty_text_breakdown.values())
        # This is informational, not an error (tool calls often have no text)
        checks.append(VerifyResult(
            name="empty_text_messages",
            status=VerifyStatus.OK,
            count=empty_text_total,
            detail=f"{empty_text_total:,} messages with empty/null text (expected for tool calls)",
            breakdown=empty_text_breakdown,
        ))

        # 8. NULL timestamps by provider
        null_ts_rows = conn.execute("""
            SELECT c.provider_name, COUNT(*)
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE m.timestamp IS NULL
            GROUP BY c.provider_name
        """).fetchall()
        null_ts_breakdown = {row[0]: row[1] for row in null_ts_rows}
        null_ts_total = sum(null_ts_breakdown.values())
        # Some providers (gemini) don't have timestamps in source data
        checks.append(VerifyResult(
            name="null_timestamps",
            status=VerifyStatus.OK if null_ts_total == 0 else VerifyStatus.WARNING,
            count=null_ts_total,
            detail=(
                "All messages have timestamps" if null_ts_total == 0
                else f"{null_ts_total:,} messages with null timestamps"
            ),
            breakdown=null_ts_breakdown,
        ))

        # 9. NULL created_at on conversations
        null_created = conn.execute("""
            SELECT provider_name, COUNT(*)
            FROM conversations
            WHERE created_at IS NULL
            GROUP BY provider_name
        """).fetchall()
        null_created_breakdown = {row[0]: row[1] for row in null_created}
        null_created_total = sum(null_created_breakdown.values())
        checks.append(VerifyResult(
            name="null_created_at",
            status=VerifyStatus.OK if null_created_total == 0 else VerifyStatus.WARNING,
            count=null_created_total,
            detail=(
                "All conversations have created_at" if null_created_total == 0
                else f"{null_created_total:,} conversations with null created_at"
            ),
            breakdown=null_created_breakdown,
        ))

        # 10. Content hash presence
        null_hash = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE content_hash IS NULL"
        ).fetchone()[0]
        checks.append(VerifyResult(
            name="content_hashes",
            status=VerifyStatus.OK if null_hash == 0 else VerifyStatus.WARNING,
            count=msg_count - null_hash,
            detail=(
                f"All {msg_count:,} messages have content hashes" if null_hash == 0
                else f"{null_hash:,} messages missing content hash"
            ),
        ))

        # 11. Provider metadata validity (can we parse it as JSON?)
        invalid_meta = 0
        meta_rows = conn.execute(
            "SELECT provider_meta FROM messages WHERE provider_meta IS NOT NULL LIMIT 10000"
        ).fetchall()
        for row in meta_rows:
            try:
                loads(row[0])
            except (ValueError, TypeError):
                invalid_meta += 1
        checks.append(VerifyResult(
            name="metadata_validity",
            status=VerifyStatus.OK if invalid_meta == 0 else VerifyStatus.WARNING,
            count=len(meta_rows) - invalid_meta,
            detail=(
                "All sampled metadata valid JSON" if invalid_meta == 0
                else f"{invalid_meta} invalid metadata entries (sampled {len(meta_rows)})"
            ),
        ))

        # 12. SQLite integrity check
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        checks.append(VerifyResult(
            name="sqlite_integrity",
            status=VerifyStatus.OK if integrity == "ok" else VerifyStatus.ERROR,
            count=1 if integrity == "ok" else 0,
            detail=integrity,
        ))

        # 13. FTS sync check
        try:
            fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
            # FTS should have one entry per message (using rowid)
            # Check if count matches total messages
            if fts_count == msg_count:
                checks.append(VerifyResult(
                    name="fts_sync",
                    status=VerifyStatus.OK,
                    count=fts_count,
                    detail=f"FTS in sync ({fts_count:,} indexed)",
                ))
            else:
                drift = abs(fts_count - msg_count)
                checks.append(VerifyResult(
                    name="fts_sync",
                    status=VerifyStatus.WARNING,
                    count=fts_count,
                    detail=f"FTS drift: {drift:,} (fts={fts_count:,}, messages={msg_count:,})",
                ))
        except Exception as e:
            checks.append(VerifyResult(
                name="fts_sync",
                status=VerifyStatus.WARNING,
                count=0,
                detail=f"Could not check FTS: {e}",
            ))

    # Build summary
    summary = {"ok": 0, "warning": 0, "error": 0}
    for check in checks:
        summary[check.status.value] = summary.get(check.status.value, 0) + 1

    return VerifyReport(checks=checks, summary=summary)


__all__ = ["verify_data", "VerifyResult", "VerifyReport", "VerifyStatus"]
