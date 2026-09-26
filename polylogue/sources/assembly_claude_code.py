"""Claude Code provider assembly — sessions-index.json + history.jsonl sidecars."""

from __future__ import annotations

import sqlite3
from collections.abc import MutableSequence
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from polylogue.core.enums import PasteBoundary
from polylogue.logging import get_logger
from polylogue.storage.blob_store import BlobStore

from .assembly import (
    ClaudeCodeHistoryPasteIndex,
    ClaudeCodeSessionIndex,
    SidecarData,
)
from .parsers.base import ParsedMessage, ParsedPasteEvidence, ParsedSession
from .parsers.claude.history import HistoryEntry, build_session_paste_index
from .parsers.claude.index import (
    SessionIndexEntry,
    enrich_session_from_index,
    parse_sessions_index,
)

logger = get_logger(__name__)

# `~/.claude/history.jsonl` is global to a Claude Code install; rooted with
# ``~/.claude/projects/...`` on the source-walk side, the history sidecar
# sits at ``~/.claude/history.jsonl`` — two levels up.
_HISTORY_RELATIVE = Path("..") / ".." / "history.jsonl"

# Strong-identity match window: history timestamp must fall within this many
# milliseconds of the archived user message timestamp. Six seconds covers
# realistic clock-skew + prompt-buffering jitter without crossing into the
# next user message.
_HISTORY_TIMESTAMP_TOLERANCE_MS = 6_000
_SQLITE_INT_MIN = -(2**63)
_SQLITE_INT_MAX = 2**63 - 1


class ClaudeCodeAssemblySpec:
    """Claude Code provider assembly — sessions-index.json + history.jsonl."""

    def discover_sidecars(
        self,
        source_paths: list[Path],
        *,
        blob_store: BlobStore | None = None,
    ) -> SidecarData:
        """Discover Claude Code sidecars.

        Returns both the session index (used for title/branch enrichment) and
        the per-session history paste index (used for ``has_paste`` evidence,
        #1583).
        """
        del blob_store
        indices: dict[Path, dict[str, SessionIndexEntry]] = {}
        history_indices: dict[Path, ClaudeCodeHistoryPasteIndex] = {}
        for path in source_paths:
            parent = path.parent
            if parent not in indices:
                index_path = parent / "sessions-index.json"
                indices[parent] = parse_sessions_index(index_path)
            if parent not in history_indices:
                history_path = (parent / _HISTORY_RELATIVE).resolve()
                history_indices[parent] = build_session_paste_index(history_path)
        session_index: ClaudeCodeSessionIndex = {}
        for entries in indices.values():
            session_index.update(entries)
        merged_history: ClaudeCodeHistoryPasteIndex = {}
        for hist in history_indices.values():
            for session_id, history_entries in hist.items():
                merged_history.setdefault(session_id, []).extend(history_entries)
        return {
            "session_index": session_index,
            "history_paste_index": merged_history,
        }

    def enrich_session(
        self,
        conv: ParsedSession,
        sidecar_data: SidecarData,
    ) -> ParsedSession:
        """Enrich a Claude Code session from session-index + history sidecars."""
        idx: ClaudeCodeSessionIndex = sidecar_data.get("session_index", {})
        if conv.provider_session_id in idx:
            conv = enrich_session_from_index(conv, idx[conv.provider_session_id])
        history_index: ClaudeCodeHistoryPasteIndex = sidecar_data.get("history_paste_index", {})
        paste_entries = history_index.get(conv.provider_session_id, [])
        if paste_entries:
            conv = _annotate_messages_with_history_paste(conv, paste_entries)
        return conv


def _annotate_messages_with_history_paste(
    conv: ParsedSession,
    paste_entries: list[HistoryEntry],
) -> ParsedSession:
    """Mark user messages whose timestamps match a paste-bearing history row.

    Operates only on the strong-identity path: sessionId already pinned by
    ``build_session_paste_index``; here we match each history row to one user
    message within ``_HISTORY_TIMESTAMP_TOLERANCE_MS``. An ambiguous history
    row that matches more than one candidate user message is dropped from
    the strong-identity path rather than silently fanning paste evidence
    across unrelated messages (#1583 ACs explicitly forbid silent
    misattribution).
    """
    if not paste_entries:
        return conv
    scratch_root = Path("/realm/tmp/work")
    with (
        TemporaryDirectory(
            prefix="polylogue-history-paste-", dir=scratch_root if scratch_root.is_dir() else None
        ) as scratch_dir,
        closing(sqlite3.connect(Path(scratch_dir) / "messages.sqlite")) as db,
    ):
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA cache_size=-2048")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("CREATE TABLE user_message (position INTEGER PRIMARY KEY, timestamp_ms INTEGER NOT NULL)")
        db.execute("CREATE INDEX user_message_timestamp ON user_message (timestamp_ms)")
        db.execute("CREATE TABLE big_user_message (position INTEGER PRIMARY KEY, timestamp_ms TEXT NOT NULL)")
        db.execute("CREATE TABLE marked (position INTEGER PRIMARY KEY, entry_index INTEGER NOT NULL)")

        for position, message in enumerate(conv.messages):
            if message.role != "user":
                continue
            timestamp_ms = _message_timestamp_ms(message)
            if timestamp_ms is None:
                continue
            if _SQLITE_INT_MIN <= timestamp_ms <= _SQLITE_INT_MAX:
                db.execute("INSERT INTO user_message VALUES (?, ?)", (position, timestamp_ms))
            else:
                db.execute("INSERT INTO big_user_message VALUES (?, ?)", (position, str(timestamp_ms)))

        if (
            db.execute("SELECT 1 FROM user_message LIMIT 1").fetchone() is None
            and db.execute("SELECT 1 FROM big_user_message LIMIT 1").fetchone() is None
        ):
            return conv

        for entry_index, entry in enumerate(paste_entries):
            if entry.timestamp_ms is None:
                continue
            lower = entry.timestamp_ms - _HISTORY_TIMESTAMP_TOLERANCE_MS
            upper = entry.timestamp_ms + _HISTORY_TIMESTAMP_TOLERANCE_MS
            count: int
            candidate_position: int | None
            if lower <= _SQLITE_INT_MAX and upper >= _SQLITE_INT_MIN:
                row = db.execute(
                    "SELECT COUNT(*), MIN(position) FROM user_message WHERE timestamp_ms BETWEEN ? AND ?",
                    (
                        max(_SQLITE_INT_MIN, lower),
                        min(_SQLITE_INT_MAX, upper),
                    ),
                ).fetchone()
                assert row is not None
                count = int(row[0])
                candidate_position = int(row[1]) if row[1] is not None else None
            else:
                count, candidate_position = 0, None
            for big_position, big_timestamp in db.execute("SELECT position, timestamp_ms FROM big_user_message"):
                if lower <= int(big_timestamp) <= upper:
                    count += 1
                    candidate_position = (
                        big_position if candidate_position is None else min(candidate_position, big_position)
                    )
            if count != 1:
                # #1656: retain diagnostics for unmatched and ambiguous rows.
                if not count:
                    logger.info(
                        "history.paste.unmatched",
                        session_id=entry.session_id,
                        timestamp_ms=entry.timestamp_ms,
                        paste_count=len(entry.pastes),
                    )
                else:
                    logger.info(
                        "history.paste.ambiguous",
                        session_id=entry.session_id,
                        timestamp_ms=entry.timestamp_ms,
                        candidate_count=count,
                    )
                continue
            db.execute("INSERT OR REPLACE INTO marked VALUES (?, ?)", (candidate_position, entry_index))

        if db.execute("SELECT 1 FROM marked LIMIT 1").fetchone() is None:
            return conv

        message_sequence = cast(MutableSequence[ParsedMessage], conv.messages)
        messages = list(message_sequence) if isinstance(message_sequence, list) else message_sequence
        for position, entry_index in db.execute("SELECT position, entry_index FROM marked ORDER BY position"):
            message = messages[position]
            messages[position] = message.model_copy(
                update={"paste_spans": _history_paste_spans(message, paste_entries[entry_index])}
            )
        return conv.model_copy(update={"messages": messages})


def _history_paste_spans(msg: ParsedMessage, entry: HistoryEntry) -> list[ParsedPasteEvidence]:
    """Append history-derived paste evidence to a message's existing spans.

    A ``history.jsonl`` row records that a paste occurred near this user
    message but carries no in-message offsets, so each paste becomes a
    ``hash_only`` span (the same boundary the live UserPromptSubmit hook uses
    for captured pastes). When the row preserved the pasted text we stamp a
    content hash; otherwise the span records only that a paste happened.
    """
    spans = list(msg.paste_spans)
    base = len(spans)
    for offset, paste in enumerate(entry.pastes):
        content_hash = sha256(paste.content.encode("utf-8")).digest() if paste.has_content else None
        spans.append(
            ParsedPasteEvidence(
                position=base + offset,
                boundary_state=PasteBoundary.HASH_ONLY.value,
                source_marker=paste.paste_id or None,
                content_hash=content_hash,
                observed_at_ms=entry.timestamp_ms,
            )
        )
    return spans


def _message_timestamp_ms(msg: ParsedMessage) -> int | None:
    """Best-effort parse of ``ParsedMessage.timestamp`` to Unix milliseconds."""
    raw = msg.timestamp
    if raw is None:
        return None
    if isinstance(raw, int):
        # Treat as ms if >= year 2001 in ms (10^12), else assume seconds.
        return raw if raw > 10**12 else raw * 1000
    text = str(raw).strip()
    if not text:
        return None
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return None


__all__ = [
    "ClaudeCodeAssemblySpec",
]
