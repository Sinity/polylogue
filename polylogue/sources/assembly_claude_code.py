"""Claude Code provider assembly — sessions-index.json + history.jsonl sidecars."""

from __future__ import annotations

from pathlib import Path

from polylogue.logging import get_logger

from .assembly import (
    ClaudeCodeHistoryPasteIndex,
    ClaudeCodeSessionIndex,
    SidecarData,
)
from .parsers.base import ParsedMessage, ParsedSession
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


class ClaudeCodeAssemblySpec:
    """Claude Code provider assembly — sessions-index.json + history.jsonl."""

    def discover_sidecars(self, source_paths: list[Path]) -> SidecarData:
        """Discover Claude Code sidecars.

        Returns both the session index (used for title/branch enrichment) and
        the per-session history paste index (used for ``has_paste`` evidence,
        #1583).
        """
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
        return {"session_index": session_index, "history_paste_index": merged_history}

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
    user_messages: list[tuple[int, ParsedMessage]] = []
    for idx, msg in enumerate(conv.messages):
        if msg.role != "user":
            continue
        ts_ms = _message_timestamp_ms(msg)
        if ts_ms is None:
            continue
        user_messages.append((idx, msg))
    if not user_messages:
        return conv

    marked_indices: set[int] = set()
    for entry in paste_entries:
        if entry.timestamp_ms is None:
            continue
        candidates: list[int] = []
        for idx, msg in user_messages:
            msg_ts = _message_timestamp_ms(msg)
            if msg_ts is None:
                continue
            if abs(msg_ts - entry.timestamp_ms) <= _HISTORY_TIMESTAMP_TOLERANCE_MS:
                candidates.append(idx)
        if len(candidates) != 1:
            # #1656: surface unmatched/ambiguous history rows as structured
            # diagnostics so operators can audit paste-evidence coverage.
            if not candidates:
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
                    candidate_count=len(candidates),
                )
            continue
        marked_indices.add(candidates[0])
    if not marked_indices:
        return conv

    new_messages: list[ParsedMessage] = []
    for idx, msg in enumerate(conv.messages):
        if idx in marked_indices:
            meta = dict(msg.provider_meta or {})
            meta["claude_code_history_paste"] = True
            new_messages.append(msg.model_copy(update={"provider_meta": meta}))
        else:
            new_messages.append(msg)
    return conv.model_copy(update={"messages": new_messages})


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
