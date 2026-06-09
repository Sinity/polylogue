"""Claude Code ``~/.claude/history.jsonl`` paste-evidence sidecar parser.

Claude Code persists every interactive prompt to a global JSONL sidecar at
``~/.claude/history.jsonl``. Each row carries the prompt as ``display`` plus
a ``pastedContents`` map keyed by paste id. Two distinct payload shapes
appear:

- ``pastedContents.<id>.content`` holds the full pasted text. Strong evidence.
- ``pastedContents`` carries only ids/hashes with no ``content``. Weak
  evidence: a paste happened, but the exact text is no longer recoverable
  from this sidecar.

The session-level ``sessionId`` is the only reliable join key back into the
archived session; project + timestamp proximity is the fallback for
older rows that predate sessionId capture (#1583 scope explicitly defers
fuzzy matching to a follow-up — this parser only emits typed rows; matching
strategy is the caller's job).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HistoryPaste:
    """A single paste recorded in ``pastedContents``."""

    paste_id: str
    paste_type: str
    content: str
    has_content: bool

    @property
    def is_hash_only(self) -> bool:
        """True when the row records that a paste existed but not its text."""
        return not self.has_content


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One row of ``~/.claude/history.jsonl``."""

    display: str
    timestamp_ms: int | None
    project: str | None
    session_id: str | None
    pastes: tuple[HistoryPaste, ...] = field(default_factory=tuple)

    @property
    def has_paste(self) -> bool:
        return bool(self.pastes)


def parse_history_jsonl(path: Path) -> Iterator[HistoryEntry]:
    """Yield typed ``HistoryEntry`` rows from a Claude Code history sidecar.

    Skips malformed lines (logging at debug) so a single bad row never aborts
    enrichment for the rest of the file.
    """
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("history.jsonl read failed (%s): %s", path, exc)
        return
    except UnicodeDecodeError as exc:
        logger.debug("history.jsonl decode failed (%s): %s", path, exc)
        return
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.debug("history.jsonl malformed line %d in %s: %s", line_no, path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        yield _entry_from_row(payload)


def _entry_from_row(row: dict[str, object]) -> HistoryEntry:
    display = row.get("display")
    timestamp = row.get("timestamp")
    project = row.get("project")
    session_id = row.get("sessionId")
    pasted = row.get("pastedContents")
    pastes: list[HistoryPaste] = []
    if isinstance(pasted, dict):
        for paste_id, paste_payload in pasted.items():
            if not isinstance(paste_payload, dict):
                continue
            content = paste_payload.get("content")
            has_content = isinstance(content, str) and content != ""
            pastes.append(
                HistoryPaste(
                    paste_id=str(paste_id),
                    paste_type=str(paste_payload.get("type") or "text"),
                    content=content if isinstance(content, str) else "",
                    has_content=has_content,
                )
            )
    return HistoryEntry(
        display=display if isinstance(display, str) else "",
        timestamp_ms=int(timestamp) if isinstance(timestamp, int) and not isinstance(timestamp, bool) else None,
        project=project if isinstance(project, str) else None,
        session_id=session_id if isinstance(session_id, str) and session_id else None,
        pastes=tuple(pastes),
    )


def build_session_paste_index(history_path: Path) -> dict[str, list[HistoryEntry]]:
    """Return ``{sessionId: [entries with paste evidence]}`` for fast lookup.

    Only rows that actually carry paste evidence (``entry.has_paste``) and a
    ``sessionId`` are indexed; the rest are dropped — they cannot be matched
    back to an archived session by the strong-identity path and would
    inflate memory for nothing.
    """
    index: dict[str, list[HistoryEntry]] = {}
    for entry in parse_history_jsonl(history_path):
        if not entry.session_id or not entry.has_paste:
            continue
        index.setdefault(entry.session_id, []).append(entry)
    return index


__all__ = [
    "HistoryEntry",
    "HistoryPaste",
    "build_session_paste_index",
    "parse_history_jsonl",
]
