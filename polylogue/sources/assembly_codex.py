"""Codex provider assembly — session_index.jsonl and history.jsonl sidecars."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path

from polylogue.core.enums import MaterialOrigin, TitleSource
from polylogue.core.json import json_document
from polylogue.logging import get_logger

from .assembly import CodexHistoryTitles, CodexThreadNames, SidecarData
from .parsers.base import ParsedSession

logger = get_logger(__name__)

_TITLE_PREVIEW_LIMIT = 80

# Per-process sidecar parse cache keyed by absolute path, validated by
# (mtime_ns, size). Canonical raw-record ingest discovers sidecars once per
# raw record; without this cache a catch-up batch re-reads the same
# append-only sidecar files thousands of times.
_SIDECAR_CACHE: dict[str, tuple[tuple[int, int], dict[str, str]]] = {}


def _cached_parse(path: Path, parse: Callable[[Path], dict[str, str]]) -> dict[str, str]:
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = str(path)
    fingerprint = (stat.st_mtime_ns, stat.st_size)
    cached = _SIDECAR_CACHE.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    data = parse(path)
    _SIDECAR_CACHE[key] = (fingerprint, data)
    return data


def _parse_codex_session_index(sessions_root: Path) -> dict[str, str]:
    """Parse ``session_index.jsonl`` — append-only, newest entry wins per thread id.

    Args:
        sessions_root: The ``sessions/`` directory. The index file lives at
            ``sessions_root.parent / "session_index.jsonl"``.

    Returns:
        Mapping of thread ID to thread name (latest entry wins).
    """
    index_path = sessions_root.parent / "session_index.jsonl"
    if not index_path.exists():
        return {}
    return _cached_parse(index_path, _parse_session_index_file)


def _parse_session_index_file(index_path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    try:
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    continue
                entry = json_document(parsed)
                tid = _coerce_codex_session_id(entry)
                name = _coerce_codex_thread_name(entry)
                if tid and name:
                    names[tid] = name  # Latest wins (append-only)
            except (json.JSONDecodeError, TypeError):
                continue
    except OSError as exc:
        logger.debug("Failed to read Codex session_index.jsonl: %s", exc)
    return names


def _parse_codex_history(sessions_root: Path) -> dict[str, str]:
    """Parse ``history.jsonl`` — the earliest authored entry per session wins.

    Live Codex appends ``{"session_id": ..., "ts": ..., "text": ...}`` rows
    for every operator-typed prompt. The earliest entry for a session is its
    opening request, which is the authoritative authored title material.
    Ties on ``ts`` keep the first-seen row so repeated parses of an
    append-only file stay deterministic.
    """
    history_path = sessions_root.parent / "history.jsonl"
    if not history_path.exists():
        return {}
    return _cached_parse(history_path, _parse_history_file)


def _parse_history_file(history_path: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    earliest_ts: dict[str, float] = {}
    try:
        for line in history_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            sid = parsed.get("session_id")
            text = parsed.get("text")
            ts = parsed.get("ts")
            if not (isinstance(sid, str) and sid and isinstance(text, str) and text.strip()):
                continue
            ts_value = float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) else float("inf")
            known = earliest_ts.get(sid)
            if known is None or ts_value < known:
                earliest_ts[sid] = ts_value
                titles[sid] = text
    except OSError as exc:
        logger.debug("Failed to read Codex history.jsonl: %s", exc)
    return titles


def _parse_codex_state_titles(sessions_root: Path) -> dict[str, str]:
    """Read ``threads.title`` from the live Codex ``state_5.sqlite`` store.

    On modern Codex installs ``state_5.sqlite`` (a live SQLite database in
    WAL mode) is a richer, more actively maintained successor of
    ``session_index.jsonl`` — it can carry a curated ``threads.title`` even
    when ``session_index.jsonl`` no longer exists. Codex itself may hold this
    file open for writing at any time, so treat it as an unreliable, optional
    evidence source: any failure (missing file, lock contention, schema
    drift, corruption) degrades to an empty mapping rather than raising.
    """
    state_path = sessions_root.parent / "state_5.sqlite"
    if not state_path.exists():
        return {}
    return _cached_parse(state_path, _parse_state_db_file)


def _parse_state_db_file(state_path: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    try:
        # mode=ro avoids ever taking a write lock; WAL mode lets us read
        # concurrently with a live Codex writer. A short timeout keeps a
        # momentarily-locked file from stalling ingest.
        conn = sqlite3.connect(f"file:{state_path}?mode=ro", uri=True, timeout=1.0)
    except sqlite3.Error as exc:
        logger.debug("Failed to open Codex state_5.sqlite: %s", exc)
        return {}
    try:
        try:
            rows = conn.execute("SELECT id, title FROM threads").fetchall()
        except sqlite3.Error as exc:
            logger.debug("Failed to query Codex state_5.sqlite threads: %s", exc)
            return {}
        for thread_id, title in rows:
            if isinstance(thread_id, str) and thread_id and isinstance(title, str) and title.strip():
                titles[thread_id] = title.strip()
    finally:
        conn.close()
    return titles


def _title_preview(text: str) -> str | None:
    """First non-empty line, bounded, or None when nothing usable remains."""
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            if len(line) > _TITLE_PREVIEW_LIMIT:
                return line[:_TITLE_PREVIEW_LIMIT] + "..."
            return line
    return None


class CodexAssemblySpec:
    """Codex provider assembly — thread-name and authored-history sidecars."""

    def discover_sidecars(self, source_paths: list[Path]) -> SidecarData:
        """Discover Codex thread names and authored-history titles.

        Returns ``{"thread_names": {...}, "history_titles": {...}}``.
        """
        thread_names: dict[str, str] = {}
        history_titles: dict[str, str] = {}
        state_titles: dict[str, str] = {}
        seen_roots: set[Path] = set()
        for path in source_paths:
            # Walk up to find the sessions root
            for parent in path.parents:
                if parent.name == "sessions" and parent not in seen_roots:
                    seen_roots.add(parent)
                    thread_names.update(_parse_codex_session_index(parent))
                    history_titles.update(_parse_codex_history(parent))
                    state_titles.update(_parse_codex_state_titles(parent))
                    break
        return {
            "thread_names": thread_names,
            "history_titles": history_titles,
            "state_titles": state_titles,
        }

    def enrich_session(
        self,
        conv: ParsedSession,
        sidecar_data: SidecarData,
    ) -> ParsedSession:
        """Resolve a Codex title: thread name → authored history →
        state_5.sqlite thread title → first human-authored message → leave
        the native id.

        A ``role=user`` row alone never becomes a title: Codex runtime
        context and operator protocol rows share that role, so only
        ``material_origin == HUMAN_AUTHORED`` text qualifies for the message
        fallback.
        """
        thread_names: CodexThreadNames = sidecar_data.get("thread_names", {})
        history_titles: CodexHistoryTitles = sidecar_data.get("history_titles", {})
        state_titles: CodexHistoryTitles = sidecar_data.get("state_titles", {})
        cid = conv.provider_session_id

        # 1. Provider thread name — authoritative, may replace a stale title.
        name = thread_names.get(cid)
        if name:
            if name != conv.title:
                return conv.model_copy(
                    update={
                        "title": name,
                        "title_source": TitleSource.ORIGIN,
                    }
                )
            return conv

        # The remaining lanes only fill in when the title is missing or is
        # the bare native id — they never replace a real title.
        if conv.title and conv.title != cid:
            return conv

        # 2. Authored history entry recorded by Codex for this session.
        history_text = history_titles.get(cid)
        if history_text:
            preview = _title_preview(history_text)
            if preview:
                return conv.model_copy(
                    update={
                        "title": preview,
                        "title_source": TitleSource.ORIGIN,
                    }
                )

        # 3. state_5.sqlite live thread title — a richer, more actively
        # maintained sibling of session_index.jsonl (which some modern
        # Codex installs no longer even write). Only fills the gap left by
        # thread name / authored history above; it never overrides an
        # authored-history title already resolved in step 2.
        state_text = state_titles.get(cid)
        if state_text:
            preview = _title_preview(state_text)
            if preview:
                return conv.model_copy(
                    update={
                        "title": preview,
                        "title_source": TitleSource.ORIGIN,
                    }
                )

        # 4. First human-authored message in the parsed session.
        for msg in conv.messages:
            if msg.material_origin is not MaterialOrigin.HUMAN_AUTHORED:
                continue
            if not (msg.text and msg.text.strip()):
                continue
            preview = _title_preview(msg.text)
            if preview:
                return conv.model_copy(
                    update={
                        "title": preview,
                        "title_source": TitleSource.HEURISTIC,
                    }
                )

        # 5. Nothing enrichable — the native id stands.
        return conv


def _coerce_codex_session_id(entry: Mapping[str, object]) -> str | None:
    """Read a thread identifier from a parsed Codex session-index entry."""
    value = entry.get("id") or entry.get("thread_id")
    return value if isinstance(value, str) and value else None


def _coerce_codex_thread_name(entry: Mapping[str, object]) -> str | None:
    """Read a thread name from a parsed Codex session-index entry."""
    value = entry.get("thread_name") or entry.get("name")
    return value if isinstance(value, str) and value else None


__all__ = [
    "CodexAssemblySpec",
    "_parse_codex_history",
    "_parse_codex_session_index",
    "_parse_codex_state_titles",
]
