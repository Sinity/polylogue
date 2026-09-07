"""Resolution helpers for non-session/message user-state targets (#1113).

These helpers validate that a target identified by
``(target_type, target_id, session_id, message_id)`` actually exists
in the archive before a mark or annotation is written, and produce the
canonical ``identity_key`` used by recall packs and workspaces.

The resolver returns the validated ``ResolvedTarget`` mapping or raises
``ValueError`` with a specific, surface-friendly message. Insight kinds
(``session``, ``work_event``, ``thread``) are validated against the
respective insight tables; ``block`` and ``attachment`` are validated
against the archive substrate; ``paste_span`` is treated as an opaque
block-derived identifier and only validated for non-empty ``target_id``.
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TypedDict

from polylogue.core.evidence import Empty, Evidence, Measured, Unavailable, resolve
from polylogue.core.user_state_targets import (
    TARGET_ATTACHMENT,
    TARGET_BLOCK,
    TARGET_PASTE_SPAN,
    TARGET_SESSION,
    TARGET_THREAD,
    TARGET_WORK_EVENT,
    identity_key,
    validate_target_kind,
)


class ResolvedTarget(TypedDict, total=False):
    """Storage row payload + identity key for a resolved user-state target."""

    target_type: str
    target_id: str
    session_id: str
    message_id: str | None
    identity_key: str


_INSIGHT_QUERIES: dict[str, str] = {
    TARGET_SESSION: "SELECT 1 FROM session_profiles WHERE session_id = ?",
    TARGET_WORK_EVENT: "SELECT 1 FROM session_work_events WHERE event_id = ? AND session_id = ?",
    TARGET_THREAD: "SELECT 1 FROM threads WHERE thread_id = ?",
}


def _index_db_path(archive_root: Path) -> Evidence[Path]:
    """Locate the readable `index.db` carrying the canonical ``sessions`` table.

    ``Empty`` means nothing is materialized, which is a real answer about the
    target. ``Unavailable`` means the tier could not be read, which is not --
    a write must refuse rather than report the target absent (polylogue-p707n).
    """
    candidate = archive_root / "index.db"
    if not candidate.exists():
        return Empty()
    try:
        with closing(sqlite3.connect(f"file:{candidate}?mode=ro", uri=True)) as conn:
            row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone()
    except sqlite3.Error as exc:
        return Unavailable(reason="index_tier_unreadable", detail=f"{type(exc).__name__}: {exc}")
    return Measured(candidate) if row is not None else Empty()


def _existence(evidence: Evidence[bool], *, subject: str) -> bool:
    """Consume an existence probe; an unreadable tier refuses, never denies."""

    def _refuse(case: Unavailable) -> bool:
        raise ValueError(f"existence of {subject} could not be checked: {case.detail or case.reason}")

    return resolve(
        evidence,
        measured=lambda value: value,
        empty=lambda: False,
        unavailable=_refuse,
        degraded=lambda case: case.value,
    )


def _row_exists_sync(db_path: Path, sql: str, params: tuple[object, ...]) -> bool:
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        row = conn.execute(sql, params).fetchone()
    return row is not None


async def _row_exists(archive_root: Path, sql: str, params: tuple[object, ...]) -> Evidence[bool]:
    """Existence probe against the `index.db`. A missing index means
    nothing is materialized, so the row does not exist; an unreadable one
    means the question was not answered."""
    located = _index_db_path(archive_root)
    if not isinstance(located, Measured):
        return located if isinstance(located, Unavailable) else Empty()
    try:
        return Measured(await asyncio.to_thread(_row_exists_sync, located.value, sql, params))
    except sqlite3.Error as exc:
        return Unavailable(reason="index_tier_unreadable", detail=f"{type(exc).__name__}: {exc}")


def _block_exists_sync(
    db_path: Path,
    *,
    session_id: str,
    message_id: str,
    block_index: int,
) -> bool:
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM blocks
            WHERE session_id = ?
              AND message_id = ?
              AND position = ?
            """,
            (session_id, message_id, block_index),
        ).fetchone()
    return row is not None


async def _block_exists(
    archive_root: Path,
    *,
    session_id: str,
    message_id: str,
    block_index_token: str,
) -> Evidence[bool]:
    try:
        block_index = int(block_index_token)
    except ValueError:
        return Empty()
    located = _index_db_path(archive_root)
    if not isinstance(located, Measured):
        return located if isinstance(located, Unavailable) else Empty()
    try:
        return Measured(
            await asyncio.to_thread(
                _block_exists_sync,
                located.value,
                session_id=session_id,
                message_id=message_id,
                block_index=block_index,
            )
        )
    except sqlite3.Error as exc:
        return Unavailable(reason="index_tier_unreadable", detail=f"{type(exc).__name__}: {exc}")


def parse_block_target_id(target_id: str) -> tuple[str, str]:
    """Split ``"{message_id}:{block_index}"`` into its components.

    Raises ``ValueError`` if the token is malformed.
    """

    if ":" not in target_id:
        raise ValueError("block target_id must be 'message_id:block_index'")
    message_part, _, block_part = target_id.rpartition(":")
    if not message_part or not block_part:
        raise ValueError("block target_id must be 'message_id:block_index'")
    return message_part, block_part


async def resolve_insight_target(
    archive_root: Path,
    *,
    target_type: str,
    target_id: str | None,
    session_id: str,
    message_id: str | None = None,
) -> ResolvedTarget:
    """Validate a non-session/non-message target and return its row payload.

    The caller is responsible for resolving ``session_id`` first so this
    helper can assume the session exists. ``target_id`` is required for
    every kind except ``session`` (where it defaults to the session_id).
    Existence is checked against the `index.db` under ``archive_root``.
    """

    validate_target_kind(target_type)

    if target_type == TARGET_SESSION:
        resolved_target_id = target_id or session_id
        if resolved_target_id != session_id:
            raise ValueError("session target_id must equal the session_id (session root)")
        if not _existence(
            await _row_exists(archive_root, _INSIGHT_QUERIES[TARGET_SESSION], (session_id,)),
            subject=f"session profile for session {session_id!r}",
        ):
            raise ValueError(f"session profile for session {session_id!r} is not materialized")
        return {
            "target_type": TARGET_SESSION,
            "target_id": session_id,
            "session_id": session_id,
            "message_id": None,
            "identity_key": identity_key(
                TARGET_SESSION,
                session_id=session_id,
                target_id=session_id,
            ),
        }

    if target_type == TARGET_WORK_EVENT:
        if not target_id:
            raise ValueError("work_event target requires target_id (event_id)")
        if not _existence(
            await _row_exists(archive_root, _INSIGHT_QUERIES[TARGET_WORK_EVENT], (target_id, session_id)),
            subject=f"work_event {target_id!r}",
        ):
            raise ValueError(f"work_event {target_id!r} is not in session {session_id!r}")
        return {
            "target_type": TARGET_WORK_EVENT,
            "target_id": target_id,
            "session_id": session_id,
            "message_id": None,
            "identity_key": identity_key(
                TARGET_WORK_EVENT,
                session_id=session_id,
                target_id=target_id,
            ),
        }

    if target_type == TARGET_THREAD:
        if not target_id:
            raise ValueError("thread target requires target_id (thread_id)")
        if not _existence(
            await _row_exists(archive_root, _INSIGHT_QUERIES[TARGET_THREAD], (target_id,)),
            subject=f"thread {target_id!r}",
        ):
            raise ValueError(f"thread {target_id!r} is not a materialized thread root")
        return {
            "target_type": TARGET_THREAD,
            "target_id": target_id,
            "session_id": session_id,
            "message_id": None,
            "identity_key": identity_key(
                TARGET_THREAD,
                session_id=session_id,
                target_id=target_id,
            ),
        }

    if target_type == TARGET_BLOCK:
        if not target_id:
            raise ValueError("block target requires target_id 'message_id:block_index'")
        msg_id, block_part = parse_block_target_id(target_id)
        try:
            block_index = int(block_part)
        except ValueError:
            raise ValueError("block target_id must be 'message_id:block_index'") from None
        if block_index < 0 or str(block_index) != block_part:
            raise ValueError("block target_id must use a canonical non-negative block_index")
        canonical_target_id = f"{msg_id}:{block_index}"
        effective_message_id = message_id or msg_id
        if effective_message_id != msg_id:
            raise ValueError("block message_id must match the message_id in target_id")
        if not _existence(
            await _block_exists(
                archive_root,
                session_id=session_id,
                message_id=effective_message_id,
                block_index_token=str(block_index),
            ),
            subject=f"block {target_id!r}",
        ):
            raise ValueError(f"block {target_id!r} is not present in session {session_id!r}")
        return {
            "target_type": TARGET_BLOCK,
            "target_id": canonical_target_id,
            "session_id": session_id,
            "message_id": effective_message_id,
            "identity_key": identity_key(
                TARGET_BLOCK,
                session_id=session_id,
                target_id=canonical_target_id,
            ),
        }

    if target_type == TARGET_ATTACHMENT:
        if not target_id:
            raise ValueError("attachment target requires target_id")
        # Attachments currently resolve as non-empty tokens scoped to a
        # session; first-class attachment refs belong with #1845.
        return {
            "target_type": TARGET_ATTACHMENT,
            "target_id": target_id,
            "session_id": session_id,
            "message_id": message_id,
            "identity_key": identity_key(
                TARGET_ATTACHMENT,
                session_id=session_id,
                target_id=target_id,
            ),
        }

    if target_type == TARGET_PASTE_SPAN:
        if not target_id:
            raise ValueError("paste_span target requires target_id")
        return {
            "target_type": TARGET_PASTE_SPAN,
            "target_id": target_id,
            "session_id": session_id,
            "message_id": message_id,
            "identity_key": identity_key(
                TARGET_PASTE_SPAN,
                session_id=session_id,
                target_id=target_id,
            ),
        }

    # session/message are resolved by the caller; this branch is
    # defensive and only fires if a future kind is added to the registry
    # without an explicit handler here.
    raise ValueError(f"no resolver handler for target_type {target_type!r}")


__all__ = [
    "ResolvedTarget",
    "parse_block_target_id",
    "resolve_insight_target",
]
