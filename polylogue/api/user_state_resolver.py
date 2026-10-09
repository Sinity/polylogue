"""Resolution helpers for non-session/message user-state targets (#1113).

These helpers validate that a target identified by
``(target_type, target_id, session_id, message_id)`` actually exists
in the archive before a mark or annotation is written, and produce the
canonical ``identity_key`` used by recall packs and workspaces.

The resolver returns the validated ``ResolvedTarget`` mapping or raises
``ValueError`` with a specific, surface-friendly message. Insight kinds
(``session``, ``thread``) are validated against the
respective insight tables; ``block`` and ``attachment`` are validated
against the archive substrate; ``paste_span`` is treated as an opaque
block-derived identifier and only validated for non-empty ``target_id``.
"""

from __future__ import annotations

import sqlite3
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, TypeVar

from polylogue.api.archive import open_readonly_connection
from polylogue.core.compute import compute_adapter, current_cancellation
from polylogue.core.evidence import Empty, Evidence, Measured, Unavailable, resolve
from polylogue.core.refs import ObjectRef
from polylogue.core.user_state_targets import (
    TARGET_ATTACHMENT,
    TARGET_BLOCK,
    TARGET_PASTE_SPAN,
    TARGET_SESSION,
    TARGET_THREAD,
    identity_key,
    validate_target_kind,
)
from polylogue.operations.attachment_target_read import _resolve_attachment_in_connections, _source_declares_attachment

if TYPE_CHECKING:
    from polylogue.operations.operation_context import PinnedOperationRead


class ResolvedTarget(TypedDict, total=False):
    """Storage row payload + identity key for a resolved user-state target."""

    target_type: str
    target_id: str
    session_id: str
    message_id: str | None
    identity_key: str


_INSIGHT_QUERIES: dict[str, str] = {
    TARGET_SESSION: "SELECT 1 FROM session_profiles WHERE session_id = ?",
    TARGET_THREAD: "SELECT 1 FROM threads WHERE thread_id = ?",
}
_T = TypeVar("_T")


@contextmanager
def _read_connection(path: Path) -> Iterator[sqlite3.Connection]:
    conn = open_readonly_connection(path, timeout_class="interactive-read", validate_schema=False)
    cancellation = current_cancellation()
    if cancellation is not None:
        cancellation.register_connection(conn)
    primary: BaseException | None = None
    try:
        yield conn
    except BaseException as failure:
        primary = failure
    try:
        conn.close()
    except BaseException as cleanup:
        if primary is not None:
            raise BaseExceptionGroup("existence probe and connection settlement failed", [primary, cleanup]) from None
        raise
    if cancellation is not None:
        cancellation.unregister_connection(conn)
    if primary is not None:
        raise primary


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
        with _read_connection(candidate) as conn:
            row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone()
    except sqlite3.Error as exc:
        return Unavailable(reason="index_tier_unreadable", detail=f"{type(exc).__name__}: {exc}")
    return Measured(candidate) if row is not None else Empty()


def _existence(evidence: Evidence[_T], *, subject: str, empty: Callable[[], _T]) -> _T:
    """Consume an existence probe; an unreadable tier refuses, never denies."""

    def _refuse(case: Unavailable) -> _T:
        raise ValueError(f"existence of {subject} could not be checked: {case.detail or case.reason}")

    return resolve(
        evidence,
        measured=lambda value: value,
        empty=empty,
        unavailable=_refuse,
        degraded=lambda case: case.value,
    )


def _row_exists_sync(db_path: Path, sql: str, params: tuple[object, ...]) -> bool:
    with _read_connection(db_path) as conn:
        row = conn.execute(sql, params).fetchone()
    return row is not None


async def _row_exists(archive_root: Path, sql: str, params: tuple[object, ...]) -> Evidence[bool]:
    """Existence probe against the `index.db`. A missing index means
    nothing is materialized, so the row does not exist; an unreadable one
    means the question was not answered."""

    def read() -> Evidence[bool]:
        located = _index_db_path(archive_root)
        if not isinstance(located, Measured):
            return located if isinstance(located, Unavailable) else Empty()
        try:
            return Measured(_row_exists_sync(located.value, sql, params))
        except sqlite3.Error as exc:
            return Unavailable(reason="index_tier_unreadable", detail=f"{type(exc).__name__}: {exc}")

    return (
        await compute_adapter()
        .submit(
            read,
            admission_class="interactive-read",
            estimated_bytes=len(sql.encode("utf-8")) + sum(len(str(value).encode("utf-8")) for value in params),
        )
        .wait()
    )


def _resolve_block_sync(
    db_path: Path,
    *,
    session_id: str,
    target_id: str,
    message_id: str | None,
) -> tuple[str, str] | None:
    with _read_connection(db_path) as conn:
        return _resolve_block_in_connection(
            conn,
            session_id=session_id,
            target_id=target_id,
            message_id=message_id,
        )


def _resolve_block_in_connection(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    target_id: str,
    message_id: str | None,
) -> tuple[str, str] | None:
    selector = ObjectRef.parse(f"block:{target_id}")
    if not selector.qualifiers:
        sql = "SELECT block_id, message_id FROM blocks WHERE block_id=?"
        params: tuple[object, ...] = (selector.object_id,)
        if message_id is not None:
            sql += " AND message_id=?"
            params += (message_id,)
    else:
        selected_message = selector.object_id
        block_part = selector.qualifiers[0]
        try:
            block_index = int(block_part)
        except ValueError:
            raise ValueError("block target_id must be 'message_id:block_index' or a stable block_id") from None
        if block_index < 0 or str(block_index) != block_part:
            raise ValueError("block target_id must use a canonical non-negative block_index")
        if message_id is not None and message_id != selected_message:
            raise ValueError("block message_id must match the message_id in target_id")
        sql = "SELECT block_id, message_id FROM blocks WHERE message_id=? AND position=?"
        params = (selected_message, block_index)
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    from polylogue.storage.sqlite.archive_tiers.write import locate_composed_message

    if locate_composed_message(conn, session_id, str(row[1])) is None:
        return None
    return str(row[0]), str(row[1])


async def _resolve_block(
    archive_root: Path,
    *,
    session_id: str,
    target_id: str,
    message_id: str | None,
) -> Evidence[tuple[str, str] | None]:

    def read() -> Evidence[tuple[str, str] | None]:
        located = _index_db_path(archive_root)
        if not isinstance(located, Measured):
            return located if isinstance(located, Unavailable) else Empty()
        try:
            return Measured(
                _resolve_block_sync(
                    located.value,
                    session_id=session_id,
                    target_id=target_id,
                    message_id=message_id,
                )
            )
        except sqlite3.Error as exc:
            return Unavailable(reason="index_tier_unreadable", detail=f"{type(exc).__name__}: {exc}")

    return (
        await compute_adapter()
        .submit(
            read,
            admission_class="interactive-read",
            estimated_bytes=len(session_id.encode("utf-8"))
            + len((message_id or "").encode("utf-8"))
            + len(target_id.encode("utf-8")),
        )
        .wait()
    )


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
    index_connection: sqlite3.Connection | None = None,
    source_connection: sqlite3.Connection | None = None,
    snapshot: PinnedOperationRead | None = None,
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
            empty=lambda: False,
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

    if target_type == TARGET_THREAD:
        if not target_id:
            raise ValueError("thread target requires target_id (thread_id)")
        if not _existence(
            await _row_exists(archive_root, _INSIGHT_QUERIES[TARGET_THREAD], (target_id,)),
            subject=f"thread {target_id!r}",
            empty=lambda: False,
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
            raise ValueError("block target requires a positional selector or stable block_id")
        if index_connection is not None:
            resolved_block = _resolve_block_in_connection(
                index_connection,
                session_id=session_id,
                target_id=target_id,
                message_id=message_id,
            )
        else:
            resolved_block = _existence(
                await _resolve_block(
                    archive_root,
                    session_id=session_id,
                    target_id=target_id,
                    message_id=message_id,
                ),
                subject=f"block {target_id!r}",
                empty=lambda: None,
            )
        if resolved_block is None:
            raise ValueError(f"block {target_id!r} is not present in session {session_id!r}")
        canonical_target_id, effective_message_id = resolved_block
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
            raise ValueError("attachment target requires a stable attachment reference_id")
        if index_connection is None or source_connection is None:
            raise ValueError("attachment target requires a pinned Index and Source read")
        resolved_attachment = _resolve_attachment_in_connections(
            index_connection,
            source_connection,
            session_id=session_id,
            reference_id=target_id,
        )
        if resolved_attachment is None:
            raise ValueError(f"attachment reference {target_id!r} is not Source-bound in session {session_id!r}")
        payload = index_connection.execute(
            "SELECT a.blob_hash, r.supplying_raw_id FROM attachment_refs r JOIN attachments a "
            "ON a.attachment_id=r.attachment_id WHERE r.session_id=? AND r.ref_id=?",
            (session_id, target_id),
        ).fetchone()
        if payload is None or (
            payload[0] is None
            and (
                snapshot is None
                or not _source_declares_attachment(
                    snapshot,
                    session_id=session_id,
                    ref_id=target_id,
                    raw_id=str(payload[1]),
                )
            )
        ):
            raise ValueError(f"attachment reference {target_id!r} has no matching retained Source descriptor")
        canonical_target_id, effective_message_id = resolved_attachment
        return {
            "target_type": TARGET_ATTACHMENT,
            "target_id": canonical_target_id,
            "session_id": session_id,
            "message_id": effective_message_id,
            "identity_key": identity_key(
                TARGET_ATTACHMENT,
                session_id=session_id,
                target_id=canonical_target_id,
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
