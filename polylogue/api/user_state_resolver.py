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


def _existence(evidence: Evidence[_T], *, subject: str) -> _T:
    """Consume an existence probe; an unreadable tier refuses, never denies."""

    def _refuse(case: Unavailable) -> _T:
        raise ValueError(f"existence of {subject} could not be checked: {case.detail or case.reason}")

    return resolve(
        evidence,
        measured=lambda value: value,
        empty=lambda: False,
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

    def read() -> Evidence[tuple[str, str] | None]:
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
        sql = "SELECT block_id, message_id FROM blocks WHERE session_id=? AND block_id=?"
        params: tuple[object, ...] = (session_id, selector.object_id)
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
        sql = "SELECT block_id, message_id FROM blocks WHERE session_id=? AND message_id=? AND position=?"
        params = (session_id, selected_message, block_index)
    row = conn.execute(sql, params).fetchone()
    return None if row is None else (str(row[0]), str(row[1]))


def _resolve_attachment_in_connections(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    *,
    session_id: str,
    reference_id: str,
) -> tuple[str, str] | None:
    """Resolve a stable attachment reference and prove its original Raw owner.

    ``attachment_id`` is content-versioned when payload acquisition succeeds.
    It therefore cannot name a durable user target. The generated ``ref_id``
    identifies the original message coordinate; ``supplying_raw_id`` names
    the Source row that first contributed it and survives carry-forward.
    """

    row = index_conn.execute(
        "SELECT r.ref_id, r.message_id, r.supplying_raw_id, a.blob_hash, a.byte_count "
        "FROM attachment_refs r JOIN attachments a ON a.attachment_id=r.attachment_id "
        "JOIN messages m ON m.message_id=r.message_id AND m.session_id=r.session_id "
        "WHERE r.session_id=? AND r.ref_id=?",
        (session_id, reference_id),
    ).fetchone()
    if row is None or row[2] is None:
        return None
    supplying_raw_id = str(row[2])
    source_raw = source_conn.execute(
        "SELECT blob_hash FROM raw_sessions WHERE raw_id=?",
        (supplying_raw_id,),
    ).fetchone()
    if source_raw is None:
        return None

    blob_hash = bytes(row[3]) if row[3] is not None else None
    if blob_hash is not None:
        native_rows = index_conn.execute(
            "SELECT id_kind, native_id FROM attachment_native_ids "
            "WHERE ref_id=? AND id_kind IN ('file', 'attachment') ORDER BY id_kind, native_id",
            (reference_id,),
        ).fetchall()
        file_ids = [str(native_id) for id_kind, native_id in native_rows if str(id_kind) == "file"]
        attachment_ids = [str(native_id) for id_kind, native_id in native_rows if str(id_kind) == "attachment"]
        if len(file_ids) == 1:
            coordinate = f"attachment:{file_ids[0]}"
        elif not file_ids and len(attachment_ids) == 1:
            coordinate = f"attachment-ref:{attachment_ids[0]}"
        else:
            return None
        source_blob_hash = bytes(source_raw[0]) if source_raw[0] is not None else None
        if (
            source_blob_hash is None
            or source_conn.execute(
                "SELECT 1 FROM blob_refs WHERE ref_id=? AND ref_type='attachment' AND source_path=? "
                "AND blob_hash=? AND size_bytes=? LIMIT 1",
                (supplying_raw_id, coordinate, blob_hash, int(row[4])),
            ).fetchone()
            is None
        ):
            return None
    return str(row[0]), str(row[1])


def _source_declares_attachment(snapshot: PinnedOperationRead, *, session_id: str, ref_id: str, raw_id: str) -> bool:
    """Parse the retained supplier and prove the exact metadata-only ref."""
    from contextlib import closing
    from tempfile import TemporaryDirectory

    from polylogue.core.identity_law import session_id as archive_session_id
    from polylogue.core.sources import origin_from_provider
    from polylogue.operations.source_target_read import _PinnedRetainedRead
    from polylogue.pipeline.ids import attachment_message_owner_key
    from polylogue.sources.dispatch import is_jsonl_source_path
    from polylogue.sources.prepared_message_sink import SqliteMessageSink, normalize_active_branch
    from polylogue.sources.revision_backfill import (
        prepare_retained_jsonl_artifact,
        prepare_retained_non_json_artifact,
    )
    from polylogue.sources.tool_outcomes import derive_tool_outcomes
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.write import (
        _attachment_id,
        _attachment_message_id_maps,
        _attachment_reference_positions,
        prepared_session_rows_from_shard,
    )

    archive = snapshot.archive
    provider, blob_hash, source_path, _kind, _size = archive.raw_revision_descriptor(raw_id)
    if not BlobStore(archive.archive_root / "blob").verify(blob_hash):
        return False
    retained = _PinnedRetainedRead(archive)
    with TemporaryDirectory(prefix="polylogue-attachment-target-") as directory:
        prepare = (
            prepare_retained_jsonl_artifact
            if is_jsonl_source_path(source_path) or Path(source_path).suffix.lower() == ".json"
            else prepare_retained_non_json_artifact
        )
        artifact = prepare(
            retained,
            raw_id,
            directory=Path(directory),
            prepare_blob_publications=False,
        )
        try:
            if artifact.error is not None or artifact.shard_path is None:
                return False
            artifact.verify_files(full=True)
            rows = prepared_session_rows_from_shard(artifact.shard_path, session_id)
            with closing(artifact.iter_sessions()) as parsed:
                session = next(
                    (
                        value
                        for value in parsed
                        if archive_session_id(origin_from_provider(value.source_name).value, value.provider_session_id)
                        == session_id
                    ),
                    None,
                )
            if session is None:
                return False
            messages = session.messages
            origin = origin_from_provider(session.source_name)
            if isinstance(messages, SqliteMessageSink):
                messages = messages.normalized_messages(session.session_events, origin=origin)
            else:
                messages = derive_tool_outcomes(
                    normalize_active_branch(messages), session.session_events, origin=origin
                )
            attachments = tuple(session.attachments)
            wanted_owner_keys = {
                key
                for attachment in attachments
                if (key := attachment_message_owner_key(attachment, rows.owner_resolution))
            }
            _resolution, by_owner_key, _owning_messages = _attachment_message_id_maps(
                session_id,
                messages,
                content_identities=rows.content_identities,
                owner_resolution=rows.owner_resolution,
                wanted_owner_keys=wanted_owner_keys,
            )
            attachments_by_message: dict[str, list[object]] = {}
            for attachment in attachments:
                owner_key = attachment_message_owner_key(attachment, rows.owner_resolution)
                message_id = by_owner_key.get(owner_key) if owner_key is not None else None
                if message_id is not None:
                    attachments_by_message.setdefault(message_id, []).append(attachment)
            positions = {
                key: position
                for message_attachments in attachments_by_message.values()
                for key, position in _attachment_reference_positions(message_attachments).items()
            }
            return any(
                (owner_key := attachment_message_owner_key(attachment, rows.owner_resolution)) is not None
                and by_owner_key.get(owner_key) is not None
                and f"{by_owner_key[owner_key]}:attachment:{positions.get(attachment.acquisition_key)}" == ref_id
                and _attachment_id("", attachment)
                == str(
                    archive._conn.execute(
                        "SELECT a.attachment_id FROM attachment_refs r JOIN attachments a "
                        "ON a.attachment_id=r.attachment_id WHERE r.session_id=? AND r.ref_id=?",
                        (session_id, ref_id),
                    ).fetchone()[0]
                )
                for attachment in attachments
            )
        except (KeyError, ValueError, sqlite3.Error):
            return False
        finally:
            artifact.discard()


def bind_attachment_source_guard(
    snapshot: PinnedOperationRead,
    *,
    session_id: str,
    ref_id: str,
    current_index_connection: sqlite3.Connection,
    current_source_connection: sqlite3.Connection,
) -> Callable[[], None]:
    """Capture attachment Source currency and recheck it at durable apply.

    The stable Index reference is admitted only while it resolves to its exact
    Source supplier. The closure retains the supplier row and all attachment
    blob coordinates for that Raw, so a later call detects supplier removal,
    replacement, or payload relinking.
    """
    archive = snapshot.archive
    index_connection = archive._conn
    source_connection = archive.source_connection
    resolved = _resolve_attachment_in_connections(
        index_connection, source_connection, session_id=session_id, reference_id=ref_id
    )
    if resolved is None:
        raise ValueError(f"attachment reference {ref_id!r} is not Source-bound in session {session_id!r}")
    supplier_row = index_connection.execute(
        "SELECT supplying_raw_id FROM attachment_refs WHERE session_id=? AND ref_id=?",
        (session_id, ref_id),
    ).fetchone()
    if supplier_row is None or supplier_row[0] is None:
        raise ValueError(f"attachment reference {ref_id!r} has no Source supplier")
    raw_id = str(supplier_row[0])
    raw = source_connection.execute("SELECT * FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()
    if raw is None:
        raise ValueError(f"attachment Source supplier {raw_id!r} is unavailable")
    payload_row = index_connection.execute(
        "SELECT a.blob_hash FROM attachment_refs r JOIN attachments a ON a.attachment_id=r.attachment_id "
        "WHERE r.session_id=? AND r.ref_id=?",
        (session_id, ref_id),
    ).fetchone()
    if payload_row is None or (
        payload_row[0] is None
        and not _source_declares_attachment(snapshot, session_id=session_id, ref_id=ref_id, raw_id=raw_id)
    ):
        raise ValueError(f"attachment reference {ref_id!r} has no matching retained Source descriptor")
    raw_columns = tuple(str(row[1]) for row in source_connection.execute("PRAGMA table_info(raw_sessions)"))
    index_ref_columns = tuple(str(row[1]) for row in index_connection.execute("PRAGMA table_info(attachment_refs)"))
    index_attachment_columns = tuple(str(row[1]) for row in index_connection.execute("PRAGMA table_info(attachments)"))
    captured_ref = tuple(
        index_connection.execute(
            "SELECT * FROM attachment_refs WHERE session_id=? AND ref_id=?", (session_id, ref_id)
        ).fetchone()
    )
    attachment_id = str(captured_ref[index_ref_columns.index("attachment_id")])
    captured_attachment = tuple(
        index_connection.execute("SELECT * FROM attachments WHERE attachment_id=?", (attachment_id,)).fetchone()
    )
    captured_native_ids = tuple(
        tuple(row)
        for row in index_connection.execute(
            "SELECT * FROM attachment_native_ids WHERE ref_id=? ORDER BY id_kind,native_id", (ref_id,)
        ).fetchall()
    )
    blobs = tuple(
        tuple(row)
        for row in source_connection.execute(
            "SELECT * FROM blob_refs WHERE ref_id=? AND ref_type='attachment' ORDER BY source_path,blob_hash,size_bytes",
            (raw_id,),
        ).fetchall()
    )
    blob_columns = tuple(str(row[1]) for row in source_connection.execute("PRAGMA table_info(blob_refs)"))
    captured_raw = tuple(raw)

    def revalidate() -> None:
        current = _resolve_attachment_in_connections(
            current_index_connection,
            current_source_connection,
            session_id=session_id,
            reference_id=ref_id,
        )
        now_supplier = current_index_connection.execute(
            "SELECT supplying_raw_id FROM attachment_refs WHERE session_id=? AND ref_id=?",
            (session_id, ref_id),
        ).fetchone()
        now_raw = current_source_connection.execute("SELECT * FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()
        now_blobs = tuple(
            tuple(row)
            for row in current_source_connection.execute(
                "SELECT * FROM blob_refs WHERE ref_id=? AND ref_type='attachment' ORDER BY source_path,blob_hash,size_bytes",
                (raw_id,),
            ).fetchall()
        )
        now_ref_row = current_index_connection.execute(
            "SELECT * FROM attachment_refs WHERE session_id=? AND ref_id=?", (session_id, ref_id)
        ).fetchone()
        now_ref = tuple(now_ref_row) if now_ref_row is not None else None
        now_attachment_id = (
            str(now_ref[index_ref_columns.index("attachment_id")])
            if now_ref is not None and "attachment_id" in index_ref_columns
            else None
        )
        now_attachment_row = (
            current_index_connection.execute(
                "SELECT * FROM attachments WHERE attachment_id=?", (now_attachment_id,)
            ).fetchone()
            if now_attachment_id is not None
            else None
        )
        now_attachment = tuple(now_attachment_row) if now_attachment_row is not None else None
        now_native_ids = tuple(
            tuple(row)
            for row in current_index_connection.execute(
                "SELECT * FROM attachment_native_ids WHERE ref_id=? ORDER BY id_kind,native_id", (ref_id,)
            ).fetchall()
        )
        if (
            current != resolved
            or now_supplier is None
            or now_supplier[0] != raw_id
            or now_raw is None
            or tuple(now_raw) != captured_raw
            or tuple(str(row[1]) for row in current_source_connection.execute("PRAGMA table_info(raw_sessions)"))
            != raw_columns
            or now_blobs != blobs
            or tuple(str(row[1]) for row in current_source_connection.execute("PRAGMA table_info(blob_refs)"))
            != blob_columns
            or tuple(str(row[1]) for row in current_index_connection.execute("PRAGMA table_info(attachment_refs)"))
            != index_ref_columns
            or tuple(str(row[1]) for row in current_index_connection.execute("PRAGMA table_info(attachments)"))
            != index_attachment_columns
            or now_ref != captured_ref
            or now_attachment != captured_attachment
            or now_native_ids != captured_native_ids
        ):
            raise ValueError(f"attachment Source reference {ref_id!r} changed before durable apply")

    return revalidate


async def _resolve_block(
    archive_root: Path,
    *,
    session_id: str,
    target_id: str,
    message_id: str | None,
) -> Evidence[tuple[str, str] | None]:

    def read() -> Evidence[bool]:
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
