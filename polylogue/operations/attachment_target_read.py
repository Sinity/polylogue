"""Resolve stable attachment references against pinned Index and retained Source."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.operations.operation_context import PinnedOperationRead


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

    message_id, separator, native_identity = reference_id.rpartition(":attachment:n:")
    if not separator or not message_id:
        return None
    from polylogue.core.identity_law import attachment_reference_id

    try:
        if attachment_reference_id(message_id, native_identity) != reference_id:
            return None
    except ValueError:
        return None

    row = index_conn.execute(
        "SELECT r.ref_id, r.message_id, r.native_identity, r.supplying_raw_id, a.blob_hash, a.byte_count "
        "FROM attachment_refs r JOIN attachments a ON a.attachment_id=r.attachment_id "
        "JOIN messages m ON m.message_id=r.message_id AND m.session_id=r.session_id "
        "WHERE r.session_id=? AND r.ref_id=?",
        (session_id, reference_id),
    ).fetchone()
    if row is None or row[3] is None:
        return None
    if str(row[1]) != message_id or str(row[2]) != native_identity:
        return None
    supplying_raw_id = str(row[3])
    source_raw = source_conn.execute(
        "SELECT blob_hash FROM raw_sessions WHERE raw_id=?",
        (supplying_raw_id,),
    ).fetchone()
    if source_raw is None:
        return None

    blob_hash = bytes(row[4]) if row[4] is not None else None
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
                (supplying_raw_id, coordinate, blob_hash, int(row[5])),
            ).fetchone()
            is None
        ):
            return None
    return str(row[0]), str(row[1])


def _source_declares_attachment(snapshot: PinnedOperationRead, *, session_id: str, ref_id: str, raw_id: str) -> bool:
    """Parse the retained supplier and prove the exact metadata-only ref."""
    from contextlib import closing
    from tempfile import TemporaryDirectory

    from polylogue.core.identity_law import (
        attachment_native_identity,
        attachment_reference_id,
    )
    from polylogue.core.identity_law import (
        session_id as archive_session_id,
    )
    from polylogue.core.sources import origin_from_provider
    from polylogue.operations.source_target_read import _PinnedRetainedRead, _prepare_source_target_artifact
    from polylogue.pipeline.ids import attachment_message_owner_key
    from polylogue.sources.parsers.base_models import ParsedAttachment, ParsedMessage
    from polylogue.sources.prepared_message_sink import SqliteMessageSink, normalize_active_branch
    from polylogue.sources.tool_outcomes import derive_tool_outcomes
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.write import (
        _attachment_id,
        _attachment_message_id_maps,
        _attachment_native_id_values,
        prepared_session_rows_from_shard,
    )

    archive = snapshot.archive
    _provider, blob_hash, _source_path, _kind, _size = archive.raw_revision_descriptor(raw_id)
    if not BlobStore(archive.archive_root / "blob").verify(blob_hash):
        return False
    retained = _PinnedRetainedRead(archive)
    with TemporaryDirectory(prefix="polylogue-attachment-target-") as directory:
        artifact = _prepare_source_target_artifact(retained, raw_id, directory=Path(directory))
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
                raw_messages = session.messages
                origin = origin_from_provider(session.source_name)
                if isinstance(raw_messages, SqliteMessageSink):
                    messages: Sequence[ParsedMessage] = raw_messages.normalized_messages(
                        session.session_events, origin=origin
                    )
                else:
                    messages = derive_tool_outcomes(
                        normalize_active_branch(raw_messages), session.session_events, origin=origin
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

            def agrees_with_index_descriptor(
                attachment: ParsedAttachment, message_id: str, expected_identity: str
            ) -> bool:
                row = archive._conn.execute(
                    "SELECT r.native_identity,a.attachment_id,a.display_name,a.media_type,a.byte_count "
                    "FROM attachment_refs r JOIN attachments a ON a.attachment_id=r.attachment_id "
                    "WHERE r.session_id=? AND r.ref_id=?",
                    (session_id, ref_id),
                ).fetchone()
                if row is None:
                    return False
                expected_native = set(_attachment_native_id_values(attachment))
                actual_native = {
                    (str(native[0]), str(native[1]))
                    for native in archive._conn.execute(
                        "SELECT id_kind,native_id FROM attachment_native_ids WHERE ref_id=?",
                        (ref_id,),
                    )
                }
                return (
                    str(row[0]) == expected_identity
                    and attachment_reference_id(message_id, expected_identity) == ref_id
                    and str(row[1]) == _attachment_id("", attachment, blob_hash=None)
                    and row[2] == attachment.name
                    and row[3] == attachment.mime_type
                    and int(row[4]) == int(attachment.size_bytes or 0)
                    and actual_native == expected_native
                )

            for attachment in attachments:
                owner_key = attachment_message_owner_key(attachment, rows.owner_resolution)
                message_id = by_owner_key.get(owner_key) if owner_key is not None else None
                if message_id is None:
                    continue
                try:
                    expected_identity = attachment_native_identity(attachment.provider_attachment_id)
                    expected_ref_id = attachment_reference_id(message_id, expected_identity)
                except ValueError:
                    continue
                if expected_ref_id == ref_id and agrees_with_index_descriptor(
                    attachment, message_id, expected_identity
                ):
                    return True
            return False
        except (KeyError, ValueError):
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
    captured_attachment = tuple(
        index_connection.execute(
            "SELECT a.* FROM attachment_refs r JOIN attachments a ON a.attachment_id=r.attachment_id "
            "WHERE r.session_id=? AND r.ref_id=?",
            (session_id, ref_id),
        ).fetchone()
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
        from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

        if ArchiveIdentity.resolve_location(ArchiveLocation.resolve(archive.archive_root)) != snapshot.identity:
            raise ValueError(f"attachment Source archive {ref_id!r} changed before durable apply")
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
        now_attachment_row = (
            current_index_connection.execute(
                "SELECT a.* FROM attachment_refs r JOIN attachments a ON a.attachment_id=r.attachment_id "
                "WHERE r.session_id=? AND r.ref_id=?",
                (session_id, ref_id),
            ).fetchone()
            if now_ref is not None
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
