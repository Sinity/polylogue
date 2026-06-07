"""Write/admin method mixin for the session repository."""

from __future__ import annotations

import asyncio
import builtins
import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.insights.feedback import LearningCorrection
from polylogue.logging import get_logger
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.repository.archive.writes.metadata import metadata_read_modify_write
from polylogue.storage.repository.archive.writes.sessions import (
    delete_session_via_backend,
)
from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.search.cache import invalidate_search_cache
from polylogue.storage.sqlite.queries import sessions as sessions_q

logger = get_logger(__name__)

_ARCHIVE_TARGET_TYPES = {
    "session": "session",
    "message": "message",
    "content_block": "block",
    "block": "block",
    "attachment": "attachment",
    "paste_span": "paste_span",
    "work_event": "work_event",
    "phase": "phase",
    "thread": "thread",
}


def _public_target_type(archive_target_type: str, metadata: dict[str, object]) -> str:
    raw = metadata.get("public_target_type")
    return str(raw) if isinstance(raw, str) and raw else archive_target_type


def _ms_to_iso(value: object) -> str:
    import datetime as _dt

    try:
        timestamp = int(str(value)) / 1000
    except (TypeError, ValueError):
        timestamp = 0
    return _dt.datetime.fromtimestamp(timestamp, tz=_dt.UTC).isoformat()


def _json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _json_object_for_write(value: str, field_name: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return dict(parsed)


def _json_list_for_write(value: str, field_name: str) -> list[object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a JSON list") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} must be a JSON list")
    return list(parsed)


class RepositoryWriteMixin:
    if TYPE_CHECKING:
        _backend: RepositoryBackendProtocol

    async def save_parsed_session(self, session: ParsedSession, content_hash: str) -> dict[str, int]:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        db_path = Path(self._backend.db_path)
        with ArchiveStore(db_path.parent) as archive:
            archive.write_parsed(session, content_hash=content_hash)
        invalidate_search_cache()
        return {
            "sessions": 1,
            "messages": len(session.messages),
            "attachments": len(session.attachments),
            "session_events": len(session.session_events),
            "skipped_sessions": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
            "skipped_session_events": 0,
        }

    async def get_metadata(self, session_id: str) -> JSONDocument:
        async with self._backend.connection() as conn:
            return await sessions_q.get_metadata(conn, session_id)

    async def _upsert_normalized_tag(self, session_id: str, tag_name: str) -> None:
        """Upsert a tag into the normalized tags table and link to session."""
        async with self._backend.transaction(), self._backend.connection() as conn:
            await conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
            cursor = await conn.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
            row = await cursor.fetchone()
            if row is not None:
                tag_id = row["id"]
                await conn.execute(
                    "INSERT OR IGNORE INTO session_tags (session_id, tag_id) VALUES (?, ?)",
                    (session_id, tag_id),
                )

    async def _delete_normalized_tag(self, session_id: str, tag_name: str) -> None:
        """Remove a tag link from the normalized tables for a session."""
        async with self._backend.transaction(), self._backend.connection() as conn:
            cursor = await conn.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
            row = await cursor.fetchone()
            if row is not None:
                await conn.execute(
                    "DELETE FROM session_tags WHERE session_id = ? AND tag_id = ?",
                    (session_id, row["id"]),
                )

    async def _metadata_read_modify_write(
        self,
        session_id: str,
        mutator: Callable[[JSONDocument], bool],
    ) -> bool:
        return await metadata_read_modify_write(self._backend, session_id, mutator)

    async def update_metadata(self, session_id: str, key: str, value: JSONValue) -> bool:
        def _set(meta: JSONDocument) -> bool:
            if key in meta and meta[key] == value:
                return False
            meta[key] = value
            return True

        return await self._metadata_read_modify_write(session_id, _set)

    async def delete_metadata(self, session_id: str, key: str) -> bool:
        def _delete(meta: JSONDocument) -> bool:
            if key in meta:
                del meta[key]
                return True
            return False

        return await self._metadata_read_modify_write(session_id, _delete)

    async def add_tag(self, session_id: str, tag: str) -> bool:
        # #1240: tags are stored only in the M2M tables (tags + session_tags).
        # The previous dual-write into ``sessions.metadata['tags']`` was
        # obsolete bookkeeping for the JSON read-fallback that has been removed.
        if not tag or not tag.strip():
            raise ValueError("tag must be a non-empty string")
        if len(tag) > 200:
            raise ValueError("tag must be at most 200 characters")

        async with self._backend.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT 1 FROM session_tags ct
                JOIN tags t ON t.id = ct.tag_id
                WHERE ct.session_id = ? AND t.name = ?
                """,
                (session_id, tag),
            )
            already_present = await cursor.fetchone() is not None

        await self._upsert_normalized_tag(session_id, tag)
        return not already_present

    async def bulk_add_tags(self, session_ids: list[str], tags: list[str]) -> int:
        """Add tags to multiple sessions within a single transaction.

        Args:
            session_ids: List of session IDs to tag.
            tags: List of tag strings to apply to each session.

        Returns:
            Number of sessions whose tag set was actually changed.
        """
        backend = self._backend
        applied_count = 0
        async with backend.transaction(), backend.connection() as conn:
            for session_id in session_ids:
                exists = await conn.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                if not await exists.fetchone():
                    continue
                changed = False
                for tag in tags:
                    await conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
                    cursor = await conn.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                    row = await cursor.fetchone()
                    if row is None:
                        continue
                    insert_result = await conn.execute(
                        "INSERT OR IGNORE INTO session_tags (session_id, tag_id) VALUES (?, ?)",
                        (session_id, row["id"]),
                    )
                    if insert_result.rowcount > 0:
                        changed = True
                if changed:
                    applied_count += 1
        return applied_count

    async def remove_tag(self, session_id: str, tag: str) -> bool:
        async with self._backend.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT 1 FROM session_tags ct
                JOIN tags t ON t.id = ct.tag_id
                WHERE ct.session_id = ? AND t.name = ?
                """,
                (session_id, tag),
            )
            had_link = await cursor.fetchone() is not None

        await self._delete_normalized_tag(session_id, tag)
        return had_link

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        async with self._backend.connection() as conn:
            return await sessions_q.list_tags(conn, provider=provider)

    async def set_metadata(self, session_id: str, metadata: JSONDocument) -> None:
        async with self._backend.connection() as conn:
            await sessions_q.set_metadata(
                conn,
                session_id,
                metadata,
                self._backend.transaction_depth,
            )

    async def delete_session(self, session_id: str) -> bool:
        return await delete_session_via_backend(self._backend, session_id)

    # ------------------------------------------------------------------
    # Marks
    # ------------------------------------------------------------------

    async def add_mark(
        self,
        session_id: str,
        mark_type: str,
        *,
        target_type: str = "session",
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> bool:
        """Add a mark to any supported user-state target. Returns True if newly inserted.

        Supported kinds: see ``polylogue.core.user_state_targets`` (#1113). The
        caller is expected to have resolved/validated ``target_id`` via the
        archive facade resolver; this method enforces the substrate-level
        invariants only.
        """
        from polylogue.core.user_state_targets import TARGET_KIND_NAMES

        if mark_type not in ("star", "pin", "archive"):
            raise ValueError(f"invalid mark_type: {mark_type!r}")
        if target_type not in TARGET_KIND_NAMES:
            raise ValueError(f"invalid target_type: {target_type!r}. Supported: {', '.join(TARGET_KIND_NAMES)}")
        resolved_target_id: str | None
        if target_type == "session":
            resolved_target_id = target_id or session_id
        elif target_type == "message":
            resolved_target_id = target_id or message_id
        else:
            resolved_target_id = target_id
        if resolved_target_id is None:
            raise ValueError(f"target_id is required for {target_type!r} marks")
        return await asyncio.to_thread(
            self._add_archive_mark_sync,
            target_type=target_type,
            target_id=resolved_target_id,
            session_id=session_id,
            message_id=message_id,
            mark_type=mark_type,
        )

    async def remove_mark(self, target_type: str, target_id: str, mark_type: str) -> bool:
        """Remove a mark from a session or message target. Returns True if deleted."""
        return await asyncio.to_thread(self._remove_archive_mark_sync, target_type, target_id, mark_type)

    async def list_marks(
        self,
        *,
        mark_type: str | None = None,
        session_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List marks, optionally filtered by type, target, session, or message."""
        return await asyncio.to_thread(
            self._list_archive_marks_sync,
            mark_type=mark_type,
            session_id=session_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )

    def _archive_user_db_path(self) -> Path:
        archive_root = getattr(self, "_archive_root", None)
        if isinstance(archive_root, Path):
            return archive_root / "user.db"
        return self._backend.db_path.parent / "user.db"

    def _add_archive_mark_sync(
        self,
        *,
        target_type: str,
        target_id: str,
        session_id: str,
        message_id: str | None,
        mark_type: str,
    ) -> bool:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
        from polylogue.storage.sqlite.archive_tiers.user_write import upsert_mark

        archive_target_type = _ARCHIVE_TARGET_TYPES[target_type]
        user_db = self._archive_user_db_path()
        initialize_archive_database(user_db, ArchiveTier.USER)
        conn = sqlite3.connect(user_db)
        try:
            existing = conn.execute(
                "SELECT 1 FROM marks WHERE target_type = ? AND target_id = ? AND mark_type = ?",
                (archive_target_type, target_id, mark_type),
            ).fetchone()
            upsert_mark(
                conn,
                archive_target_type,
                target_id,
                mark_type,
                metadata={
                    "public_target_type": target_type,
                    "session_id": session_id,
                    **({"message_id": message_id} if message_id else {}),
                },
            )
            conn.commit()
            return existing is None
        finally:
            conn.close()

    def _remove_archive_mark_sync(self, target_type: str, target_id: str, mark_type: str) -> bool:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return False
        archive_target_type = _ARCHIVE_TARGET_TYPES[target_type]
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM marks WHERE target_type = ? AND target_id = ? AND mark_type = ?",
                    (archive_target_type, target_id, mark_type),
                )
            return int(cursor.rowcount) > 0
        finally:
            conn.close()

    def _list_archive_marks_sync(
        self,
        *,
        mark_type: str | None,
        session_id: str | None,
        target_type: str | None,
        target_id: str | None,
        message_id: str | None,
    ) -> list[dict[str, str]]:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return []
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT target_type, target_id, mark_type, created_at_ms, metadata_json
                FROM marks
                ORDER BY created_at_ms DESC, mark_id DESC
                """
            ).fetchall()
        finally:
            conn.close()

        import json

        requested_archive_target_type = _ARCHIVE_TARGET_TYPES[target_type] if target_type else None
        items: list[dict[str, str]] = []
        for row in rows:
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except json.JSONDecodeError:
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            public_type = _public_target_type(str(row["target_type"]), metadata)
            row_session_id = str(metadata.get("session_id") or "")
            row_message_id = str(metadata.get("message_id") or "")
            if mark_type and row["mark_type"] != mark_type:
                continue
            if requested_archive_target_type and row["target_type"] != requested_archive_target_type:
                continue
            if target_type and public_type != target_type:
                continue
            if target_id and row["target_id"] != target_id:
                continue
            if session_id and row_session_id != session_id:
                continue
            if message_id and row_message_id != message_id:
                continue
            items.append(
                {
                    "target_type": public_type,
                    "target_id": str(row["target_id"]),
                    "session_id": row_session_id,
                    "message_id": row_message_id,
                    "mark_type": str(row["mark_type"]),
                    "created_at": str(row["created_at_ms"]),
                }
            )
        return items

    async def save_annotation(
        self,
        *,
        annotation_id: str,
        target_type: str,
        target_id: str,
        session_id: str,
        note_text: str,
        message_id: str | None = None,
    ) -> bool:
        """Insert or update an annotation. Returns True if newly inserted."""
        return await asyncio.to_thread(
            self._save_archive_annotation_sync,
            annotation_id=annotation_id,
            target_type=target_type,
            target_id=target_id,
            session_id=session_id,
            message_id=message_id,
            note_text=note_text,
        )

    async def get_annotation(self, annotation_id: str) -> dict[str, str] | None:
        """Get an annotation by ID."""
        return await asyncio.to_thread(self._get_archive_annotation_sync, annotation_id)

    async def list_annotations(
        self,
        *,
        session_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List annotations, optionally filtered by target, session, or message."""
        return await asyncio.to_thread(
            self._list_archive_annotations_sync,
            session_id=session_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )

    async def delete_annotation(self, annotation_id: str) -> bool:
        """Delete an annotation. Returns True if deleted."""
        return await asyncio.to_thread(self._delete_archive_annotation_sync, annotation_id)

    def _save_archive_annotation_sync(
        self,
        *,
        annotation_id: str,
        target_type: str,
        target_id: str,
        session_id: str,
        message_id: str | None,
        note_text: str,
    ) -> bool:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
        from polylogue.storage.sqlite.archive_tiers.user_write import upsert_annotation

        archive_target_type = _ARCHIVE_TARGET_TYPES[target_type]
        user_db = self._archive_user_db_path()
        initialize_archive_database(user_db, ArchiveTier.USER)
        conn = sqlite3.connect(user_db)
        try:
            existing = conn.execute(
                "SELECT 1 FROM annotations WHERE annotation_id = ?",
                (annotation_id,),
            ).fetchone()
            body = self._archive_annotation_body(
                note_text=note_text,
                public_target_type=target_type,
                session_id=session_id,
                message_id=message_id,
            )
            upsert_annotation(
                conn,
                archive_target_type,
                target_id,
                body,
                annotation_id=annotation_id,
            )
            conn.commit()
            return existing is None
        finally:
            conn.close()

    @staticmethod
    def _archive_annotation_body(
        *,
        note_text: str,
        public_target_type: str,
        session_id: str,
        message_id: str | None,
    ) -> str:
        import json

        metadata = {
            "public_target_type": public_target_type,
            "session_id": session_id,
            **({"message_id": message_id} if message_id else {}),
        }
        return json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n" + note_text

    @staticmethod
    def _parse_archive_annotation_body(body: str) -> tuple[dict[str, object], str]:
        import json

        header, sep, rest = body.partition("\n")
        if not sep:
            return {}, body
        try:
            parsed = json.loads(header)
        except json.JSONDecodeError:
            return {}, body
        return (parsed if isinstance(parsed, dict) else {}), rest

    def _get_archive_annotation_sync(self, annotation_id: str) -> dict[str, str] | None:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return None
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT annotation_id, target_type, target_id, body, created_at_ms, updated_at_ms
                FROM annotations
                WHERE annotation_id = ?
                """,
                (annotation_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self._archive_annotation_row_to_dict(row)

    def _list_archive_annotations_sync(
        self,
        *,
        session_id: str | None,
        target_type: str | None,
        target_id: str | None,
        message_id: str | None,
    ) -> list[dict[str, str]]:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return []
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT annotation_id, target_type, target_id, body, created_at_ms, updated_at_ms
                FROM annotations
                ORDER BY updated_at_ms DESC, annotation_id DESC
                """
            ).fetchall()
        finally:
            conn.close()
        requested_archive_target_type = _ARCHIVE_TARGET_TYPES[target_type] if target_type else None
        items: list[dict[str, str]] = []
        for row in rows:
            item = self._archive_annotation_row_to_dict(row)
            if requested_archive_target_type and row["target_type"] != requested_archive_target_type:
                continue
            if target_type and item["target_type"] != target_type:
                continue
            if target_id and item["target_id"] != target_id:
                continue
            if session_id and item["session_id"] != session_id:
                continue
            if message_id and item["message_id"] != message_id:
                continue
            items.append(item)
        return items

    def _archive_annotation_row_to_dict(self, row: sqlite3.Row) -> dict[str, str]:
        metadata, note_text = self._parse_archive_annotation_body(str(row["body"]))
        public_type = _public_target_type(str(row["target_type"]), metadata)
        return {
            "annotation_id": str(row["annotation_id"]),
            "target_type": public_type,
            "target_id": str(row["target_id"]),
            "session_id": str(metadata.get("session_id") or ""),
            "message_id": str(metadata.get("message_id") or ""),
            "note_text": note_text,
            "created_at": str(row["created_at_ms"]),
            "updated_at": str(row["updated_at_ms"]),
        }

    def _delete_archive_annotation_sync(self, annotation_id: str) -> bool:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return False
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                cursor = conn.execute("DELETE FROM annotations WHERE annotation_id = ?", (annotation_id,))
            return int(cursor.rowcount) > 0
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Saved views
    # ------------------------------------------------------------------

    async def save_view(self, view_id: str, name: str, query_json: str) -> bool:
        """Save a named query view. Returns True if newly created."""
        return await asyncio.to_thread(self._save_archive_view_sync, view_id, name, query_json)

    async def get_view(self, view_id: str) -> dict[str, str] | None:
        """Get a saved view by ID."""
        return await asyncio.to_thread(self._get_archive_view_sync, view_id)

    async def get_view_by_name(self, name: str) -> dict[str, str] | None:
        """Get a saved view by name."""
        return await asyncio.to_thread(self._get_archive_view_by_name_sync, name)

    async def list_views(self) -> list[dict[str, str]]:
        """List all saved views."""
        return await asyncio.to_thread(self._list_archive_views_sync)

    async def delete_view(self, view_id: str) -> bool:
        """Delete a saved view. Returns True if deleted."""
        return await asyncio.to_thread(self._delete_archive_view_sync, view_id)

    def _save_archive_view_sync(self, view_id: str, name: str, query_json: str) -> bool:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        user_db = self._archive_user_db_path()
        initialize_archive_database(user_db, ArchiveTier.USER)
        conn = sqlite3.connect(user_db)
        try:
            existing = conn.execute("SELECT 1 FROM saved_views WHERE view_id = ?", (view_id,)).fetchone()
            timestamp = int(time.time() * 1000)
            created_at = (
                int(row[0])
                if (
                    row := conn.execute(
                        "SELECT created_at_ms FROM saved_views WHERE view_id = ? OR name = ?",
                        (view_id, name),
                    ).fetchone()
                )
                is not None
                else timestamp
            )
            with conn:
                conn.execute("DELETE FROM saved_views WHERE name = ? AND view_id != ?", (name, view_id))
                conn.execute(
                    """
                    INSERT INTO saved_views (view_id, name, query_json, created_at_ms, updated_at_ms)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(view_id) DO UPDATE SET
                        name = excluded.name,
                        query_json = excluded.query_json,
                        updated_at_ms = excluded.updated_at_ms
                    """,
                    (
                        view_id,
                        name,
                        json.dumps(
                            _json_object_for_write(query_json, "query_json"), sort_keys=True, separators=(",", ":")
                        ),
                        created_at,
                        timestamp,
                    ),
                )
            return existing is None
        finally:
            conn.close()

    def _archive_view_row_to_dict(self, row: sqlite3.Row) -> dict[str, str]:
        return {
            "view_id": str(row["view_id"]),
            "name": str(row["name"]),
            "query_json": str(row["query_json"]),
            "created_at": _ms_to_iso(row["created_at_ms"]),
        }

    def _get_archive_view_sync(self, view_id: str) -> dict[str, str] | None:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return None
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT view_id, name, query_json, created_at_ms FROM saved_views WHERE view_id = ?",
                (view_id,),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else self._archive_view_row_to_dict(row)

    def _get_archive_view_by_name_sync(self, name: str) -> dict[str, str] | None:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return None
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT view_id, name, query_json, created_at_ms FROM saved_views WHERE name = ?",
                (name,),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else self._archive_view_row_to_dict(row)

    def _list_archive_views_sync(self) -> list[dict[str, str]]:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return []
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT view_id, name, query_json, created_at_ms FROM saved_views ORDER BY created_at_ms DESC"
            ).fetchall()
        finally:
            conn.close()
        return [self._archive_view_row_to_dict(row) for row in rows]

    def _delete_archive_view_sync(self, view_id: str) -> bool:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return False
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                cursor = conn.execute("DELETE FROM saved_views WHERE view_id = ?", (view_id,))
            return int(cursor.rowcount) > 0
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Recall packs
    # ------------------------------------------------------------------

    async def save_recall_pack(self, pack_id: str, label: str, session_ids_json: str, payload_json: str) -> bool:
        """Save a recall pack. Returns True if newly created."""
        return await asyncio.to_thread(
            self._save_archive_recall_pack_sync, pack_id, label, session_ids_json, payload_json
        )

    async def get_recall_pack(self, pack_id: str) -> dict[str, str] | None:
        """Get a recall pack by ID."""
        return await asyncio.to_thread(self._get_archive_recall_pack_sync, pack_id)

    async def list_recall_packs(self) -> list[dict[str, str]]:
        """List all recall packs."""
        return await asyncio.to_thread(self._list_archive_recall_packs_sync)

    async def delete_recall_pack(self, pack_id: str) -> bool:
        """Delete a recall pack. Returns True if deleted."""
        return await asyncio.to_thread(self._delete_archive_recall_pack_sync, pack_id)

    def _save_archive_recall_pack_sync(
        self,
        pack_id: str,
        label: str,
        session_ids_json: str,
        payload_json: str,
    ) -> bool:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        user_db = self._archive_user_db_path()
        initialize_archive_database(user_db, ArchiveTier.USER)
        conn = sqlite3.connect(user_db)
        try:
            existing = conn.execute("SELECT 1 FROM recall_packs WHERE recall_pack_id = ?", (pack_id,)).fetchone()
            timestamp = int(time.time() * 1000)
            created_at = (
                int(row[0])
                if (
                    row := conn.execute(
                        "SELECT created_at_ms FROM recall_packs WHERE recall_pack_id = ?",
                        (pack_id,),
                    ).fetchone()
                )
                is not None
                else timestamp
            )
            payload = _json_object_for_write(payload_json, "payload_json")
            session_ids = _json_list_for_write(session_ids_json, "session_ids_json")
            payload.setdefault("session_ids", session_ids)
            with conn:
                conn.execute(
                    """
                    INSERT INTO recall_packs (recall_pack_id, name, payload_json, created_at_ms, updated_at_ms)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(recall_pack_id) DO UPDATE SET
                        name = excluded.name,
                        payload_json = excluded.payload_json,
                        updated_at_ms = excluded.updated_at_ms
                    """,
                    (pack_id, label, json.dumps(payload, sort_keys=True, separators=(",", ":")), created_at, timestamp),
                )
            return existing is None
        finally:
            conn.close()

    def _archive_recall_pack_row_to_dict(self, row: sqlite3.Row) -> dict[str, str]:
        payload_json = str(row["payload_json"])
        payload = _json_object(payload_json)
        session_ids = payload.get("session_ids", [])
        return {
            "pack_id": str(row["recall_pack_id"]),
            "label": str(row["name"]),
            "session_ids_json": json.dumps(session_ids if isinstance(session_ids, list) else []),
            "payload_json": payload_json,
            "created_at": _ms_to_iso(row["created_at_ms"]),
        }

    def _get_archive_recall_pack_sync(self, pack_id: str) -> dict[str, str] | None:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return None
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT recall_pack_id, name, payload_json, created_at_ms FROM recall_packs WHERE recall_pack_id = ?",
                (pack_id,),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else self._archive_recall_pack_row_to_dict(row)

    def _list_archive_recall_packs_sync(self) -> list[dict[str, str]]:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return []
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT recall_pack_id, name, payload_json, created_at_ms
                FROM recall_packs
                ORDER BY created_at_ms DESC
                """
            ).fetchall()
        finally:
            conn.close()
        return [self._archive_recall_pack_row_to_dict(row) for row in rows]

    def _delete_archive_recall_pack_sync(self, pack_id: str) -> bool:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return False
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                cursor = conn.execute("DELETE FROM recall_packs WHERE recall_pack_id = ?", (pack_id,))
            return int(cursor.rowcount) > 0
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Reader workspaces
    # ------------------------------------------------------------------

    async def save_workspace(
        self,
        *,
        workspace_id: str,
        name: str,
        mode: str,
        open_targets_json: str,
        layout_json: str,
        active_target_json: str,
    ) -> bool:
        """Save a durable reader workspace. Returns True if newly created."""
        return await asyncio.to_thread(
            self._save_archive_workspace_sync,
            workspace_id,
            name,
            mode,
            open_targets_json,
            layout_json,
            active_target_json,
        )

    async def get_workspace(self, workspace_id: str) -> dict[str, str] | None:
        """Get a reader workspace by ID."""
        return await asyncio.to_thread(self._get_archive_workspace_sync, workspace_id)

    async def list_workspaces(self) -> list[dict[str, str]]:
        """List durable reader workspaces."""
        return await asyncio.to_thread(self._list_archive_workspaces_sync)

    async def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a reader workspace. Returns True if deleted."""
        return await asyncio.to_thread(self._delete_archive_workspace_sync, workspace_id)

    def _save_archive_workspace_sync(
        self,
        workspace_id: str,
        name: str,
        mode: str,
        open_targets_json: str,
        layout_json: str,
        active_target_json: str,
    ) -> bool:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        user_db = self._archive_user_db_path()
        initialize_archive_database(user_db, ArchiveTier.USER)
        conn = sqlite3.connect(user_db)
        try:
            existing = conn.execute("SELECT 1 FROM workspaces WHERE workspace_id = ?", (workspace_id,)).fetchone()
            timestamp = int(time.time() * 1000)
            created_at = (
                int(row[0])
                if (
                    row := conn.execute(
                        "SELECT created_at_ms FROM workspaces WHERE workspace_id = ? OR name = ?",
                        (workspace_id, name),
                    ).fetchone()
                )
                is not None
                else timestamp
            )
            settings = {
                "mode": mode,
                "open_targets": _json_list_for_write(open_targets_json, "open_targets_json"),
                "layout": _json_object_for_write(layout_json, "layout_json"),
                "active_target": _json_object_for_write(active_target_json, "active_target_json"),
            }
            with conn:
                conn.execute("DELETE FROM workspaces WHERE name = ? AND workspace_id != ?", (name, workspace_id))
                conn.execute(
                    """
                    INSERT INTO workspaces (workspace_id, name, settings_json, created_at_ms, updated_at_ms)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(workspace_id) DO UPDATE SET
                        name = excluded.name,
                        settings_json = excluded.settings_json,
                        updated_at_ms = excluded.updated_at_ms
                    """,
                    (
                        workspace_id,
                        name,
                        json.dumps(settings, sort_keys=True, separators=(",", ":")),
                        created_at,
                        timestamp,
                    ),
                )
            return existing is None
        finally:
            conn.close()

    def _archive_workspace_row_to_dict(self, row: sqlite3.Row) -> dict[str, str]:
        settings = _json_object(str(row["settings_json"]))
        return {
            "workspace_id": str(row["workspace_id"]),
            "name": str(row["name"]),
            "mode": str(settings.get("mode") or "tabs"),
            "open_targets_json": json.dumps(settings.get("open_targets") or []),
            "layout_json": json.dumps(settings.get("layout") or {}),
            "active_target_json": json.dumps(settings.get("active_target") or {}),
            "created_at": _ms_to_iso(row["created_at_ms"]),
            "updated_at": _ms_to_iso(row["updated_at_ms"]),
        }

    def _get_archive_workspace_sync(self, workspace_id: str) -> dict[str, str] | None:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return None
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT workspace_id, name, settings_json, created_at_ms, updated_at_ms FROM workspaces WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else self._archive_workspace_row_to_dict(row)

    def _list_archive_workspaces_sync(self) -> list[dict[str, str]]:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return []
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT workspace_id, name, settings_json, created_at_ms, updated_at_ms
                FROM workspaces
                ORDER BY updated_at_ms DESC, workspace_id ASC
                """
            ).fetchall()
        finally:
            conn.close()
        return [self._archive_workspace_row_to_dict(row) for row in rows]

    def _delete_archive_workspace_sync(self, workspace_id: str) -> bool:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return False
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                cursor = conn.execute("DELETE FROM workspaces WHERE workspace_id = ?", (workspace_id,))
            return int(cursor.rowcount) > 0
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Learning corrections (#1131)
    #
    # Persisted in ``user_corrections``. Lives outside the content-hash
    # boundary: applying or removing a correction never touches the
    # ``sessions.content_hash`` column. See
    # :mod:`polylogue.storage.insights.feedback` for the SQL surface and
    # :mod:`polylogue.insights.feedback` for the merge semantics.
    # ------------------------------------------------------------------

    async def record_correction(
        self,
        session_id: str,
        kind: str,
        payload: dict[str, str],
        *,
        note: str | None = None,
    ) -> LearningCorrection:
        from polylogue.insights.feedback import parse_correction_kind

        typed_kind = parse_correction_kind(kind)
        return await asyncio.to_thread(
            self._record_archive_correction_sync,
            session_id,
            typed_kind.value,
            payload,
            note,
        )

    async def list_corrections(
        self,
        *,
        session_id: str | None = None,
        kind: str | None = None,
    ) -> builtins.list[LearningCorrection]:
        from polylogue.insights.feedback import parse_correction_kind

        typed_kind = parse_correction_kind(kind) if kind is not None else None
        return await asyncio.to_thread(
            self._list_archive_corrections_sync,
            session_id,
            typed_kind.value if typed_kind is not None else None,
        )

    async def delete_correction(self, session_id: str, kind: str) -> bool:
        from polylogue.insights.feedback import parse_correction_kind

        typed_kind = parse_correction_kind(kind)
        return await asyncio.to_thread(self._delete_archive_correction_sync, session_id, typed_kind.value)

    async def clear_corrections(self, session_id: str) -> int:
        return await asyncio.to_thread(self._clear_archive_corrections_sync, session_id)

    def _record_archive_correction_sync(
        self,
        session_id: str,
        kind: str,
        payload: dict[str, str],
        note: str | None,
    ) -> LearningCorrection:
        from datetime import datetime

        from polylogue.insights.feedback import parse_correction_kind
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
        from polylogue.storage.sqlite.archive_tiers.user_write import upsert_correction

        user_db = self._archive_user_db_path()
        initialize_archive_database(user_db, ArchiveTier.USER)
        conn = sqlite3.connect(user_db)
        try:
            correction_payload: dict[str, object] = dict(payload)
            if note is not None:
                correction_payload["note"] = note
            row = upsert_correction(conn, "session", session_id, kind, correction_payload)
            conn.commit()
            return LearningCorrection(
                session_id=session_id,
                kind=parse_correction_kind(row.correction_type),
                payload={str(key): str(value) for key, value in row.payload.items() if key != "note"},
                note=str(row.payload["note"]) if "note" in row.payload else None,
                created_at=datetime.fromisoformat(_ms_to_iso(row.updated_at_ms)),
            )
        finally:
            conn.close()

    def _list_archive_corrections_sync(
        self,
        session_id: str | None,
        kind: str | None,
    ) -> builtins.list[LearningCorrection]:
        from datetime import datetime

        from polylogue.insights.feedback import parse_correction_kind

        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return []
        clauses: list[str] = ["target_type = 'session'"]
        params: list[object] = []
        if session_id is not None:
            clauses.append("target_id = ?")
            params.append(session_id)
        if kind is not None:
            clauses.append("correction_type = ?")
            params.append(kind)
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                f"""
                SELECT target_id, correction_type, payload_json, updated_at_ms
                FROM corrections
                WHERE {" AND ".join(clauses)}
                ORDER BY target_id, correction_type
                """,
                tuple(params),
            ).fetchall()
        finally:
            conn.close()
        corrections: builtins.list[LearningCorrection] = []
        for row in rows:
            payload_raw = _json_object(str(row["payload_json"]))
            note = payload_raw.pop("note", None)
            corrections.append(
                LearningCorrection(
                    session_id=str(row["target_id"]),
                    kind=parse_correction_kind(str(row["correction_type"])),
                    payload={str(key): str(value) for key, value in payload_raw.items()},
                    note=str(note) if note is not None else None,
                    created_at=datetime.fromisoformat(_ms_to_iso(row["updated_at_ms"])),
                )
            )
        return corrections

    def _delete_archive_correction_sync(self, session_id: str, kind: str) -> bool:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return False
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                cursor = conn.execute(
                    """
                    DELETE FROM corrections
                    WHERE target_type = 'session'
                      AND target_id = ?
                      AND correction_type = ?
                    """,
                    (session_id, kind),
                )
            return int(cursor.rowcount) > 0
        finally:
            conn.close()

    def _clear_archive_corrections_sync(self, session_id: str) -> int:
        user_db = self._archive_user_db_path()
        if not user_db.exists():
            return 0
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM corrections WHERE target_type = 'session' AND target_id = ?",
                    (session_id,),
                )
            return int(cursor.rowcount)
        finally:
            conn.close()
