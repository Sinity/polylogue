"""Shared storage-record builders and DB seeding helpers for tests."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast
from uuid import uuid4

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import (
    BlockType,
    Origin,
    Provider,
    SemanticBlockType,
    TitleSource,
    ToolOutcome,
    ValidationMode,
    ValidationStatus,
)
from polylogue.core.json import dumps, loads, require_json_document, require_json_value
from polylogue.core.sources import origin_from_provider, provider_from_origin
from polylogue.core.timestamps import _timestamp_sort_key
from polylogue.core.types import AttachmentId, ContentHash, MessageId, SessionId
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
)
from polylogue.storage.runtime import (
    AttachmentRecord,
    BlockRecord,
    MessageRecord,
    RawSessionRecord,
    SessionRecord,
)
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection import connection_context, open_connection
from tests.infra.live_ingest import write_session_counts_sync

if TYPE_CHECKING:
    from polylogue.archive.session.domain_models import Session

JSONRecord: TypeAlias = dict[str, object]
MessageMapping: TypeAlias = Mapping[str, object]
RecordPayload: TypeAlias = dict[str, object]


class _AutoTimestampSentinel:
    """Marker for builders that should synthesize a fresh timestamp."""


class _AutoMessageIdSentinel:
    """Marker for builders that should target the most recent message."""


# Thread-safety lock for writes (matches store.py pattern)
_WRITE_LOCK = threading.Lock()
_AUTO_TIMESTAMP: Final = _AutoTimestampSentinel()
_AUTO_MESSAGE_ID: Final = _AutoMessageIdSentinel()

#: ``MessageRecord.identity_source`` values, matching the two branches of the
#: computed ``messages.message_id``. The builders below synthesize a provider
#: message id, so they declare ``NATIVE_IDENTITY``; pass
#: ``identity_source=CONTENT_DERIVED_IDENTITY`` to seed the other shape -- an
#: id-less export whose identity is the digest of its own declared semantics.
#: That digest covers ``timestamp``/``occurred_at_ms``, so a content-derived
#: message needs an explicit timestamp to get a reproducible id.
NATIVE_IDENTITY: Final = "native"
CONTENT_DERIVED_IDENTITY: Final = "content"


def _session_id(value: str) -> SessionId:
    return SessionId(value)


def _message_id(value: str) -> MessageId:
    return MessageId(value)


def _attachment_id(value: str) -> AttachmentId:
    return AttachmentId(value)


def _content_hash(value: str) -> ContentHash:
    return ContentHash(_writer_hash(value))


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _optional_json_document(value: object, *, context: str = "JSON object") -> JSONRecord | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None
    return cast(JSONRecord, dict(require_json_document(dict(value), context=context)))


def _json_string_or_none(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return dumps(require_json_value(value, context=context))


def _coerce_str(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _coerce_int(value: object, default: int) -> int:
    return value if isinstance(value, int) else default


def _coerce_sort_key(value: object, default: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_content_hash(value: object, default: str) -> ContentHash:
    return _content_hash(_coerce_str(value, default))


def _origin_value(provider: str) -> Origin:
    return origin_from_provider(Provider.from_string(provider))


def _resolve_timestamp(value: str | None | _AutoTimestampSentinel) -> str | None:
    return datetime.now(timezone.utc).isoformat() if isinstance(value, _AutoTimestampSentinel) else value


def _resolve_attachment_message_id(
    *,
    value: str | None | _AutoMessageIdSentinel,
    messages: list[MessageRecord],
) -> str | None:
    if isinstance(value, _AutoMessageIdSentinel):
        return str(messages[-1].message_id) if messages else None
    return value


def _coerce_builder_timestamp(value: object) -> str | None | _AutoTimestampSentinel:
    if isinstance(value, str) or value is None:
        return value
    return _AUTO_TIMESTAMP


def _merge_media_type_into_metadata(metadata: str | None, media_type: str | None) -> str | None:
    """#1240: store media_type inside the block-metadata JSON envelope."""
    if not media_type:
        return metadata
    base: dict[str, object] = {}
    if metadata:
        try:
            parsed = loads(metadata)
        except Exception:
            return metadata
        if isinstance(parsed, dict):
            base.update(parsed)
    base.setdefault("media_type", media_type)
    return dumps(base)


def _content_block_record(
    *,
    message_id: str,
    session_id: str,
    block_index: int,
    block_type: str,
    text: str | None = None,
    tool_name: str | None = None,
    tool_id: str | None = None,
    tool_input: str | None = None,
    media_type: str | None = None,
    metadata: str | None = None,
    semantic_type: str | None = None,
    tool_result_is_error: int | None = None,
    tool_result_exit_code: int | None = None,
    tool_outcome: ToolOutcome | None = None,
    tool_result_outcome_unknown_reason: str | None = None,
) -> BlockRecord:
    # #1240: media_type is now stored inside the block-metadata JSON.
    merged_metadata = _merge_media_type_into_metadata(metadata, media_type)
    return BlockRecord(
        block_id=BlockRecord.make_id(message_id, block_index),
        message_id=_message_id(message_id),
        session_id=_session_id(session_id),
        block_index=block_index,
        type=BlockType.from_string(block_type),
        text=text,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_input=tool_input,
        metadata=merged_metadata,
        semantic_type=None if semantic_type is None else SemanticBlockType.from_string(semantic_type),
        tool_result_is_error=tool_result_is_error,
        tool_result_exit_code=tool_result_exit_code,
        tool_outcome=tool_outcome,
        tool_result_outcome_unknown_reason=tool_result_outcome_unknown_reason,
    )


def _content_block_from_mapping(
    *,
    block: MessageMapping,
    message_id: str,
    session_id: str,
    block_index: int,
) -> BlockRecord:
    raw_tool_input = block.get("tool_input", block.get("input"))
    raw_metadata = block.get("metadata")
    return _content_block_record(
        message_id=message_id,
        session_id=session_id,
        block_index=block_index,
        block_type=_optional_str(block.get("type")) or "text",
        text=_optional_str(block.get("text")),
        tool_name=_optional_str(block.get("tool_name")) or _optional_str(block.get("name")),
        tool_id=_optional_str(block.get("tool_id")) or _optional_str(block.get("id")),
        tool_input=_json_string_or_none(raw_tool_input, context="content block tool input"),
        media_type=_optional_str(block.get("media_type")),
        metadata=_json_string_or_none(raw_metadata, context="content block metadata"),
        semantic_type=_optional_str(block.get("semantic_type")),
        tool_result_is_error=_optional_int(block.get("tool_result_is_error", block.get("is_error"))),
        tool_result_exit_code=_optional_int(block.get("tool_result_exit_code", block.get("exit_code"))),
        tool_outcome=(ToolOutcome(str(block["tool_outcome"])) if block.get("tool_outcome") is not None else None),
        tool_result_outcome_unknown_reason=_optional_str(block.get("tool_result_outcome_unknown_reason")),
    )


def _normalize_content_blocks(
    *,
    raw_blocks: object,
    message_id: str,
    session_id: str,
) -> list[BlockRecord]:
    if not isinstance(raw_blocks, list):
        return []
    blocks: list[BlockRecord] = []
    for idx, raw_block in enumerate(raw_blocks):
        if isinstance(raw_block, BlockRecord):
            blocks.append(raw_block)
            continue
        if isinstance(raw_block, Mapping):
            if not all(isinstance(key, str) for key in raw_block):
                raise TypeError("content block keys must be strings")
            blocks.append(
                _content_block_from_mapping(
                    block=cast(MessageMapping, raw_block),
                    message_id=message_id,
                    session_id=session_id,
                    block_index=idx,
                )
            )
    return blocks


def make_content_block(
    *,
    message_id: str,
    session_id: str,
    block_index: int,
    block_type: str = "text",
    text: str | None = None,
    tool_name: str | None = None,
    tool_id: str | None = None,
    tool_input: str | None = None,
    media_type: str | None = None,
    metadata: str | None = None,
    semantic_type: str | None = None,
    tool_result_is_error: int | None = None,
    tool_result_exit_code: int | None = None,
    tool_result_outcome_unknown_reason: str | None = None,
) -> BlockRecord:
    return _content_block_record(
        message_id=message_id,
        session_id=session_id,
        block_index=block_index,
        block_type=block_type,
        text=text,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_input=tool_input,
        media_type=media_type,
        metadata=metadata,
        semantic_type=semantic_type,
        tool_result_is_error=tool_result_is_error,
        tool_result_exit_code=tool_result_exit_code,
        tool_result_outcome_unknown_reason=tool_result_outcome_unknown_reason,
    )


# =============================================================================
# STORE FUNCTIONS (moved from store.py for testing)
# =============================================================================


def store_records(
    *,
    session: SessionRecord,
    messages: list[MessageRecord],
    attachments: list[AttachmentRecord],
    conn: sqlite3.Connection | None = None,
) -> dict[str, int]:
    """Store session records through the one production archive writer.

    There is deliberately no record-level SQL route here.  A second index-tier
    writer can only seed rows the real writer could not produce --
    ``identity_source``/``native_id`` disagreement and hand-summed
    ``authored_user_*`` counters were exactly that (polylogue-ugkho) -- so
    seeding goes through ``write_parsed_session_to_archive``, the choke point
    live ingest and replay share, and every derived column on a seeded row is
    the one production derives.

    Thread-safe with write lock. Returns count of inserted/updated records.
    """
    with connection_context(conn) as db_conn, _WRITE_LOCK:
        existing = db_conn.execute(
            "SELECT lower(hex(content_hash)) AS content_hash FROM sessions WHERE origin = ? AND native_id = ?",
            (session.origin.value, session.native_id),
        ).fetchone()
        new_hash = _writer_hash(session.content_hash)
        parsed = _record_to_parsed_session(session, messages, attachments)
        write_parsed_session_to_archive(
            db_conn,
            parsed,
            content_hash=new_hash,
        )
        db_conn.commit()
        written_attachments = sum(1 for attachment in attachments if attachment.message_id is not None)
        unchanged = existing is not None and str(existing["content_hash"]) == new_hash
        return {
            "sessions": 0 if unchanged else 1,
            "messages": 0 if unchanged else len(messages),
            "attachments": written_attachments,
            "skipped_sessions": 1 if unchanged else 0,
            "skipped_messages": len(messages) if unchanged else 0,
            "skipped_attachments": len(attachments) - written_attachments,
        }


# =============================================================================
# DATABASE SETUP UTILITIES
# =============================================================================


def db_setup(workspace_env: Mapping[str, Path]) -> Path:
    """Return the archive's index database path inside the workspace.

    Seeding and reads both resolve to the configured archive root, so the
    builders write the same store the facade/CLI/MCP read.
    """
    root = workspace_env["archive_root"]
    root.mkdir(parents=True, exist_ok=True)
    return root / "index.db"


def _archive_root_for(db_path: Path) -> Path:
    """Resolve the archive root that contains ``db_path``.

    ``db_path`` is the index database file (``.../index.db``); the archive
    root is its parent directory, where the rest of the store lives.
    """
    return db_path.parent


def _record_to_parsed_session(
    session: SessionRecord,
    messages: list[MessageRecord],
    attachments: list[AttachmentRecord],
) -> ParsedSession:
    """Convert builder records into the parser envelope the archive ingests.

    ``MessageRecord.identity_source`` selects which branch of the computed
    ``messages.message_id`` a seeded row takes, mirroring what a real export
    decides by supplying or withholding a provider message id:

    * ``"native"`` -- hand the writer a provider id, so the row is
      ``session_id:n:<native_id>`` with ``identity_source = 'native'``.
    * ``"content"`` -- hand the writer no provider id, so the row falls back to
      ``session_id:c:<content_identity>.<content_occurrence>`` with
      ``identity_source = 'content'``, exactly as an id-less export does.

    Both columns are written by ``write_parsed_session_to_archive`` alone, so a
    seeded row can never carry a native id with a ``content`` marker or the
    reverse (polylogue-ugkho).
    """

    def _provider_message_id(value: object | None) -> str | None:
        if value is None:
            return None
        text = str(value)
        prefix = f"{session.session_id}:"
        return text[len(prefix) :] if text.startswith(prefix) else text

    def _maybe_json_object(value: object) -> dict[str, object] | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str) and value:
            parsed = loads(value)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        return None

    def _blocks(message: MessageRecord) -> list[ParsedContentBlock]:
        parsed_blocks: list[ParsedContentBlock] = []
        for block in message.blocks or []:
            is_error = None if block.tool_result_is_error is None else bool(block.tool_result_is_error)
            exit_code = block.tool_result_exit_code
            unknown_reason = block.tool_result_outcome_unknown_reason
            if (
                block.type is BlockType.TOOL_RESULT
                and is_error is None
                and exit_code is None
                and block.tool_outcome is None
                and unknown_reason is None
            ):
                unknown_reason = "not_reported"
            parsed_blocks.append(
                ParsedContentBlock(
                    type=block.type,
                    text=block.text,
                    tool_name=block.tool_name,
                    tool_id=block.tool_id,
                    tool_input=_maybe_json_object(block.tool_input),
                    metadata=_maybe_json_object(block.metadata),
                    is_error=is_error,
                    exit_code=exit_code,
                    tool_outcome=block.tool_outcome,
                    outcome_unknown_reason=unknown_reason,
                )
            )
        return parsed_blocks

    def _seeded_provider_message_id(message: MessageRecord) -> str:
        if message.identity_source == CONTENT_DERIVED_IDENTITY:
            # The empty provider id is how a real id-less export reaches the
            # writer (``sources/parsers/codex.py`` emits exactly this), and
            # ``_stored_message_native_id`` maps it to a NULL ``native_id``.
            return ""
        return _provider_message_id(message.provider_message_id or message.message_id) or str(message.message_id)

    parsed_messages = [
        ParsedMessage(
            provider_message_id=_seeded_provider_message_id(message),
            role=message.role if message.role is not None else Role.USER,
            text=message.text,
            blocks=_blocks(message),
            message_type=message.message_type,
            material_origin=message.material_origin,
            parent_message_provider_id=_provider_message_id(message.parent_message_id),
            position=position,
            branch_index=message.branch_index,
            variant_index=message.branch_index,
            is_active_path=message.is_active_path,
            occurred_at_ms=(int(message.sort_key * 1000) if message.sort_key is not None else None),
            input_tokens=message.input_tokens,
            output_tokens=message.output_tokens,
            cache_read_tokens=message.cache_read_tokens,
            cache_write_tokens=message.cache_write_tokens,
            duration_ms=message.duration_ms,
            model_name=message.model_name,
        )
        for position, message in enumerate(messages)
    ]

    parsed_attachments = [
        ParsedAttachment(
            provider_attachment_id=str(attachment.attachment_native_id or attachment.attachment_id),
            message_provider_id=_provider_message_id(attachment.message_id),
            name=attachment.display_name,
            mime_type=attachment.mime_type,
            size_bytes=attachment.size_bytes,
            path=attachment.path,
            source_url=attachment.source_url,
            caption=attachment.caption,
        )
        for attachment in attachments
    ]
    working_directories_raw = session.working_directories_json
    parsed_wds = loads(working_directories_raw) if isinstance(working_directories_raw, str) else None
    working_directories = [item for item in parsed_wds if isinstance(item, str)] if isinstance(parsed_wds, list) else []

    return ParsedSession(
        source_name=provider_from_origin(session.origin),
        provider_session_id=session.native_id,
        title=session.title,
        # A real parser that sets a title always sets title_source alongside
        # it (assembly_codex.py, assembly_gemini.py, etc.) -- write.py's
        # session upsert treats title_source as the sole gate for "is this a
        # real title" (archive_tiers/archive.py's has_real_title check,
        # polylogue-cijx.4 decision 3), so a builder-set title with no
        # title_source silently degrades to the structural "N msgs" fallback
        # at read time. Mirror real-parser provenance here rather than
        # leaving every test-built session's explicit title invisible to
        # that gate.
        title_source=TitleSource.ORIGIN if session.title else None,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=parsed_messages,
        attachments=parsed_attachments,
        parent_session_provider_id=(str(session.parent_session_id) if session.parent_session_id is not None else None),
        branch_type=session.branch_type,
        reported_duration_ms=None,
        working_directories=working_directories,
        git_branch=session.git_branch,
        git_repository_url=session.git_repository_url,
        provider_project_ref=session.provider_project_ref,
    )


def _writer_hash(value: object) -> str:
    text = str(value)
    try:
        raw = bytes.fromhex(text)
    except ValueError:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text if len(raw) == 32 else hashlib.sha256(text.encode("utf-8")).hexdigest()


async def save_current_archive_records(
    repository: Any,
    *,
    session: SessionRecord,
    messages: list[MessageRecord],
    attachments: list[AttachmentRecord],
) -> dict[str, int]:
    """Seed current archive rows through the parsed-session writer."""

    parsed = _record_to_parsed_session(session, messages, attachments)
    return await asyncio.to_thread(
        write_session_counts_sync,
        repository.backend.db_path,
        parsed,
        content_hash=_writer_hash(session.content_hash),
    )


async def save_session_to_archive(
    backend: Any,
    *,
    session: SessionRecord,
    messages: Sequence[MessageRecord] = (),
    attachments: Sequence[AttachmentRecord] = (),
) -> dict[str, int]:
    """Seed a session into a SQLiteBackend through the live archive writer.

    Backend-based twin of :func:`save_current_archive_records`. Wraps the
    backend in a repository so population goes through the one production
    writer (``write_parsed_session_to_archive``) rather than any record-level
    backend write path. Content blocks must be attached to their
    ``MessageRecord.content_blocks`` (no separate block-write step exists).

    ``raw_id`` is not propagated by the index-only parsed-session fixture seam,
    so a follow-up UPDATE keyed on ``(origin, native_id)`` patches the column
    when the session record carries one.
    """
    from polylogue.storage.repository import SessionRepository

    result = await save_current_archive_records(
        SessionRepository(backend=backend),
        session=session,
        messages=list(messages),
        attachments=list(attachments),
    )

    if session.raw_id is not None:
        async with backend.connection() as conn:
            await conn.execute(
                "UPDATE sessions SET raw_id = ? WHERE origin = ? AND native_id = ?",
                (session.raw_id, session.origin.value, session.native_id),
            )
            await conn.commit()

    return result


# =============================================================================
# MESSAGE/SESSION BUILDERS (Fluent API)
# =============================================================================


class SessionBuilder:
    """Fluent builder for creating sessions in test databases."""

    def __init__(self, db_path: Path, session_id: str) -> None:
        self.db_path = db_path
        now = datetime.now(timezone.utc).isoformat()
        self.conv = SessionRecord(
            session_id=_session_id(session_id),
            native_id=f"ext-{session_id}",
            origin=_origin_value("test"),
            title="Test Session",
            created_at=now,
            updated_at=now,
            sort_key=_timestamp_sort_key(now),
            content_hash=_content_hash(f"session-builder:{session_id}"),
        )
        self.messages: list[MessageRecord] = []
        self.attachments: list[AttachmentRecord] = []

    def title(self, title: str | None) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"title": title})
        return self

    def provider(self, provider: str) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"origin": _origin_value(provider)})
        return self

    def created_at(self, created_at: str) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"created_at": created_at})
        return self

    def updated_at(self, updated_at: str) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"updated_at": updated_at, "sort_key": _timestamp_sort_key(updated_at)})
        return self

    def metadata(self, metadata: JSONRecord | None) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"metadata": metadata})
        return self

    def working_directories(self, paths: list[str]) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"working_directories_json": dumps(paths)})
        return self

    def git_branch(self, branch: str | None) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"git_branch": branch})
        return self

    def git_repository_url(self, repository_url: str | None) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"git_repository_url": repository_url})
        return self

    def provider_project_ref(self, project_ref: str | None) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"provider_project_ref": project_ref})
        return self

    def parent_session(self, parent_id: str) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"parent_session_id": _session_id(parent_id)})
        return self

    def branch_type(self, branch_type: str) -> SessionBuilder:
        self.conv = self.conv.model_copy(update={"branch_type": BranchType(branch_type)})
        return self

    def add_message(
        self,
        message_id: str | None = None,
        role: str | None = "user",
        text: str = "Test message",
        timestamp: str | None | _AutoTimestampSentinel = _AUTO_TIMESTAMP,
        **kwargs: object,
    ) -> SessionBuilder:
        msg_id = f"m{len(self.messages) + 1}" if message_id is None else message_id
        ts = _resolve_timestamp(timestamp)

        existing_blocks = _normalize_content_blocks(
            raw_blocks=kwargs.pop("blocks", []),
            message_id=msg_id,
            session_id=str(self.conv.session_id),
        )
        all_blocks = existing_blocks

        block_types = {blk.type for blk in all_blocks}
        role_value = None if role is None else Role.normalize(role)
        word_count = len(text.split()) if text.strip() else 0
        has_tool_use = (
            1 if (block_types & {BlockType.TOOL_USE, BlockType.TOOL_RESULT}) or role_value is Role.TOOL else 0
        )
        has_thinking = 1 if BlockType.THINKING in block_types else 0
        default_sort_key = _timestamp_sort_key(ts) if ts is not None else None
        default_content_hash = uuid4().hex[:16]

        payload: RecordPayload = {
            "message_id": _message_id(msg_id),
            "session_id": self.conv.session_id,
            "role": role_value,
            "text": text,
            "sort_key": _coerce_sort_key(
                kwargs.pop("sort_key", default_sort_key),
                default_sort_key,
            ),
            "content_hash": _coerce_content_hash(
                kwargs.pop("content_hash", default_content_hash), default_content_hash
            ),
            "blocks": all_blocks,
            "word_count": _coerce_int(kwargs.pop("word_count", word_count), word_count),
            "has_tool_use": _coerce_int(kwargs.pop("has_tool_use", has_tool_use), has_tool_use),
            "has_thinking": _coerce_int(kwargs.pop("has_thinking", has_thinking), has_thinking),
            # This builder synthesizes a provider message id, so the row it
            # seeds takes the native branch. Declared rather than implied, so
            # a caller can ask for CONTENT_DERIVED_IDENTITY instead and get a
            # row with no native_id at all.
            "identity_source": NATIVE_IDENTITY,
        }
        payload.update(kwargs)
        msg = MessageRecord.model_validate(payload)
        self.messages.append(msg)
        return self

    def add_attachment(
        self,
        attachment_id: str | None = None,
        message_id: str | None | _AutoMessageIdSentinel = _AUTO_MESSAGE_ID,
        mime_type: str = "application/octet-stream",
        size_bytes: int = 1024,
        path: str | None = None,
        display_name: str | None = None,
    ) -> SessionBuilder:
        att_id = f"att{len(self.attachments) + 1}" if attachment_id is None else attachment_id
        resolved_message_id = _resolve_attachment_message_id(value=message_id, messages=self.messages)
        att = AttachmentRecord(
            attachment_id=_attachment_id(att_id),
            session_id=self.conv.session_id,
            message_id=None if resolved_message_id is None else _message_id(resolved_message_id),
            mime_type=mime_type,
            size_bytes=size_bytes,
            path=path,
            display_name=display_name,
            attachment_native_id=att_id,
        )
        self.attachments.append(att)
        return self

    def save(self) -> SessionRecord:
        parsed = _record_to_parsed_session(self.conv, self.messages, self.attachments)
        with _WRITE_LOCK, open_connection(self.db_path) as conn:
            write_parsed_session_to_archive(
                conn,
                parsed,
                content_hash=_writer_hash(self.conv.content_hash),
            )
        return self.conv

    def native_session_id(self) -> str:
        """The archive's deterministic session id for the built session."""
        from polylogue.core.identity_law import session_id as archive_session_id

        return archive_session_id(self.conv.origin.value, self.conv.native_id)

    async def build(self) -> Session | None:
        from polylogue.api import Polylogue

        self.save()
        root = _archive_root_for(self.db_path)
        async with Polylogue(archive_root=root, db_path=root / "index.db") as plg:
            return await plg.get_session(self.native_session_id())


# =============================================================================
# QUICK BUILDERS (For simple cases)
# =============================================================================


def make_hash(s: str) -> str:
    """Create a 16-char content hash for test data."""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def make_session(
    session_id: str = "conv1",
    source_name: str = "test",
    title: str = "Test Session",
    created_at: str | None = None,
    updated_at: str | None = None,
    **kwargs: object,
) -> SessionRecord:
    now = datetime.now(timezone.utc).isoformat()
    resolved_created_at = created_at if created_at is not None else (updated_at or now)
    resolved_updated_at = updated_at if updated_at is not None else (created_at or now)
    default_content_hash = uuid4().hex
    payload: RecordPayload = {
        "session_id": _session_id(session_id),
        "origin": _origin_value(source_name),
        "native_id": _coerce_str(
            kwargs.pop("provider_session_id", session_id),
            session_id,
        ),
        "title": title,
        "created_at": resolved_created_at,
        "updated_at": resolved_updated_at,
        "content_hash": _coerce_content_hash(kwargs.pop("content_hash", default_content_hash), default_content_hash),
    }
    payload.update(kwargs)
    return SessionRecord.model_validate(payload)


def make_message(
    message_id: str = "m1",
    session_id: str = "conv1",
    role: str = "user",
    text: str | None = "Test message",
    timestamp: str | None = None,
    **kwargs: object,
) -> MessageRecord:
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    if "provider_meta" in kwargs:
        require_json_value(kwargs["provider_meta"], context="message provider_meta")
    existing_blocks = _normalize_content_blocks(
        raw_blocks=kwargs.pop("blocks", []),
        message_id=message_id,
        session_id=session_id,
    )
    all_blocks = existing_blocks

    block_types = {blk.type for blk in all_blocks}
    role_value = Role.normalize(role)
    word_count = len(text.split()) if isinstance(text, str) and text.strip() else 0
    has_tool_use = 1 if (block_types & {BlockType.TOOL_USE, BlockType.TOOL_RESULT}) or role_value is Role.TOOL else 0
    has_thinking = 1 if BlockType.THINKING in block_types else 0
    default_sort_key = _timestamp_sort_key(ts)
    default_content_hash = uuid4().hex[:16]

    payload: RecordPayload = {
        "message_id": _message_id(message_id),
        "session_id": _session_id(session_id),
        "role": role_value,
        "text": text,
        "sort_key": _coerce_sort_key(
            kwargs.pop("sort_key", default_sort_key),
            default_sort_key,
        ),
        "content_hash": _coerce_content_hash(kwargs.pop("content_hash", default_content_hash), default_content_hash),
        "blocks": all_blocks,
        "word_count": _coerce_int(kwargs.pop("word_count", word_count), word_count),
        "has_tool_use": _coerce_int(kwargs.pop("has_tool_use", has_tool_use), has_tool_use),
        "has_thinking": _coerce_int(kwargs.pop("has_thinking", has_thinking), has_thinking),
        # See SessionBuilder.add_message: the synthesized provider id makes
        # this the native branch unless the caller overrides.
        "identity_source": NATIVE_IDENTITY,
    }
    payload.update(kwargs)
    return MessageRecord.model_validate(payload)


def make_attachment(
    attachment_id: str = "att1",
    session_id: str = "conv1",
    message_id: str | None = None,
    mime_type: str = "application/octet-stream",
    size_bytes: int = 1024,
    name: str | None = None,
    **kwargs: object,
) -> AttachmentRecord:
    payload: RecordPayload = {
        "attachment_id": _attachment_id(attachment_id),
        "session_id": _session_id(session_id),
        "message_id": None if message_id is None else _message_id(message_id),
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "display_name": name,
        "attachment_native_id": attachment_id,
    }
    payload.update(kwargs)
    return AttachmentRecord.model_validate(payload)


def make_raw_session(
    raw_id: str = "raw1",
    source_name: str = "test",
    source_path: str = "/tmp/test.json",
    *,
    blob_size: int = 2,
    acquired_at: str | None = None,
    payload_provider: str | Provider | None = None,
    validation_status: str | ValidationStatus | None = None,
    validation_provider: str | Provider | None = None,
    validation_mode: str | ValidationMode | None = None,
    **kwargs: object,
) -> RawSessionRecord:
    timestamp = acquired_at or datetime.now(timezone.utc).isoformat()
    payload: RecordPayload = {
        "raw_id": raw_id,
        "source_name": source_name,
        "source_path": source_path,
        "blob_size": blob_size,
        "acquired_at": timestamp,
        "payload_provider": (
            payload_provider
            if isinstance(payload_provider, Provider) or payload_provider is None
            else Provider.from_string(payload_provider)
        ),
        "validation_status": (
            validation_status
            if isinstance(validation_status, ValidationStatus) or validation_status is None
            else ValidationStatus.from_string(validation_status)
        ),
        "validation_provider": (
            validation_provider
            if isinstance(validation_provider, Provider) or validation_provider is None
            else Provider.from_string(validation_provider)
        ),
        "validation_mode": (
            validation_mode
            if isinstance(validation_mode, ValidationMode) or validation_mode is None
            else ValidationMode.from_string(validation_mode)
        ),
    }
    payload.update(kwargs)
    return RawSessionRecord.model_validate(payload)


class DbFactory:
    """Low-ceremony DB seeder built on top of SessionBuilder."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        id: str | None = None,
        provider: str = "test",
        title: str = "Test Session",
        messages: list[JSONRecord] | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        metadata: JSONRecord | None = None,
    ) -> str:
        cid = id or str(uuid4())
        created_iso = (created_at or datetime.now(timezone.utc)).isoformat()
        updated_iso = (updated_at or datetime.now(timezone.utc)).isoformat()

        builder = (
            SessionBuilder(self.db_path, cid)
            .provider(provider)
            .title(title)
            .created_at(created_iso)
            .updated_at(updated_iso)
            .metadata(metadata)
        )

        for msg in messages or []:
            message_id = _optional_str(msg.get("id"))
            text_value = _optional_str(msg.get("text"))
            if text_value is None:
                text_value = _optional_str(msg.get("content")) or "hello"
            message_kwargs: dict[str, object] = {
                "provider_message_id": _optional_str(msg.get("provider_message_id")),
                "parent_message_id": _optional_str(msg.get("parent_message_id")),
                "branch_index": _optional_int(msg.get("branch_index")) or 0,
                "blocks": msg.get("blocks", []),
            }
            if (word_count := _optional_int(msg.get("word_count"))) is not None:
                message_kwargs["word_count"] = word_count
            if (has_tool_use := _optional_int(msg.get("has_tool_use"))) is not None:
                message_kwargs["has_tool_use"] = has_tool_use
            if (has_thinking := _optional_int(msg.get("has_thinking"))) is not None:
                message_kwargs["has_thinking"] = has_thinking

            builder.add_message(
                message_id=message_id,
                role=_optional_str(msg.get("role")) or "user",
                text=text_value,
                timestamp=_coerce_builder_timestamp(msg.get("timestamp", _AUTO_TIMESTAMP)),
                **message_kwargs,
            )

            attachments = msg.get("attachments")
            if not isinstance(attachments, list):
                continue
            for raw_attachment in attachments:
                if not isinstance(raw_attachment, Mapping):
                    continue
                builder.add_attachment(
                    attachment_id=_optional_str(raw_attachment.get("id")),
                    message_id=message_id if message_id is not None else _AUTO_MESSAGE_ID,
                    mime_type=_optional_str(raw_attachment.get("mime_type")) or "application/octet-stream",
                    size_bytes=_optional_int(raw_attachment.get("size_bytes")) or 1024,
                    path=_optional_str(raw_attachment.get("path")),
                    display_name=(
                        _optional_str(raw_attachment.get("name"))
                        or _optional_str(raw_attachment.get("title"))
                        or _optional_str(raw_attachment.get("display_name"))
                    ),
                )

        builder.save()
        return cid

    def mark_as_phantom_debris(self, native_id: str, *, provider: str = "test") -> str:
        """Attach an ``agent-*.meta.json``-shaped phantom raw artifact to an
        already-created session and link it via ``sessions.raw_id``.

        A session created via :meth:`create_session` with no raw content at
        all has ``raw_id IS NULL``; this attaches the phantom
        ``agent-*.meta.json`` sidecar shape a record-shape classifier
        positively refuses, so a caller that wants an "empty" session to
        carry positively-failing raw evidence must call this after
        :meth:`create_session`.

        Resolves the blob store from ``self.db_path``'s own parent directory
        (this factory's archive root), never the ambient
        ``POLYLOGUE_ARCHIVE_ROOT``/``blob_store_root()`` config -- a caller
        that seeds a ``DbFactory`` pointed at an archive root distinct from
        the ambient one (e.g. verifying config-supplied paths win over
        ambient defaults) must have the phantom blob land in the same
        archive the classifier will actually read back from.
        """
        from polylogue.storage.blob_store import BlobStore

        # SessionBuilder.__init__ always stores native_id as f"ext-{session_id}"
        # (the "id" callers pass to create_session), so the lookup below must
        # match that same transform, not the bare caller-supplied id.
        stored_native_id = f"ext-{native_id}"
        origin = _origin_value(provider)
        archive_root = self.db_path.parent
        source_db = archive_root / "source.db"
        store = BlobStore(archive_root / "blob")
        raw_id, blob_size = store.write_from_bytes(
            f'{{"agentType":"general-purpose","for":"{stored_native_id}"}}'.encode()
        )

        with sqlite3.connect(source_db) as source_conn:
            source_conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
                ) VALUES (?, ?, ?, ?, 0, ?, ?, 1)
                """,
                (
                    raw_id,
                    origin.value,
                    stored_native_id,
                    f"agent-{stored_native_id}.meta.json",
                    bytes.fromhex(raw_id),
                    blob_size,
                ),
            )
            source_conn.commit()

        with sqlite3.connect(self.db_path) as index_conn:
            cursor = index_conn.execute(
                "UPDATE sessions SET raw_id = ? WHERE native_id = ? AND origin = ?",
                (raw_id, stored_native_id, origin.value),
            )
            if cursor.rowcount != 1:
                raise AssertionError(
                    f"mark_as_phantom_debris: expected exactly one session row for "
                    f"native_id={stored_native_id!r} origin={origin.value!r}, updated {cursor.rowcount}"
                )
            index_conn.commit()
        return raw_id


def materialize_session_insights(
    db_path: Path,
    *,
    session_ids: Sequence[str] | None = None,
    progress_callback: Any | None = None,
) -> Any:
    """Materialize session insights through the shared production materializer.

    ``Polylogue.rebuild_insights`` deliberately refuses in-process execution
    (:class:`InsightMaintenanceRequiresDaemonError`): a rebuild sweep is a
    sealed, page-bounded machine owned by ``polylogued run``.  Tests that
    assert *what the materializer produces* — not *who may authorize a sweep* —
    call the same function both sanctioned owners ultimately reach:
    ``InsightsRebuildActuator.apply`` (the plan builder) and the daemon's
    per-session publication path
    (``storage/derived/session/derivation.py``) both call
    ``rebuild_session_insights_sync``.

    Authorization-route coverage lives in
    ``tests/unit/api/test_operation_executor_routes.py`` instead.
    """

    from polylogue.storage.derived.session.rebuild import rebuild_session_insights_sync
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(db_path) as conn:
        return rebuild_session_insights_sync(
            conn,
            session_ids=None if session_ids is None else list(session_ids),
            progress_callback=progress_callback,
        )
