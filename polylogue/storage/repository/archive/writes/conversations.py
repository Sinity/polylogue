"""Conversation write helpers for the repository mixin.

Hash boundary
------------
The content hash on ``ConversationRecord`` represents the canonical identity of an
imported conversation.  It is computed from these fields (see
``_conversation_hash_payload`` in ``pipeline/ids.py``):

    Inside the hash
    ~~~~~~~~~~~~~~~
    - conversation: title, created_at, updated_at
    - messages: id, role, text, timestamp, content_blocks
    - content_blocks: type, text, tool_name, tool_id, tool_input, media_type
    - attachments: id, message_id, name, mime_type, size_bytes
    - provider_events: event_index, event_type, timestamp, source_message_id, payload

    Outside the hash (intentionally excluded)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - user metadata (tags, summaries, notes) — editable without triggering re-import
    - provider_meta (source_name, cwd, git context) — operational metadata
    - action_events — derived from messages + content_blocks at write time by
      ``build_action_event_records()``; they carry no independent semantics

When the content hash matches the stored row, message, attachment, content_block,
and action_event writes are skipped (idempotency).  Provider events are always
replaced when explicitly supplied, regardless of hash match — they carry
supplementary indexing-level data that may be updated without changing the
semantic content.

``save_via_backend()`` validates that the row graph passed by the caller agrees
with ``ConversationRecord.content_hash`` by re-deriving the hash from the
storage records and comparing.  A mismatch is logged as a warning and forces a
full write instead of the idempotency skip.
"""

from __future__ import annotations

import builtins
from datetime import datetime, timezone

from polylogue.archive.conversation.models import Conversation
from polylogue.archive.provider.events import ProviderEvent
from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONValue, json_document, loads
from polylogue.pipeline.ids import _content_block_payload, _conversation_hash_payload, _normalize_for_hash
from polylogue.storage.action_events.rows import attach_blocks_to_messages, build_action_event_records
from polylogue.storage.conversation_replacement import (
    recount_and_prune_attachments_async,
    replace_conversation_runtime_state_async,
)
from polylogue.storage.insights.session.refresh import (
    delete_session_insights_for_conversation_async,
    refresh_thread_after_conversation_delete_async,
)
from polylogue.storage.insights.session.threads import thread_root_id_async
from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.runtime import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    ProviderEventRecord,
)
from polylogue.storage.search.cache import invalidate_search_cache
from polylogue.storage.sqlite.queries import action_events as action_events_q
from polylogue.storage.sqlite.queries import attachments as attachments_q
from polylogue.storage.sqlite.queries import conversations as conversations_q
from polylogue.storage.sqlite.queries import messages as messages_q
from polylogue.storage.sqlite.queries import provider_events as provider_events_q
from polylogue.storage.sqlite.queries import stats as stats_q
from polylogue.storage.sqlite.queries.conversations_identity import repoint_user_state_by_identity


def provider_conversation_id(conversation_id: str, provider: str | None) -> str:
    """Strip only the canonical provider prefix from conversation IDs."""
    if not provider:
        return conversation_id
    prefix = f"{provider}:"
    return conversation_id[len(prefix) :] if conversation_id.startswith(prefix) else conversation_id


def _content_hash_from_metadata_or_domain(conversation: Conversation, metadata: dict[str, object]) -> str:
    """Compute the conversation content hash through the canonical ids.py path.

    Previously had a divergent hash computation that differed from
    polylogue.pipeline.ids.conversation_content_hash().  Now builds
    hash-form message and attachment dicts from the domain Conversation
    model and delegates to the shared _conversation_hash_payload helper.
    """
    existing = metadata.get("content_hash")
    if isinstance(existing, str) and existing.strip():
        return existing

    messages_payload: list[dict[str, JSONValue]] = []
    for message in conversation.messages:
        msg_entry: dict[str, JSONValue] = {
            "id": str(message.id),
            "role": str(message.role),
            "text": _normalize_for_hash(message.text),
            "timestamp": _normalize_for_hash(message.timestamp.isoformat() if message.timestamp else None),
        }
        if message.content_blocks:
            msg_entry["content_blocks"] = [
                _content_block_payload(b)  # type: ignore[arg-type]
                for b in message.content_blocks
            ]
        messages_payload.append(msg_entry)

    attachments_payload: list[dict[str, JSONValue]] = []
    for message in conversation.messages:
        for attachment in message.attachments:
            attachments_payload.append(
                {
                    "id": _normalize_for_hash(attachment.id),
                    "message_id": _normalize_for_hash(message.id),
                    "name": _normalize_for_hash(attachment.name),
                    "mime_type": _normalize_for_hash(attachment.mime_type),
                    "size_bytes": _normalize_for_hash(attachment.size_bytes),
                }
            )

    return hash_payload(
        _conversation_hash_payload(
            title=conversation.title,
            created_at=conversation.created_at.isoformat() if conversation.created_at else None,
            updated_at=conversation.updated_at.isoformat() if conversation.updated_at else None,
            messages=messages_payload,
            attachments=attachments_payload,
            provider_events=[
                {
                    "event_index": event.event_index,
                    "event_type": _normalize_for_hash(event.event_type),
                    "timestamp": _normalize_for_hash(event.timestamp.isoformat() if event.timestamp else None),
                    "source_message_id": _normalize_for_hash(event.source_message_id),
                    "payload": hash_payload(event.payload),
                }
                for event in conversation.provider_events
            ],
        )
    )


def conversation_to_record(conversation: Conversation) -> ConversationRecord:
    from polylogue.types import ContentHash, ConversationId

    created_at_str = conversation.created_at.isoformat() if conversation.created_at else None
    updated_at_str = conversation.updated_at.isoformat() if conversation.updated_at else (created_at_str or None)
    metadata = json_document(conversation.metadata)
    metadata_record: dict[str, object] = {}
    metadata_record.update(metadata)
    provider_meta: dict[str, object] = {}
    provider_meta.update(json_document(metadata.get("provider_meta")))

    raw_source = provider_meta.get("source") if provider_meta else None
    source_name_val = raw_source if isinstance(raw_source, str) else ""

    return ConversationRecord(
        conversation_id=ConversationId(str(conversation.id)),
        provider_name=conversation.provider,
        provider_conversation_id=provider_conversation_id(
            conversation_id=str(conversation.id),
            provider=conversation.provider,
        ),
        title=conversation.title or "",
        created_at=created_at_str,
        updated_at=updated_at_str,
        content_hash=ContentHash(_content_hash_from_metadata_or_domain(conversation, metadata_record)),
        provider_meta=provider_meta,
        metadata=metadata_record,
        source_name=source_name_val,
    )


def provider_event_to_record(event: ProviderEvent) -> ProviderEventRecord:
    return ProviderEventRecord(
        event_id=event.id,
        conversation_id=event.conversation_id,
        provider_name=str(event.provider),
        event_index=event.event_index,
        event_type=event.event_type,
        timestamp=event.timestamp.isoformat() if event.timestamp else None,
        sort_key=event.sort_key,
        payload=event.payload,
        source_message_id=event.source_message_id,
        raw_id=event.raw_id,
        materializer_version=event.materializer_version,
    )


def _content_block_record_to_hash_payload(block: ContentBlockRecord) -> dict[str, JSONValue]:
    """Build a hash-stable payload for a ``ContentBlockRecord``.

    Mirrors ``_content_block_payload()`` in ``pipeline/ids.py`` but accepts the
    storage record type instead of the parsed source type.
    """
    payload: dict[str, JSONValue] = {
        "type": str(block.type),
        "text": _normalize_for_hash(block.text),
    }
    if block.tool_name:
        payload["tool_name"] = _normalize_for_hash(block.tool_name)
    if block.tool_id:
        payload["tool_id"] = _normalize_for_hash(block.tool_id)
    if block.tool_input is not None:
        # ContentBlockRecord stores tool_input as a JSON string, but the canonical
        # hash hashes the dict representation.  Parse and re-hash for consistency.
        try:
            parsed = loads(block.tool_input)
            if isinstance(parsed, dict):
                payload["tool_input"] = hash_payload(parsed)
            else:
                payload["tool_input"] = hash_payload(block.tool_input)
        except Exception:
            payload["tool_input"] = hash_payload(block.tool_input)
    # #1240: media_type is carried inside block.metadata (JSON) for image/document
    # blocks instead of a dedicated column. Re-extract it so the row-graph hash
    # matches the parser-side hash computed in pipeline/ids.py.
    if block.metadata:
        try:
            parsed_meta = loads(block.metadata)
        except Exception:
            parsed_meta = None
        if isinstance(parsed_meta, dict):
            media_type = parsed_meta.get("media_type")
            if isinstance(media_type, str) and media_type:
                payload["media_type"] = _normalize_for_hash(media_type)
    return payload


def _compute_hash_from_row_graph(
    *,
    conversation: ConversationRecord,
    messages: builtins.list[MessageRecord],
    attachments: builtins.list[AttachmentRecord],
    provider_events: builtins.list[ProviderEventRecord] | None = None,
) -> str:
    """Re-derive the conversation content hash from the storage record graph.

    Uses the same canonical ``_conversation_hash_payload`` helper as
    ``pipeline/ids.py`` but sources field values from storage records instead
    of the domain ``Conversation`` model.  This function exists so that
    ``save_via_backend`` can validate that the row graph the caller supplied
    actually agrees with ``ConversationRecord.content_hash``.

    Returns the full SHA-256 hex digest.
    """
    messages_payload: list[dict[str, JSONValue]] = []
    for msg in messages:
        timestamp_str = None
        if msg.sort_key is not None:
            try:
                dt = datetime.fromtimestamp(msg.sort_key, tz=timezone.utc)
                timestamp_str = dt.isoformat()
            except (ValueError, OverflowError, OSError):
                timestamp_str = str(msg.sort_key)

        msg_entry: dict[str, JSONValue] = {
            "id": str(msg.message_id),
            "role": _normalize_for_hash(str(msg.role) if msg.role else None),
            "text": _normalize_for_hash(msg.text),
            "timestamp": _normalize_for_hash(timestamp_str),
        }
        if msg.content_blocks:
            msg_entry["content_blocks"] = [_content_block_record_to_hash_payload(b) for b in msg.content_blocks]
        messages_payload.append(msg_entry)

    attachments_payload: list[dict[str, JSONValue]] = []
    for att in attachments:
        att_name: str | None = None
        if att.provider_meta and isinstance(att.provider_meta, dict):
            raw_name = att.provider_meta.get("name")
            if isinstance(raw_name, str):
                att_name = raw_name
        if att_name is None:
            att_name = att.attachment_id

        attachments_payload.append(
            {
                "id": _normalize_for_hash(att.attachment_id),
                "message_id": _normalize_for_hash(att.message_id),
                "name": _normalize_for_hash(att_name),
                "mime_type": _normalize_for_hash(att.mime_type),
                "size_bytes": _normalize_for_hash(att.size_bytes),
            }
        )

    provider_events_payload: list[dict[str, JSONValue]] = []
    if provider_events:
        for event in provider_events:
            provider_events_payload.append(
                {
                    "event_index": event.event_index,
                    "event_type": _normalize_for_hash(event.event_type),
                    "timestamp": _normalize_for_hash(event.timestamp),
                    "source_message_id": _normalize_for_hash(event.source_message_id),
                    "payload": hash_payload(event.payload),
                }
            )

    return hash_payload(
        _conversation_hash_payload(
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=messages_payload,
            attachments=attachments_payload,
            provider_events=provider_events_payload,
        )
    )


async def save_via_backend(
    backend: RepositoryBackendProtocol,
    conversation: ConversationRecord,
    messages: builtins.list[MessageRecord],
    attachments: builtins.list[AttachmentRecord],
    content_blocks: builtins.list[ContentBlockRecord] | None = None,
    provider_events: builtins.list[ProviderEventRecord] | None = None,
) -> dict[str, int]:
    import time as _time

    t_start = _time.perf_counter()
    timings: dict[str, float] = {}
    counts: dict[str, int] = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "provider_events": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
        "skipped_provider_events": 0,
    }
    should_replace_provider_events = provider_events is not None
    provider_events_for_write = provider_events or []

    async with backend.transaction(), backend.connection() as conn:
        t0 = _time.perf_counter()
        existing_hash: str | None = None
        cursor = await conn.execute(
            "SELECT content_hash FROM conversations WHERE conversation_id = ?",
            (conversation.conversation_id,),
        )
        row = await cursor.fetchone()
        if row:
            existing_hash = row["content_hash"]
        timings["hash_check"] = _time.perf_counter() - t0

        content_unchanged = existing_hash is not None and existing_hash == conversation.content_hash

        # ------------------------------------------------------------------
        # Cross-validate: does the row graph the caller supplied actually
        # agree with ConversationRecord.content_hash?  A mismatch means the
        # caller built records that diverged from what was hashed — the hash
        # is no longer a trustworthy signal for the idempotency skip.
        # ------------------------------------------------------------------
        row_graph_hash = _compute_hash_from_row_graph(
            conversation=conversation,
            messages=messages,
            attachments=attachments,
            provider_events=provider_events_for_write if should_replace_provider_events else None,
        )
        if row_graph_hash != conversation.content_hash:
            from polylogue.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(
                "save_via_backend_hash_mismatch",
                conversation_id=str(conversation.conversation_id),
                record_hash=str(conversation.content_hash),
                row_graph_hash=row_graph_hash,
                content_unchanged=content_unchanged,
            )
            if content_unchanged:
                # The stored DB hash agrees with our (stale) record hash, but
                # the row graph we are about to write disagrees.  The record
                # hash is no longer trustworthy — force a full write so that
                # changed rows are not silently skipped.
                content_unchanged = False

        t0 = _time.perf_counter()
        await conversations_q.save_conversation_record(conn, conversation, backend.transaction_depth)
        timings["save_conv"] = _time.perf_counter() - t0

        if content_unchanged:
            counts["skipped_conversations"] = 1
            counts["skipped_messages"] = len(messages)
            counts["skipped_attachments"] = len(attachments)
            if should_replace_provider_events:
                t0 = _time.perf_counter()
                await provider_events_q.replace_provider_events(
                    conn,
                    conversation.conversation_id,
                    provider_events_for_write,
                    backend.transaction_depth,
                )
                counts["provider_events"] = len(provider_events_for_write)
                timings["provider_events"] = _time.perf_counter() - t0
        else:
            counts["conversations"] = 1
            t0 = _time.perf_counter()
            affected_attachment_ids = await replace_conversation_runtime_state_async(conn, conversation.conversation_id)
            timings["replace_runtime"] = _time.perf_counter() - t0

            if messages:
                t0 = _time.perf_counter()
                await messages_q.save_messages(conn, messages, backend.transaction_depth)
                timings["save_msgs"] = _time.perf_counter() - t0
                counts["messages"] = len(messages)

                t0 = _time.perf_counter()
                await stats_q.upsert_conversation_stats(
                    conn,
                    conversation.conversation_id,
                    conversation.provider_name,
                    messages,
                    backend.transaction_depth,
                )
                timings["upsert_stats"] = _time.perf_counter() - t0

            t0 = _time.perf_counter()
            all_blocks: builtins.list[ContentBlockRecord] = list(content_blocks or [])
            for message in messages:
                all_blocks.extend(message.content_blocks)
            if all_blocks:
                await attachments_q.save_content_blocks(conn, all_blocks, backend.transaction_depth)
            timings["save_blocks"] = _time.perf_counter() - t0

            t0 = _time.perf_counter()
            action_messages = attach_blocks_to_messages(messages, all_blocks)
            action_records = build_action_event_records(conversation, action_messages)
            await action_events_q.replace_action_events(
                conn,
                conversation.conversation_id,
                action_records,
                backend.transaction_depth,
            )
            timings["action_events"] = _time.perf_counter() - t0

            t0 = _time.perf_counter()
            if should_replace_provider_events:
                await provider_events_q.replace_provider_events(
                    conn,
                    conversation.conversation_id,
                    provider_events_for_write,
                    backend.transaction_depth,
                )
                counts["provider_events"] = len(provider_events_for_write)
            timings["provider_events"] = _time.perf_counter() - t0

            t0 = _time.perf_counter()
            new_attachment_ids: set[str] = {str(att.attachment_id) for att in attachments}
            affected_attachment_ids |= new_attachment_ids
            if attachments:
                await attachments_q.save_attachments(conn, attachments, backend.transaction_depth)
                counts["attachments"] = len(attachments)
            await recount_and_prune_attachments_async(conn, affected_attachment_ids)
            timings["attachments"] = _time.perf_counter() - t0

        # Rebind any orphaned marks/annotations whose identity_key matches the
        # freshly-written conversation. See #1114 — identity_key survives reset
        # because conversation_id is deterministic across reimport.
        t0 = _time.perf_counter()
        await repoint_user_state_by_identity(conn, conversation.conversation_id)
        timings["repoint_user_state"] = _time.perf_counter() - t0

    total = _time.perf_counter() - t_start
    if total > 2.0:
        from polylogue.logging import get_logger

        get_logger(__name__).info(
            "slow_save",
            total_s=round(total, 2),
            msgs=len(messages),
            blocks=counts.get("messages", 0),
            cid=str(conversation.conversation_id)[:20],
            **{k: round(v, 3) for k, v in timings.items()},
        )

    invalidate_search_cache()
    return counts


async def delete_conversation_via_backend(
    backend: RepositoryBackendProtocol,
    conversation_id: str,
) -> bool:
    from polylogue.storage.sqlite.queries import conversations as conversations_q

    async with backend.transaction(), backend.connection() as conn:
        parent_row = await (
            await conn.execute(
                "SELECT parent_conversation_id FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
        ).fetchone()
        child_rows = await (
            await conn.execute(
                "SELECT conversation_id FROM conversations WHERE parent_conversation_id = ?",
                (conversation_id,),
            )
        ).fetchall()
        existing_row = await (
            await conn.execute(
                "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
        ).fetchone()
        deleted = False
        if existing_row is not None:
            await delete_session_insights_for_conversation_async(
                conn,
                conversation_id,
                transaction_depth=backend.transaction_depth,
            )
            deleted = await conversations_q.delete_conversation_sql(
                conn,
                conversation_id,
                backend.transaction_depth,
            )
        if deleted:
            affected_seeds = {str(row["conversation_id"]) for row in child_rows}
            if parent_row is not None and parent_row["parent_conversation_id"] is not None:
                affected_seeds.add(str(parent_row["parent_conversation_id"]))
            affected_roots: set[str] = set()
            for seed in affected_seeds:
                root_id = await thread_root_id_async(conn, seed)
                if root_id is not None:
                    affected_roots.add(root_id)
            for root_id in affected_roots:
                await refresh_thread_after_conversation_delete_async(
                    conn,
                    root_id,
                    transaction_depth=backend.transaction_depth,
                )
    if deleted:
        invalidate_search_cache()
    return deleted
