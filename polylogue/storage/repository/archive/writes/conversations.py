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

``save_via_backend()`` trusts ``ConversationRecord.content_hash`` directly: it
compares the stored DB hash against the record hash to decide the idempotency
skip, exactly as the bulk daemon ingest path does
(``pipeline/services/ingest_batch/_core.py:_check_content_unchanged``).  There is
intentionally no re-derivation of the hash from the storage row graph (#1747).
The row graph is a lossy projection of the parsed conversation and the fallback
identity rules (``msg-N`` provider-message ids, attachment digest provenance)
are not reconstructible from the stored records without duplicating — and
forever tracking — the canonical ``pipeline/ids.py`` logic.  An earlier
cross-validator attempted that reconstruction; it diverged on essentially every
conversation (full ``message_id`` instead of the bare provider id, a
``sort_key`` epoch instead of the original ISO timestamp, no attachment digest),
producing permanent ``save_via_backend_hash_mismatch`` warning noise and
defeating the idempotency skip on the single-conversation (CLI/API) path.
"""

from __future__ import annotations

import builtins
from datetime import datetime, timezone

from polylogue.archive.conversation.models import Conversation
from polylogue.archive.provider.events import ProviderEvent
from polylogue.archive.topology.edge import TopologyEdgeRecord
from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONValue, json_document
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
from polylogue.storage.sqlite.queries import topology_edges as topology_edges_q
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
        source_name=str(event.provider),
        event_index=event.event_index,
        event_type=event.event_type,
        timestamp=event.timestamp.isoformat() if event.timestamp else None,
        sort_key=event.sort_key,
        payload=event.payload,
        source_message_id=event.source_message_id,
        raw_id=event.raw_id,
        materializer_version=event.materializer_version,
    )


async def save_via_backend(
    backend: RepositoryBackendProtocol,
    conversation: ConversationRecord,
    messages: builtins.list[MessageRecord],
    attachments: builtins.list[AttachmentRecord],
    content_blocks: builtins.list[ContentBlockRecord] | None = None,
    provider_events: builtins.list[ProviderEventRecord] | None = None,
    topology_edges: builtins.list[TopologyEdgeRecord] | None = None,
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

        # Trust the canonical pipeline content hash for the idempotency skip,
        # exactly as the bulk daemon ingest path does (#1747).  The stored DB
        # hash and the record hash are both produced by
        # ``pipeline/ids.conversation_content_hashes``; when they agree the
        # parsed payload is unchanged and the row writes can be skipped.
        content_unchanged = existing_hash is not None and existing_hash == conversation.content_hash

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

            # Always upsert the aggregate stats row, even for a zero-message
            # conversation (#1747).  An empty message list yields an all-zero
            # row so ``--typed-only`` (paste_count = 0) and provider rollups
            # that count from ``conversation_stats`` stay consistent with
            # ``get_provider_conversation_counts`` instead of dropping the
            # conversation through a missing LEFT JOIN row.
            t0 = _time.perf_counter()
            await stats_q.upsert_conversation_stats(
                conn,
                conversation.conversation_id,
                conversation.source_name,
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

        # Persist any topology edges asserted by this conversation, then run
        # the resolver helper so out-of-order children that were waiting on
        # this conversation's native id get backfilled to status=resolved
        # (#1258 / #866 slice A).
        t0 = _time.perf_counter()
        if topology_edges:
            await topology_edges_q.upsert_topology_edges(conn, topology_edges)
        # Resolve previously-unresolved edges that point at this conversation.
        now_iso = datetime.now(timezone.utc).isoformat()
        await topology_edges_q.resolve_topology_edges_for_conversation(
            conn,
            conversation_id=str(conversation.conversation_id),
            source_name=conversation.source_name,
            provider_conversation_id=conversation.provider_conversation_id,
            resolved_at=now_iso,
        )
        # Slice B race closure (#1259): if the parent was committed *after*
        # the in-memory ``PrepareCache`` for this child was built but
        # *before* this child's save_via_backend ran, the upsert above wrote
        # the edge as ``unresolved`` even though the parent now exists. Sweep
        # the child's own unresolved edges and resolve any whose parent is
        # already in ``conversations``.
        if topology_edges:
            await topology_edges_q.resolve_unresolved_edges_for_child(
                conn,
                src_conversation_id=str(conversation.conversation_id),
                resolved_at=now_iso,
            )
        timings["topology_edges"] = _time.perf_counter() - t0

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
