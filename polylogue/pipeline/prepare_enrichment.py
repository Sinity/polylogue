"""DB-aware enrichment for transformed record bundles."""

from __future__ import annotations

from polylogue.archive.topology.edge import (
    TopologyEdgeRecord,
    TopologyEdgeStatus,
    branch_type_to_edge_type,
)
from polylogue.pipeline.ids import provider_event_id
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.prepare_models import (
    PrepareCache,
    PreparedBundle,
    RecordBundle,
    TransformResult,
)
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.archive_views import ExistingSession
from polylogue.storage.runtime import (
    AttachmentRecord,
    ContentBlockRecord,
    MessageRecord,
    ProviderEventRecord,
    SessionRecord,
)
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.types import MessageId, SessionId


def _str_from_meta(provider_meta: dict[str, object] | None, key: str) -> str | None:
    """Extract a string value from provider_meta if present."""
    if not provider_meta:
        return None
    value = provider_meta.get(key)
    return value if isinstance(value, str) else None


def enrich_bundle_from_db(
    convo: ParsedSession,
    source_name: str,
    transform: TransformResult,
    cache: PrepareCache,
    *,
    raw_id: str | None = None,
) -> PreparedBundle:
    candidate_cid = transform.candidate_cid
    content_hash = transform.content_hash

    existing = cache.existing.get(candidate_cid)
    if existing:
        cid = SessionId(existing.session_id)
        changed = existing.content_hash != content_hash
    else:
        cid = candidate_cid
        changed = False

    # Resolve parent fast-path (sessions.parent_session_id) AND
    # always record an explicit topology edge when the parser asserted a
    # parent reference, so out-of-order ingest does not silently lose the
    # edge (#1258). The fast-path semantics are preserved: parent_session_id
    # is set only when the parent is in cache; the topology edge is written
    # unconditionally.
    parent_session_id: SessionId | None = None
    topology_edges: list[TopologyEdgeRecord] = []
    if convo.parent_session_provider_id:
        candidate_parent = make_session_id(convo.source_name, convo.parent_session_provider_id)
        parent_known = candidate_parent in cache.known_ids
        if parent_known:
            parent_session_id = candidate_parent
        edge_type = branch_type_to_edge_type(convo.branch_type)
        topology_edges.append(
            TopologyEdgeRecord(
                src_session_id=cid,
                dst_provider_native_id=convo.parent_session_provider_id,
                dst_provider_name=convo.source_name,
                edge_type=edge_type,
                resolved_dst_session_id=candidate_parent if parent_known else None,
                status=(TopologyEdgeStatus.RESOLVED if parent_known else TopologyEdgeStatus.UNRESOLVED),
                resolved_at=(transform.bundle.session.updated_at if parent_known else None),
            )
        )

    existing_message_ids = cache.message_ids.get(cid, {})
    stable_message_id_map: dict[str, MessageId] = {}
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        key = str(provider_message_id)
        stable_message_id_map[key] = existing_message_ids.get(key) or transform.message_id_map[key]

    merged_provider_meta: dict[str, object] = {"source": source_name}
    if convo.provider_meta:
        merged_provider_meta.update(convo.provider_meta)

    # Extract canonical fields from provider_meta (#864 Slice 2)
    import json as _json

    working_directories_json: str | None = None
    git_branch: str | None = None
    git_repository_url: str | None = None
    if convo.provider_meta:
        wds = convo.provider_meta.get("working_directories")
        if isinstance(wds, list):
            working_directories_json = _json.dumps(wds)
        cwd = convo.provider_meta.get("cwd")
        if isinstance(cwd, str) and working_directories_json is None:
            working_directories_json = _json.dumps([cwd])
        gb = convo.provider_meta.get("gitBranch")
        if isinstance(gb, str):
            git_branch = gb
        git_obj = convo.provider_meta.get("git")
        if isinstance(git_obj, dict):
            git_branch = git_branch or git_obj.get("branch")
            git_repository_url = git_obj.get("repository_url")

    session_record = SessionRecord(
        session_id=cid,
        provider_session_id=convo.provider_session_id,
        title=convo.title,
        created_at=transform.bundle.session.created_at,
        updated_at=transform.bundle.session.updated_at,
        sort_key=transform.bundle.session.sort_key,
        content_hash=content_hash,
        provider_meta=merged_provider_meta,
        source_name=source_name,
        working_directories_json=working_directories_json,
        git_branch=git_branch,
        git_repository_url=git_repository_url,
        parent_session_id=parent_session_id,
        branch_type=convo.branch_type,
        raw_id=raw_id,
    )

    patched_messages: list[MessageRecord] = []
    for idx, (msg_rec, msg) in enumerate(zip(transform.bundle.messages, convo.messages, strict=True), start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        mid = stable_message_id_map[str(provider_message_id)]
        parent_message_id: MessageId | None = None
        if msg.parent_message_provider_id:
            parent_message_id = stable_message_id_map.get(str(msg.parent_message_provider_id))
        patched_messages.append(
            MessageRecord(
                message_id=mid,
                session_id=cid,
                provider_message_id=msg_rec.provider_message_id,
                role=msg_rec.role,
                text=msg_rec.text,
                sort_key=msg_rec.sort_key,
                content_hash=msg_rec.content_hash,
                parent_message_id=parent_message_id,
                branch_index=msg_rec.branch_index,
                source_name=msg_rec.source_name,
                word_count=msg_rec.word_count,
                has_tool_use=msg_rec.has_tool_use,
                has_thinking=msg_rec.has_thinking,
                has_paste=msg_rec.has_paste,
                paste_boundary_state=msg_rec.paste_boundary_state,
            )
        )

    reverse_mid: dict[MessageId, MessageId] = {}
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        old = transform.message_id_map[str(provider_message_id)]
        reverse_mid[old] = stable_message_id_map[str(provider_message_id)]

    patched_blocks = [
        ContentBlockRecord(
            block_id=ContentBlockRecord.make_id(reverse_mid.get(block.message_id, block.message_id), block.block_index),
            message_id=reverse_mid.get(block.message_id, block.message_id),
            session_id=cid,
            block_index=block.block_index,
            type=block.type,
            text=block.text,
            tool_name=block.tool_name,
            tool_id=block.tool_id,
            tool_input=block.tool_input,
            metadata=block.metadata,
            semantic_type=block.semantic_type,
        )
        for block in transform.bundle.content_blocks
    ]

    patched_attachments: list[AttachmentRecord] = []
    for attachment in transform.bundle.attachments:
        att_message_id: MessageId | None = None
        if attachment.message_id is not None:
            att_message_id = reverse_mid.get(attachment.message_id, attachment.message_id)
        # #1252: typed native identifiers flow from ParsedAttachment →
        # MaterializedAttachment → AttachmentRecord without re-reading
        # provider_meta. Fall back to the JSON envelope only for AttachmentRecords
        # constructed by older code paths that have not yet adopted
        # populate the typed surface (the daemon ingest path always populates
        # the typed fields).
        patched_attachments.append(
            AttachmentRecord(
                attachment_id=attachment.attachment_id,
                session_id=cid,
                message_id=att_message_id,
                mime_type=attachment.mime_type,
                size_bytes=attachment.size_bytes,
                path=attachment.path,
                provider_meta=attachment.provider_meta,
                provider_attachment_id=attachment.provider_attachment_id,
                provider_file_id=(attachment.provider_file_id or _str_from_meta(attachment.provider_meta, "fileId")),
                provider_drive_id=(attachment.provider_drive_id or _str_from_meta(attachment.provider_meta, "driveId")),
                upload_origin=attachment.upload_origin,
            )
        )

    patched_provider_events = [
        ProviderEventRecord(
            event_id=provider_event_id(cid, event.event_index),
            session_id=cid,
            source_name=event.source_name,
            event_index=event.event_index,
            event_type=event.event_type,
            timestamp=event.timestamp,
            sort_key=event.sort_key,
            payload=event.payload,
            source_message_id=(
                reverse_mid.get(event.source_message_id, event.source_message_id)
                if event.source_message_id is not None
                else None
            ),
            raw_id=raw_id,
            materializer_version=event.materializer_version,
        )
        for event in transform.bundle.provider_events
    ]

    return PreparedBundle(
        bundle=RecordBundle(
            session=session_record,
            messages=patched_messages,
            attachments=patched_attachments,
            content_blocks=patched_blocks,
            provider_events=patched_provider_events,
            topology_edges=topology_edges,
        ),
        materialization_plan=transform.materialization_plan,
        cid=cid,
        changed=changed,
    )


async def _build_single_cache(
    backend: SQLiteBackend,
    convo: ParsedSession,
    candidate_cid: SessionId,
    _unused: object,
) -> PrepareCache:
    cache = PrepareCache()

    async with backend.connection() as conn:
        cursor = await conn.execute(
            "SELECT session_id, content_hash FROM sessions WHERE session_id = ? LIMIT 1",
            (candidate_cid,),
        )
        row = await cursor.fetchone()
    if row:
        cid = row["session_id"]
        cache.existing[cid] = ExistingSession(session_id=cid, content_hash=row["content_hash"])
        cache.known_ids.add(cid)

    if convo.parent_session_provider_id:
        candidate_parent = make_session_id(convo.source_name, convo.parent_session_provider_id)
        async with backend.connection() as conn:
            cursor = await conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (candidate_parent,),
            )
            if await cursor.fetchone():
                cache.known_ids.add(candidate_parent)

    existing_cid: SessionId | None = candidate_cid if candidate_cid in cache.known_ids else None
    if existing_cid:
        async with backend.connection() as conn:
            cursor = await conn.execute(
                "SELECT provider_message_id, message_id FROM messages "
                "WHERE session_id = ? AND provider_message_id IS NOT NULL",
                (existing_cid,),
            )
            rows = await cursor.fetchall()
        cache.message_ids[existing_cid] = {
            str(row["provider_message_id"]): MessageId(row["message_id"]) for row in rows if row["provider_message_id"]
        }

    return cache


__all__ = [
    "_build_single_cache",
    "enrich_bundle_from_db",
]
