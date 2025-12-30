from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from .assets import asset_path
from .config import Config, Profile, Source
from .db import open_connection
from .drive_ingest import iter_drive_conversations
from .index import index_status, rebuild_index, update_index_for_conversations
from .ingest import IngestBundle, ingest_bundle
from .redaction import sanitize_text
from .render import render_conversation
from .source_ingest import (
    ParsedAttachment,
    ParsedConversation,
    ParsedMessage,
    iter_source_conversations,
)
from .store import ConversationRecord, MessageRecord, AttachmentRecord, RunRecord, record_run


@dataclass
class PlanResult:
    timestamp: int
    counts: Dict[str, int]
    sources: List[str]
    cursors: Dict[str, dict]


@dataclass
class RunResult:
    run_id: str
    counts: Dict[str, int]
    drift: Dict[str, dict]
    indexed: bool
    index_error: Optional[str]
    duration_ms: int


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _maybe_redact(text: Optional[str], *, redact_store: bool) -> Optional[str]:
    if not redact_store or text is None:
        return text
    return sanitize_text(text)


def _attachment_seed(provider_name: str, attachment: ParsedAttachment) -> str:
    return "|".join(
        str(value)
        for value in [
            provider_name,
            attachment.provider_attachment_id,
            attachment.message_provider_id,
            attachment.name,
            attachment.mime_type,
            attachment.size_bytes,
            attachment.path,
        ]
        if value is not None
    )


def _attachment_content_id(
    provider_name: str,
    attachment: ParsedAttachment,
    *,
    archive_root: Path,
) -> str:
    meta = dict(attachment.provider_meta or {})
    for key in ("sha256", "digest", "hash"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value
    raw_path = attachment.path
    if isinstance(raw_path, str) and raw_path:
        path = Path(raw_path)
        if path.exists() and path.is_file():
            digest = _hash_file(path)
            meta.setdefault("sha256", digest)
            attachment.provider_meta = meta
            assets_root = archive_root / "assets"
            if assets_root in path.parents:
                target = asset_path(archive_root, digest)
                if target != path:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        path.unlink()
                    else:
                        path.replace(target)
                    attachment.path = str(target)
            return digest
    seed = _attachment_seed(provider_name, attachment)
    return _hash_text(seed)


def _lookup_existing_conversation(convo: ParsedConversation) -> Optional[str]:
    with open_connection(None) as conn:
        row = conn.execute(
            """
            SELECT content_hash
            FROM conversations
            WHERE provider_name = ? AND provider_conversation_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (convo.provider_name, convo.provider_conversation_id),
        ).fetchone()
    return row["content_hash"] if row else None


def _conversation_id(provider_id: str, content_hash: str) -> str:
    return f"{provider_id}:{content_hash}"


def _message_id(provider_message_id: str, content_hash: str) -> str:
    return f"{provider_message_id}:{content_hash}"


def _select_sources(config: Config, source_names: Optional[Sequence[str]]) -> List[Source]:
    if not source_names:
        return list(config.sources)
    name_set = set(source_names)
    return [source for source in config.sources if source.name in name_set]


def plan_sources(
    config: Config,
    *,
    profile: Optional[Profile] = None,
    ui: Optional[object] = None,
    source_names: Optional[Sequence[str]] = None,
) -> PlanResult:
    counts = {"conversations": 0, "messages": 0, "attachments": 0}
    sources = []
    cursors: Dict[str, dict] = {}
    effective_profile = profile or Profile()
    for source in _select_sources(config, source_names):
        sources.append(source.name)
        cursor_state: Dict[str, object] = {}
        if source.type == "drive":
            conversations = iter_drive_conversations(
                source=source,
                profile=effective_profile,
                archive_root=config.archive_root,
                ui=ui,
                download_assets=False,
                cursor_state=cursor_state,
            )
        else:
            conversations = iter_source_conversations(source, cursor_state=cursor_state)
        for convo in conversations:
            counts["conversations"] += 1
            counts["messages"] += len(convo.messages)
            counts["attachments"] += len(convo.attachments)
        if cursor_state:
            cursors[source.name] = cursor_state
    return PlanResult(timestamp=int(time.time()), counts=counts, sources=sources, cursors=cursors)


def _ingest_conversation(
    convo: ParsedConversation,
    source_name: str,
    *,
    redact_store: bool,
    archive_root: Path,
) -> Tuple[str, Dict[str, int], bool]:
    redacted_messages: List[ParsedMessage] = []
    for msg in convo.messages:
        redacted_messages.append(
            ParsedMessage(
                provider_message_id=msg.provider_message_id,
                role=msg.role,
                text=_maybe_redact(msg.text, redact_store=redact_store) or "",
                timestamp=msg.timestamp,
                provider_meta=msg.provider_meta,
            )
        )
    combined_text = "\n".join(msg.text for msg in redacted_messages)
    content_hash = _hash_text(combined_text)
    conversation_id = _conversation_id(convo.provider_conversation_id, content_hash)
    existing_hash = _lookup_existing_conversation(convo)
    changed = existing_hash is not None and existing_hash != content_hash

    conversation_record = ConversationRecord(
        conversation_id=conversation_id,
        provider_name=convo.provider_name,
        provider_conversation_id=convo.provider_conversation_id,
        title=_maybe_redact(convo.title, redact_store=redact_store),
        created_at=convo.created_at,
        updated_at=convo.updated_at,
        content_hash=content_hash,
        provider_meta={"source": source_name},
    )

    messages: List[MessageRecord] = []
    message_ids: Dict[str, str] = {}
    for idx, msg in enumerate(redacted_messages, start=1):
        message_hash = _hash_text(msg.text)
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        message_id = _message_id(provider_message_id, message_hash)
        message_ids[str(provider_message_id)] = message_id
        messages.append(
            MessageRecord(
                message_id=message_id,
                conversation_id=conversation_id,
                provider_message_id=provider_message_id,
                role=msg.role,
                text=msg.text,
                timestamp=msg.timestamp,
                content_hash=message_hash,
                provider_meta=msg.provider_meta,
            )
        )

    attachments: List[AttachmentRecord] = []
    for att in convo.attachments:
        attachment_id = _attachment_content_id(convo.provider_name, att, archive_root=archive_root)
        meta = dict(att.provider_meta or {})
        if att.provider_attachment_id:
            meta.setdefault("provider_id", att.provider_attachment_id)
        attachments.append(
            AttachmentRecord(
                attachment_id=attachment_id,
                conversation_id=conversation_id,
                message_id=message_ids.get(att.message_provider_id or ""),
                mime_type=att.mime_type,
                size_bytes=att.size_bytes,
                path=att.path,
                provider_meta=meta,
            )
        )

    result = ingest_bundle(
        IngestBundle(
            conversation=conversation_record,
            messages=messages,
            attachments=attachments,
        )
    )
    return conversation_id, {
        "conversations": result.conversations,
        "messages": result.messages,
        "attachments": result.attachments,
        "skipped_conversations": result.skipped_conversations,
        "skipped_messages": result.skipped_messages,
        "skipped_attachments": result.skipped_attachments,
    }, changed


def _all_conversation_ids(source_names: Optional[Sequence[str]] = None) -> List[str]:
    with open_connection(None) as conn:
        rows = conn.execute(
            "SELECT conversation_id, provider_name, provider_meta FROM conversations"
        ).fetchall()
    if not source_names:
        return [row["conversation_id"] for row in rows]
    selected: List[str] = []
    name_set = set(source_names)
    for row in rows:
        if row["provider_name"] in name_set:
            selected.append(row["conversation_id"])
            continue
        meta = row["provider_meta"]
        if not meta:
            continue
        try:
            payload = json.loads(meta)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("source") in name_set:
            selected.append(row["conversation_id"])
    return selected


def _write_run_json(archive_root: Path, payload: dict) -> Path:
    runs_dir = archive_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"run-{payload['timestamp']}.json"
    run_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_path


def run_sources(
    *,
    config: Config,
    profile: Profile,
    stage: str = "all",
    plan: Optional[PlanResult] = None,
    ui: Optional[object] = None,
    source_names: Optional[Sequence[str]] = None,
    profile_name: Optional[str] = None,
    redact_store: bool = False,
) -> RunResult:
    start = time.perf_counter()
    counts = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }
    changed_counts = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
    }
    processed_ids: List[str] = []

    if stage in {"ingest", "all"}:
        for source in _select_sources(config, source_names):
            if source.type == "drive":
                conversations = iter_drive_conversations(
                    source=source,
                    profile=profile,
                    archive_root=config.archive_root,
                    ui=ui,
                    download_assets=True,
                )
            else:
                conversations = iter_source_conversations(source)
            for convo in conversations:
                convo_id, result_counts, content_changed = _ingest_conversation(
                    convo,
                    source.name,
                    redact_store=redact_store,
                    archive_root=config.archive_root,
                )
                ingest_changed = (
                    result_counts["conversations"]
                    + result_counts["messages"]
                    + result_counts["attachments"]
                ) > 0
                if ingest_changed:
                    processed_ids.append(convo_id)
                if content_changed:
                    changed_counts["conversations"] += 1
                for key, value in result_counts.items():
                    counts[key] += value

    if stage in {"render", "all"}:
        if stage == "render":
            ids = _all_conversation_ids(source_names)
        else:
            ids = processed_ids
        for convo_id in ids:
            render_conversation(
                conversation_id=convo_id,
                archive_root=config.archive_root,
                html_mode=profile.html,
                sanitize_html=profile.sanitize_html,
            )

    indexed = False
    index_error: Optional[str] = None
    if profile.index:
        try:
            if stage == "index":
                rebuild_index()
                indexed = True
            elif stage == "all":
                idx = index_status()
                if not idx["exists"]:
                    rebuild_index()
                    indexed = True
                elif processed_ids:
                    update_index_for_conversations(processed_ids)
                    indexed = True
        except Exception as exc:
            index_error = str(exc)
            indexed = False

    duration_ms = int((time.perf_counter() - start) * 1000)
    drift = {
        "new": {"conversations": 0, "messages": 0, "attachments": 0},
        "removed": {"conversations": 0, "messages": 0, "attachments": 0},
        "changed": dict(changed_counts),
    }
    processed_conversations = counts["conversations"] + counts["skipped_conversations"]
    processed_messages = counts["messages"] + counts["skipped_messages"]
    processed_attachments = counts["attachments"] + counts["skipped_attachments"]
    if plan:
        expected_conversations = plan.counts.get("conversations", 0)
        expected_messages = plan.counts.get("messages", 0)
        expected_attachments = plan.counts.get("attachments", 0)
        drift["new"]["conversations"] = max(processed_conversations - expected_conversations, 0)
        drift["new"]["messages"] = max(processed_messages - expected_messages, 0)
        drift["new"]["attachments"] = max(processed_attachments - expected_attachments, 0)
        drift["removed"]["conversations"] = max(expected_conversations - processed_conversations, 0)
        drift["removed"]["messages"] = max(expected_messages - processed_messages, 0)
        drift["removed"]["attachments"] = max(expected_attachments - processed_attachments, 0)
    else:
        drift["new"]["conversations"] = counts["conversations"]
        drift["new"]["messages"] = counts["messages"]
        drift["new"]["attachments"] = counts["attachments"]

    run_id = uuid4().hex
    run_payload = {
        "run_id": run_id,
        "timestamp": int(time.time()),
        "counts": counts,
        "drift": drift,
        "indexed": indexed,
        "index_error": index_error,
        "duration_ms": duration_ms,
        "profile": profile_name,
    }
    _write_run_json(config.archive_root, run_payload)
    with open_connection(None) as conn:
        record_run(
            conn,
            RunRecord(
                run_id=run_id,
                timestamp=str(run_payload["timestamp"]),
                plan_snapshot=plan.counts if plan else None,
                counts=counts,
                drift=drift,
                indexed=indexed,
                duration_ms=duration_ms,
                profile=profile_name,
            ),
        )
        conn.commit()

    return RunResult(
        run_id=run_id,
        counts=counts,
        drift=drift,
        indexed=indexed,
        index_error=index_error,
        duration_ms=duration_ms,
    )


def latest_run() -> Optional[dict]:
    with open_connection(None) as conn:
        row = conn.execute(
            "SELECT * FROM runs ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


__all__ = ["plan_sources", "run_sources", "PlanResult", "RunResult", "latest_run"]
