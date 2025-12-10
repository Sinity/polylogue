from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..archive import Archive
from ..branching import BranchPlan, MessageRecord
from ..config import CONFIG
from ..db import open_connection, replace_attachments, replace_branches, replace_messages, upsert_conversation
from ..persistence.database import ConversationDatabase
from ..persistence.state import ConversationStateRepository


@dataclass
class ConversationRegistrar:
    """Coordinates SQLite persistence for conversations."""

    state_repo: ConversationStateRepository
    database: ConversationDatabase
    archive: Archive

    def get_state(self, provider: Optional[str], conversation_id: Optional[str]) -> Optional[Dict[str, object]]:
        if not provider or not conversation_id:
            return None
        return self.state_repo.get(provider, conversation_id)

    def mark_dirty(
        self,
        provider: str,
        conversation_id: str,
        payload: Dict[str, object],
    ) -> None:
        entry = self.state_repo.get(provider, conversation_id) or {}
        entry.update(payload)
        self.state_repo.upsert(provider, conversation_id, entry)

    def record_document(
        self,
        *,
        provider: Optional[str],
        conversation_id: Optional[str],
        slug: str,
        title: str,
        content_hash: Optional[str],
        collapse_threshold: int,
        attachment_policy: Dict[str, object],
        markdown_path: Path,
        html_path: Optional[Path],
        html_enabled: bool,
        attachments_dir: Optional[Path],
        updated_at: Optional[str],
        created_at: Optional[str],
        last_imported: Optional[str],
        attachment_bytes: int,
        tokens: int,
        words: int,
        dirty: bool,
        extra_state: Optional[Dict[str, object]] = None,
    ) -> None:
        if not provider or not conversation_id:
            return

        state_payload: Dict[str, object] = {
            "slug": slug,
            "title": title,
            "lastUpdated": updated_at,
            "lastImported": last_imported,
            "contentHash": content_hash,
            "collapseThreshold": collapse_threshold,
            "attachmentPolicy": attachment_policy,
            "outputPath": str(markdown_path),
            "htmlPath": str(html_path) if html_path else None,
            "attachmentsDir": str(attachments_dir) if attachments_dir else None,
            "html": html_enabled,
            "dirty": dirty,
        }
        if extra_state:
            state_payload.update({k: v for k, v in extra_state.items() if v is not None})
        self.state_repo.upsert(provider, conversation_id, state_payload)

        metadata = {
            **state_payload,
            "tokens": tokens,
            "words": words,
            "attachment_bytes": attachment_bytes,
            "dirty": dirty,
        }

        with open_connection(self.database.resolve_path()) as conn:
            upsert_conversation(
                conn,
                provider=provider,
                conversation_id=conversation_id,
                slug=slug,
                title=title,
                current_branch=None,
                root_message_id=None,
                last_updated=updated_at,
                content_hash=content_hash,
                metadata=metadata,
            )
            conn.commit()

    def record_branch_plan(
        self,
        *,
        provider: str,
        conversation_id: str,
        slug: str,
        plan: BranchPlan,
        branch_stats: Dict[str, Dict[str, object]],
        records_by_id: Dict[str, MessageRecord],
        attachment_bytes: int,
    ) -> None:
        if not provider or not conversation_id:
            return

        canonical_stats = branch_stats.get(plan.canonical_branch_id, {}) if plan else {}
        total_tokens = int(canonical_stats.get("token_count", 0) or 0)
        total_words = int(canonical_stats.get("word_count", 0) or 0)
        root_message_id = None
        canonical_branch = plan.branches.get(plan.canonical_branch_id)
        if canonical_branch and canonical_branch.message_ids:
            root_message_id = canonical_branch.message_ids[0]

        branch_payloads: List[Dict[str, object]] = []
        for branch_id, info in plan.branches.items():
            stats = branch_stats.get(branch_id, {})
            branch_payloads.append(
                {
                    "branch_id": branch_id,
                    "parent_branch_id": info.parent_branch_id,
                    "label": branch_id,
                    "is_current": info.is_canonical,
                    "depth": info.depth,
                    "metadata": {
                        "message_count": stats.get("message_count", 0),
                        "token_count": stats.get("token_count", 0),
                        "word_count": stats.get("word_count", 0),
                        "first_message_id": stats.get("first_message_id"),
                        "last_message_id": stats.get("last_message_id"),
                        "first_timestamp": stats.get("first_timestamp"),
                        "last_timestamp": stats.get("last_timestamp"),
                        "divergence_index": info.divergence_index,
                    },
                }
            )

        with open_connection(self.database.resolve_path()) as conn:
            if plan:
                conn.execute(
                    """
                    UPDATE conversations
                       SET current_branch = ?,
                           root_message_id = COALESCE(?, root_message_id),
                           slug = COALESCE(?, slug),
                           metadata_json = json_set(
                               COALESCE(metadata_json, '{}'),
                               '$.token_count', ?,
                               '$.word_count', ?,
                               '$.attachment_bytes', ?
                           )
                     WHERE provider = ? AND conversation_id = ?
                    """,
                    (
                        plan.canonical_branch_id,
                        root_message_id,
                        slug,
                        total_tokens,
                        total_words,
                        attachment_bytes,
                        provider,
                        conversation_id,
                    ),
                )

            replace_branches(
                conn,
                provider=provider,
                conversation_id=conversation_id,
                branches=branch_payloads,
            )

            for branch_id, info in plan.branches.items():
                branch_records = [
                    records_by_id[mid]
                    for mid in info.message_ids
                    if mid in records_by_id
                ]
                messages = []
                for idx, record in enumerate(branch_records):
                    content_hash = record.content_hash
                    if not content_hash and record.text:
                        content_hash = hashlib.sha256(record.text.encode("utf-8")).hexdigest()
                    attachment_names = ",".join(
                        sorted({name for name, _ in record.links})
                    ) if record.links else None

                    # Schema v5: Extract structured content
                    content_text = record.text if record.text else None
                    content_json = None
                    model = None

                    # Try to extract model from chunk/metadata
                    if isinstance(record.chunk, dict):
                        model = record.chunk.get("model") or record.metadata.get("model")
                        # Check if this is a tool call or structured content
                        if record.role in ("tool_call", "function_call"):
                            content_json = json.dumps(record.chunk.get("arguments", record.chunk), ensure_ascii=False)
                        elif "tool_calls" in record.chunk or "function_call" in record.chunk:
                            content_json = json.dumps({
                                "tool_calls": record.chunk.get("tool_calls"),
                                "function_call": record.chunk.get("function_call"),
                            }, ensure_ascii=False)

                    # Schema v5: Mark leaf nodes
                    is_leaf = 0
                    if idx == len(branch_records) - 1:
                        is_leaf = 1

                    messages.append(
                        {
                            "message_id": record.message_id,
                            "parent_id": record.parent_id,
                            "role": record.role,
                            "model": model,
                            "position": idx,
                            "timestamp": record.timestamp,
                            "token_count": record.token_count,
                            "word_count": record.word_count,
                            "attachment_count": record.attachments,
                            "attachment_names": attachment_names,
                            "content_text": content_text,
                            "content_json": content_json,
                            "rendered_text": record.text,
                            "content_hash": content_hash,
                            "raw_json": json.dumps(record.chunk, ensure_ascii=False),
                            "is_leaf": is_leaf,
                            "metadata": record.metadata,
                        }
                    )
                replace_messages(
                    conn,
                    provider=provider,
                    conversation_id=conversation_id,
                    branch_id=branch_id,
                    messages=messages,
                )
            conn.commit()

    def record_attachments(
        self,
        *,
        provider: str,
        conversation_id: str,
        attachments: List[Dict[str, object]],
    ) -> None:
        if not provider or not conversation_id:
            return
        with open_connection(self.database.resolve_path()) as conn:
            replace_attachments(
                conn,
                provider=provider,
                conversation_id=conversation_id,
                attachments=attachments,
            )
            conn.commit()


def create_default_registrar() -> ConversationRegistrar:
    """Factory helper to build a registrar with default repositories."""

    return ConversationRegistrar(
        state_repo=ConversationStateRepository(),
        database=ConversationDatabase(),
        archive=Archive(CONFIG),
    )
