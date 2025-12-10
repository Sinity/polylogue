"""Database-first renderer that generates markdown from database content."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..db import open_connection
from ..render import MarkdownDocument, AttachmentInfo
from ..util import sanitize_filename


@dataclass
class ConversationData:
    """Conversation metadata from database."""
    provider: str
    conversation_id: str
    slug: str
    title: Optional[str]
    current_branch: Optional[str]
    root_message_id: Optional[str]
    last_updated: Optional[str]
    content_hash: Optional[str]
    metadata: Optional[Dict]


@dataclass
class MessageData:
    """Message data from database."""
    message_id: str
    parent_id: Optional[str]
    position: int
    timestamp: Optional[str]
    role: Optional[str]
    content_hash: Optional[str]
    rendered_text: Optional[str]
    raw_json: Optional[str]
    token_count: int
    word_count: int
    attachment_count: int
    attachment_names: Optional[str]
    metadata: Optional[Dict]


@dataclass
class AttachmentData:
    """Attachment data from database."""
    attachment_name: str
    attachment_path: Optional[str]
    size_bytes: Optional[int]
    content_hash: Optional[str]
    mime_type: Optional[str]
    text_content: Optional[str]
    text_bytes: Optional[int]
    truncated: bool
    ocr_used: bool


class DatabaseRenderer:
    """Renders markdown/HTML from database conversations."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize renderer with optional database path."""
        self.db_path = db_path

    def load_conversation(
        self,
        provider: str,
        conversation_id: str,
    ) -> Optional[ConversationData]:
        """Load conversation metadata from database."""
        with open_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT provider, conversation_id, slug, title, current_branch,
                       root_message_id, last_updated, content_hash, metadata_json
                FROM conversations
                WHERE provider = ? AND conversation_id = ?
                """,
                (provider, conversation_id),
            ).fetchone()

            if not row:
                return None

            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else None

            return ConversationData(
                provider=row["provider"],
                conversation_id=row["conversation_id"],
                slug=row["slug"],
                title=row["title"],
                current_branch=row["current_branch"],
                root_message_id=row["root_message_id"],
                last_updated=row["last_updated"],
                content_hash=row["content_hash"],
                metadata=metadata,
            )

    def load_messages(
        self,
        provider: str,
        conversation_id: str,
        branch_id: str,
    ) -> List[MessageData]:
        """Load messages for a conversation branch from database."""
        with open_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT message_id, parent_id, position, timestamp, role,
                       content_hash, rendered_text, raw_json, token_count,
                       word_count, attachment_count, attachment_names, metadata_json
                FROM messages
                WHERE provider = ? AND conversation_id = ? AND branch_id = ?
                ORDER BY position
                """,
                (provider, conversation_id, branch_id),
            ).fetchall()

            messages = []
            for row in rows:
                metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else None
                messages.append(MessageData(
                    message_id=row["message_id"],
                    parent_id=row["parent_id"],
                    position=row["position"],
                    timestamp=row["timestamp"],
                    role=row["role"],
                    content_hash=row["content_hash"],
                    rendered_text=row["rendered_text"],
                    raw_json=row["raw_json"],
                    token_count=row["token_count"] or 0,
                    word_count=row["word_count"] or 0,
                    attachment_count=row["attachment_count"] or 0,
                    attachment_names=row["attachment_names"],
                    metadata=metadata,
                ))

            return messages

    def load_attachments(
        self,
        provider: str,
        conversation_id: str,
    ) -> List[AttachmentData]:
        """Load attachments for a conversation from database."""
        with open_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT attachment_name, attachment_path, size_bytes, content_hash,
                       mime_type, text_content, text_bytes, truncated, ocr_used
                FROM attachments
                WHERE provider = ? AND conversation_id = ?
                """,
                (provider, conversation_id),
            ).fetchall()

            attachments = []
            for row in rows:
                attachments.append(AttachmentData(
                    attachment_name=row["attachment_name"],
                    attachment_path=row["attachment_path"],
                    size_bytes=row["size_bytes"],
                    content_hash=row["content_hash"],
                    mime_type=row["mime_type"],
                    text_content=row["text_content"],
                    text_bytes=row["text_bytes"],
                    truncated=bool(row["truncated"]),
                    ocr_used=bool(row["ocr_used"]),
                ))

            return attachments

    def render_conversation(
        self,
        provider: str,
        conversation_id: str,
        output_dir: Path,
        *,
        branch_id: Optional[str] = None,
        include_attachments: bool = True,
    ) -> Path:
        """Render a single conversation from DB to markdown.

        Args:
            provider: Provider name (e.g., "chatgpt", "claude")
            conversation_id: Provider's conversation ID
            output_dir: Directory to write markdown files
            branch_id: Specific branch to render (defaults to current_branch)
            include_attachments: Whether to extract attachment files

        Returns:
            Path to the rendered conversation.md file
        """
        # 1. Load conversation metadata
        conversation = self.load_conversation(provider, conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {provider}/{conversation_id}")

        # 2. Determine which branch to render
        target_branch = branch_id or conversation.current_branch
        if not target_branch:
            # If no branch specified and no current_branch, try to find the first branch
            with open_connection(self.db_path) as conn:
                row = conn.execute(
                    """
                    SELECT branch_id FROM branches
                    WHERE provider = ? AND conversation_id = ?
                    ORDER BY is_current DESC, depth ASC
                    LIMIT 1
                    """,
                    (provider, conversation_id),
                ).fetchone()
                if row:
                    target_branch = row["branch_id"]
                else:
                    raise ValueError(f"No branches found for conversation {provider}/{conversation_id}")

        # 3. Load messages for the branch
        messages = self.load_messages(provider, conversation_id, target_branch)

        # 4. Build markdown document
        body_parts = []
        for msg in messages:
            role_label = (msg.role or "unknown").capitalize()
            body_parts.append(f"## {role_label}\n\n{msg.rendered_text or ''}\n")

        body = "\n".join(body_parts)

        # 5. Build metadata
        metadata = {
            "polylogue": {
                "provider": provider,
                "conversationId": conversation_id,
                "title": conversation.title or "Untitled",
                "slug": conversation.slug,
                "lastUpdated": conversation.last_updated,
                "contentHash": conversation.content_hash,
            }
        }

        # Merge in conversation metadata if present
        if conversation.metadata:
            for key, value in conversation.metadata.items():
                if key not in metadata:
                    metadata[key] = value

        # 6. Calculate stats
        total_tokens = sum(msg.token_count for msg in messages)
        total_words = sum(msg.word_count for msg in messages)
        total_attachments = sum(msg.attachment_count for msg in messages)

        stats = {
            "message_count": len(messages),
            "token_count": total_tokens,
            "word_count": total_words,
            "attachment_count": total_attachments,
        }

        # 7. Load attachments if requested
        attachments_info: List[AttachmentInfo] = []
        if include_attachments:
            attachments = self.load_attachments(provider, conversation_id)
            for att in attachments:
                attachments_info.append(AttachmentInfo(
                    name=att.attachment_name,
                    link=f"attachment://{att.attachment_name}",
                    local_path=Path(att.attachment_path) if att.attachment_path else None,
                    size_bytes=att.size_bytes,
                    remote=False,
                ))

        document = MarkdownDocument(
            body=body,
            metadata=metadata,
            attachments=attachments_info,
            stats=stats,
        )

        # 8. Write to disk
        conversation_dir = output_dir / conversation.slug
        conversation_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = conversation_dir / "conversation.md"

        markdown_path.write_text(document.to_markdown(), encoding="utf-8")

        return markdown_path

    def render_all(
        self,
        output_dir: Path,
        *,
        provider: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Path]:
        """Render all conversations matching criteria.

        Args:
            output_dir: Directory to write markdown files
            provider: Optional provider filter
            since: Optional datetime filter (only render conversations updated since)

        Returns:
            List of paths to rendered conversation.md files
        """
        conversations = self.query_conversations(provider=provider, since=since)

        results = []
        for conv in conversations:
            try:
                path = self.render_conversation(
                    conv.provider,
                    conv.conversation_id,
                    output_dir,
                )
                results.append(path)
            except Exception as e:
                print(f"Warning: Failed to render {conv.provider}/{conv.conversation_id}: {e}")
                continue

        return results

    def query_conversations(
        self,
        *,
        provider: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[ConversationData]:
        """Query conversations from database.

        Args:
            provider: Optional provider filter
            since: Optional datetime filter

        Returns:
            List of conversation metadata
        """
        with open_connection(self.db_path) as conn:
            query = """
                SELECT provider, conversation_id, slug, title, current_branch,
                       root_message_id, last_updated, content_hash, metadata_json
                FROM conversations
                WHERE 1=1
            """
            params: List = []

            if provider:
                query += " AND provider = ?"
                params.append(provider)

            if since:
                query += " AND last_updated >= ?"
                params.append(since.isoformat())

            query += " ORDER BY last_updated DESC"

            rows = conn.execute(query, params).fetchall()

            conversations = []
            for row in rows:
                metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else None
                conversations.append(ConversationData(
                    provider=row["provider"],
                    conversation_id=row["conversation_id"],
                    slug=row["slug"],
                    title=row["title"],
                    current_branch=row["current_branch"],
                    root_message_id=row["root_message_id"],
                    last_updated=row["last_updated"],
                    content_hash=row["content_hash"],
                    metadata=metadata,
                ))

            return conversations
