from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from polylogue.store import AttachmentRecord, ConversationRecord, MessageRecord, store_records


class DbFactory:
    """Helper to seed the database with consistent records."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def create_conversation(
        self,
        id: str | None = None,
        provider: str = "test",
        title: str = "Test Conversation",
        messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """Create a conversation with messages in the DB."""
        cid = id or str(uuid4())

        conv_rec = ConversationRecord(
            conversation_id=cid,
            provider_name=provider,
            provider_conversation_id=f"ext-{cid}",
            title=title,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash=uuid4().hex,
            version=1,
        )

        msg_recs = []
        att_recs = []

        if messages:
            for _, msg in enumerate(messages):
                mid = msg.get("id") or str(uuid4())
                m_rec = MessageRecord(
                    message_id=mid,
                    conversation_id=cid,
                    role=msg.get("role", "user"),
                    text=msg.get("text", "hello"),
                    timestamp=msg.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                    content_hash=uuid4().hex,
                    version=1,
                )
                msg_recs.append(m_rec)

                # Handle attachments if any embedded in msg dict (custom convention for factory)
                if "attachments" in msg:
                    for att in msg["attachments"]:
                        aid = att.get("id") or str(uuid4())
                        att_recs.append(
                            AttachmentRecord(
                                attachment_id=aid,
                                conversation_id=cid,
                                message_id=mid,
                                mime_type=att.get("mime_type", "application/octet-stream"),
                                size_bytes=att.get("size_bytes", 1024),
                                path=att.get("path"),
                                provider_meta=att.get("meta"),
                            )
                        )

        from polylogue.db import open_connection

        with open_connection(self.db_path) as conn:
            store_records(
                conversation=conv_rec,
                messages=msg_recs,
                attachments=att_recs,
                conn=conn,
            )
        return cid
