from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

from ..db import open_connection, upsert_conversation
from ..paths import STATE_HOME
from .database import ConversationDatabase


@dataclass
class ConversationStateRepository:
    """Read/write access to conversation metadata stored in SQLite."""

    database: ConversationDatabase = field(default_factory=ConversationDatabase)

    def load(self) -> Dict[str, Any]:
        conversations: Dict[str, Dict[str, Dict[str, Any]]] = {}
        with open_connection(self.database.resolve_path()) as conn:
            rows = conn.execute(
                "SELECT provider, conversation_id, metadata_json FROM conversations"
            ).fetchall()
        for row in rows:
            provider = row["provider"]
            convo_id = row["conversation_id"]
            metadata = self._decode_metadata(row["metadata_json"])
            conversations.setdefault(provider, {})[convo_id] = metadata
        return {"conversations": conversations}

    def get(self, provider: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        with open_connection(self.database.resolve_path()) as conn:
            row = conn.execute(
                """
                SELECT metadata_json
                  FROM conversations
                 WHERE provider = ? AND conversation_id = ?
                """,
                (provider, conversation_id),
            ).fetchone()
        if not row:
            return None
        return self._decode_metadata(row["metadata_json"])

    def upsert(self, provider: str, conversation_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        existing = self.get(provider, conversation_id) or {}
        merged = dict(existing)
        merged.update(payload)
        slug = merged.get("slug") or conversation_id
        title = merged.get("title")
        current_branch = merged.get("current_branch") or merged.get("currentBranch")
        root_message_id = merged.get("root_message_id")
        last_updated = merged.get("lastUpdated")
        content_hash = merged.get("contentHash")
        with open_connection(self.database.resolve_path()) as conn:
            upsert_conversation(
                conn,
                provider=provider,
                conversation_id=conversation_id,
                slug=str(slug),
                title=title,
                current_branch=current_branch,
                root_message_id=root_message_id,
                last_updated=last_updated,
                content_hash=content_hash,
                metadata=merged,
            )
            conn.commit()
        return merged

    def remove(self, provider: str, conversation_id: str) -> None:
        with open_connection(self.database.resolve_path()) as conn:
            conn.execute(
                "DELETE FROM conversations WHERE provider = ? AND conversation_id = ?",
                (provider, conversation_id),
            )
            conn.commit()

    def provider_items(self, provider: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        with open_connection(self.database.resolve_path()) as conn:
            rows = conn.execute(
                """
                SELECT conversation_id, metadata_json
                  FROM conversations
                 WHERE provider = ?
                """,
                (provider,),
            ).fetchall()
        for row in rows:
            yield row["conversation_id"], self._decode_metadata(row["metadata_json"])

    def providers(self) -> Iterable[str]:
        with open_connection(self.database.resolve_path()) as conn:
            rows = conn.execute("SELECT DISTINCT provider FROM conversations").fetchall()
        return tuple(row["provider"] for row in rows)

    @property
    def path(self) -> Optional[Any]:  # compatibility shim for diagnostics
        return self.database.resolve_path() or (STATE_HOME / "polylogue.db")

    @staticmethod
    def _decode_metadata(raw: Optional[str]) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
