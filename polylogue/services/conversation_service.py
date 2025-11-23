from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Optional, Tuple

from pathlib import Path

from ..persistence.state import ConversationStateRepository
from ..persistence.database import ConversationDatabase
from .conversation_registrar import ConversationRegistrar, create_default_registrar


@dataclass
class ConversationService:
    registrar: ConversationRegistrar
    _state_cache: Optional[Dict[str, object]] = field(default=None, init=False, repr=False)
    _state_signature: Optional[Tuple[int, int]] = field(default=None, init=False, repr=False)

    @property
    def state_repo(self) -> ConversationStateRepository:
        return self.registrar.state_repo

    @property
    def database(self) -> ConversationDatabase:
        return self.registrar.database

    @property
    def state_path(self):
        return self.database.resolve_path()

    def get_state(self, provider: str, conversation_id: str) -> Optional[Dict[str, object]]:
        return self.registrar.get_state(provider, conversation_id)

    def _state_path(self) -> Path:
        resolved = self.database.resolve_path()
        if resolved:
            return resolved
        return Path("polylogue.db")

    def _current_signature(self) -> Optional[Tuple[int, int]]:
        path = self._state_path()
        try:
            stat = path.stat()
            return (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            return None
        except OSError:
            return None

    def invalidate_state_cache(self) -> None:
        self._state_cache = None
        self._state_signature = None

    def load_state(self) -> Dict[str, object]:
        signature = self._current_signature()
        if self._state_cache is None or signature != self._state_signature:
            self._state_cache = self.state_repo.load()
            self._state_signature = signature
        return self._state_cache

    def iter_state(self) -> Iterator[Tuple[str, str, Dict[str, object]]]:
        state = self.load_state()
        conversations = state.get("conversations")
        if not isinstance(conversations, dict):
            return iter(())
        for provider, convs in conversations.items():
            if not isinstance(convs, dict):
                continue
            for conversation_id, payload in convs.items():
                if isinstance(payload, dict):
                    yield provider, conversation_id, payload

    def iter_conversations(self) -> Iterable[Dict[str, object]]:
        rows = self.database.query(
            "SELECT provider, conversation_id, slug, title FROM conversations"
        )
        for row in rows:
            yield dict(row)

    def delete_conversation(self, provider: str, conversation_id: str) -> None:
        self.database.execute(
            "DELETE FROM conversations WHERE provider = ? AND conversation_id = ?",
            (provider, conversation_id),
        )


def create_conversation_service(registrar: Optional[ConversationRegistrar] = None) -> ConversationService:
    registrar = registrar or create_default_registrar()
    return ConversationService(registrar=registrar)


__all__ = ["ConversationService", "create_conversation_service"]
