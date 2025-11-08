from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

from .. import util
from .state_store import StateStore


def _default_store() -> StateStore:
    return StateStore(util.STATE_PATH)


@dataclass
class ConversationStateRepository:
    """Repository wrapper around the JSON state cache."""

    store: StateStore = field(default_factory=_default_store)

    def load(self) -> Dict[str, Any]:
        state = self.store.load()
        return state if isinstance(state, dict) else {}

    def get(self, provider: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        provider_map = self._provider_map(provider)
        entry = provider_map.get(conversation_id)
        return entry if isinstance(entry, dict) else None

    def upsert(self, provider: str, conversation_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        def _mutate(state: dict) -> None:
            conversations = state.get("conversations")
            if not isinstance(conversations, dict):
                conversations = {}
                state["conversations"] = conversations
            provider_map = conversations.get(provider)
            if not isinstance(provider_map, dict):
                provider_map = {}
                conversations[provider] = provider_map
            provider_map[conversation_id] = dict(payload)

        self.store.mutate(_mutate)
        result = self.get(provider, conversation_id)
        return result or {}

    def remove(self, provider: str, conversation_id: str) -> None:
        def _mutate(state: dict) -> None:
            conversations = state.get("conversations")
            if not isinstance(conversations, dict):
                return
            provider_map = conversations.get(provider)
            if not isinstance(provider_map, dict):
                return
            provider_map.pop(conversation_id, None)
            if not provider_map:
                conversations.pop(provider, None)

        self.store.mutate(_mutate)

    def provider_items(self, provider: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        provider_map = self._provider_map(provider)
        for key, value in provider_map.items():
            if isinstance(value, dict):
                yield key, value

    def providers(self) -> Iterable[str]:
        state = self.load()
        conversations = state.get("conversations")
        if not isinstance(conversations, dict):
            return ()
        return tuple(conversations.keys())

    def _provider_map(self, provider: str) -> Dict[str, Any]:
        state = self.load()
        conversations = state.get("conversations")
        if not isinstance(conversations, dict):
            return {}
        provider_map = conversations.get(provider)
        if not isinstance(provider_map, dict):
            return {}
        return provider_map
