"""ChatGPT conversation-level typed models."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from polylogue.lib.timestamps import parse_timestamp

from .chatgpt_message_models import ChatGPTMessage


class ChatGPTNode(BaseModel):
    """A node in the ChatGPT conversation tree."""

    model_config = ConfigDict(extra="allow")

    id: str
    message: ChatGPTMessage | None = None
    parent: str | None = None
    children: list[str] = Field(default_factory=list)


class ChatGPTConversation(BaseModel):
    """A complete ChatGPT conversation export."""

    model_config = ConfigDict(extra="allow")

    id: str
    conversation_id: str
    title: str
    create_time: float
    update_time: float
    mapping: dict[str, ChatGPTNode] = Field(default_factory=dict)
    current_node: str
    default_model_slug: str | None = None
    is_archived: bool = False
    is_starred: bool | None = None
    is_read_only: bool | None = None
    gizmo_id: str | None = None
    gizmo_type: str | None = None
    memory_scope: str = "global_enabled"
    safe_urls: list[str] = Field(default_factory=list)
    blocked_urls: list[str] = Field(default_factory=list)
    moderation_results: list[object] = Field(default_factory=list)

    @property
    def messages(self) -> list[ChatGPTMessage]:
        messages: list[ChatGPTMessage] = []
        root_id = None
        for node_id, entry in self.mapping.items():
            if entry.parent is None or entry.parent == "client-created-root":
                root_id = node_id
                break

        if not root_id:
            return messages

        current_id = root_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            node = self.mapping.get(current_id)
            if not node:
                break
            if node.message:
                messages.append(node.message)
            if node.children:
                current_id = node.children[0]
            else:
                break
        return messages

    @property
    def created_at(self) -> datetime | None:
        return parse_timestamp(self.create_time)

    @property
    def updated_at(self) -> datetime | None:
        return parse_timestamp(self.update_time)

    def iter_user_assistant_pairs(self) -> Iterator[tuple[ChatGPTMessage, ChatGPTMessage]]:
        messages = self.messages
        i = 0
        while i < len(messages) - 1:
            user_msg = messages[i]
            asst_msg = messages[i + 1]
            if user_msg.role_normalized == "user" and asst_msg.role_normalized == "assistant":
                yield (user_msg, asst_msg)
                i += 2
            else:
                i += 1
