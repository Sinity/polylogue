"""Shared file/export builders for inbox and provider-source tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def make_chatgpt_node(
    msg_id: str,
    role: str,
    content_parts: list[str],
    children: list[str] | None = None,
    timestamp: float | None = None,
    metadata: dict | None = None,
    parent: str | None = None,
) -> dict[str, Any]:
    """Generate a ChatGPT export mapping node for parser tests."""
    node = {
        "id": msg_id,
        "message": {
            "id": msg_id,
            "author": {"role": role},
            "content": {"parts": content_parts},
        },
    }
    if children:
        node["children"] = children
    if parent:
        node["parent"] = parent
    if timestamp:
        node["message"]["create_time"] = timestamp
    if metadata:
        node["message"]["metadata"] = metadata
    return node


def make_claude_chat_message(
    uuid: str,
    sender: str,
    text: str,
    attachments: list[dict] | None = None,
    files: list[dict] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Generate a Claude AI chat_messages entry for parser tests."""
    msg = {"uuid": uuid, "text": text}
    if sender:
        msg["sender"] = sender
    if attachments:
        msg["attachments"] = attachments
    if files:
        msg["files"] = files
    if timestamp:
        msg["created_at"] = timestamp
    return msg


class ChatGPTExportBuilder:
    """Builder for ChatGPT export payloads with mapping nodes."""

    def __init__(self, conv_id: str):
        self.conv_id = conv_id
        self._title: str | None = None
        self._nodes: list[dict[str, Any]] = []
        self._node_counter = 0
        self._timestamp = 1704067200.0

    def title(self, title: str) -> ChatGPTExportBuilder:
        self._title = title
        return self

    def add_node(
        self,
        role: str,
        *content_parts: str,
        node_id: str | None = None,
        metadata: dict | None = None,
        model_slug: str | None = None,
    ) -> ChatGPTExportBuilder:
        self._node_counter += 1
        nid = node_id or f"node-{self._node_counter}"

        meta = metadata or {}
        if model_slug:
            meta["model_slug"] = model_slug

        self._nodes.append(
            make_chatgpt_node(
                nid,
                role,
                list(content_parts),
                timestamp=self._timestamp,
                metadata=meta if meta else None,
            )
        )
        self._timestamp += 1.0
        return self

    def add_system_node(self, content: str, node_id: str | None = None) -> ChatGPTExportBuilder:
        return self.add_node("system", content, node_id=node_id)

    def add_tool_node(
        self,
        tool_name: str,
        result: str,
        node_id: str | None = None,
    ) -> ChatGPTExportBuilder:
        return self.add_node("tool", result, node_id=node_id, metadata={"name": tool_name})

    def build(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.conv_id,
            "mapping": {node["id"]: node for node in self._nodes},
        }
        if self._title:
            result["title"] = self._title
        return result

    def write_to(self, path: Path) -> Path:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(), indent=2), encoding="utf-8")
        return path


class GenericConversationBuilder:
    """Builder for simple message-list provider payloads."""

    def __init__(self, conv_id: str):
        self.conv_id = conv_id
        self._title: str | None = None
        self._messages: list[dict[str, Any]] = []
        self._msg_counter = 0

    def title(self, title: str) -> GenericConversationBuilder:
        self._title = title
        return self

    def add_message(
        self,
        role: str,
        content: str,
        message_id: str | None = None,
        text: str | None = None,
    ) -> GenericConversationBuilder:
        self._msg_counter += 1
        msg_id = message_id or f"m{self._msg_counter}"
        msg: dict[str, Any] = {"id": msg_id, "role": role}
        if text is not None:
            msg["text"] = text
        else:
            msg["content"] = content
        self._messages.append(msg)
        return self

    def add_user(self, content: str, **kwargs) -> GenericConversationBuilder:
        return self.add_message("user", content, **kwargs)

    def add_assistant(self, content: str, **kwargs) -> GenericConversationBuilder:
        return self.add_message("assistant", content, **kwargs)

    def build(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self.conv_id, "messages": self._messages}
        if self._title:
            result["title"] = self._title
        return result

    def write_to(self, path: Path) -> Path:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(), indent=2), encoding="utf-8")
        return path


class InboxBuilder:
    """Builder for inbox directories populated with provider exports."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.files: list[tuple[Path, str]] = []

    def add_json_file(self, filename: str, data: Any) -> InboxBuilder:
        import json

        path = self.base_path / filename
        self.files.append((path, json.dumps(data, indent=2)))
        return self

    def add_jsonl_file(self, filename: str, entries: list[Any]) -> InboxBuilder:
        import json

        path = self.base_path / filename
        content = "\n".join(json.dumps(entry) for entry in entries) + "\n"
        self.files.append((path, content))
        return self

    def add_codex_conversation(
        self,
        conv_id: str,
        title: str | None = None,
        messages: list[tuple[str, str]] | None = None,
        filename: str | None = None,
    ) -> InboxBuilder:
        builder = GenericConversationBuilder(conv_id)
        if title:
            builder.title(title)
        for role, content in messages or [("user", "Hello"), ("assistant", "Hi there!")]:
            builder.add_message(role, content)
        return self.add_json_file(filename or f"{conv_id}.json", builder.build())

    def add_chatgpt_export(
        self,
        conv_id: str,
        title: str | None = None,
        nodes: list[dict] | None = None,
        filename: str | None = None,
    ) -> InboxBuilder:
        if nodes is None:
            builder = ChatGPTExportBuilder(conv_id)
            if title:
                builder.title(title)
            builder.add_node("user", "Hello").add_node("assistant", "Hi there!")
            payload = builder.build()
        else:
            payload = {
                "id": conv_id,
                "mapping": {node["id"]: node for node in nodes},
            }
            if title:
                payload["title"] = title
        return self.add_json_file(filename or f"chatgpt_{conv_id}.json", payload)

    def add_claude_export(
        self,
        conv_id: str,
        name: str | None = None,
        chat_messages: list[dict] | None = None,
        filename: str | None = None,
        wrap_in_conversations: bool = True,
    ) -> InboxBuilder:
        conversation = {
            "id": conv_id,
            "chat_messages": chat_messages
            or [
                make_claude_chat_message("m1", "human", "Hello"),
                make_claude_chat_message("m2", "assistant", "Hi there!"),
            ],
        }
        if name:
            conversation["name"] = name
        payload = {"conversations": [conversation]} if wrap_in_conversations else conversation
        return self.add_json_file(filename or f"claude_{conv_id}.json", payload)

    def build(self) -> Path:
        for path, content in self.files:
            path.write_text(content, encoding="utf-8")
        return self.base_path

    def get_file_path(self, filename: str) -> Path:
        return self.base_path / filename
