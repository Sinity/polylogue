"""Test utilities and builders for Polylogue test suite.

These helpers reduce boilerplate in parametrized tests and provide consistent
test data generation across the test suite.

Usage:
    from tests.helpers import ConversationBuilder, make_message, db_setup

Created during aggressive test consolidation to eliminate repeated patterns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from polylogue.storage.backends.sqlite import open_connection
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord, store_records


# =============================================================================
# DATABASE SETUP UTILITIES
# =============================================================================


def db_setup(workspace_env) -> Path:
    """Initialize database path in workspace environment.

    Usage in tests:
        db_path = db_setup(workspace_env)
        builder = ConversationBuilder(db_path, "test-conv")
    """
    db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


# =============================================================================
# MESSAGE/CONVERSATION BUILDERS (Fluent API)
# =============================================================================


class MessageBuilder:
    """Fluent builder for MessageRecord.

    Example:
        msg = (MessageBuilder("m1", "conv1")
               .role("user")
               .text("Hello!")
               .timestamp("2024-01-01T10:00:00Z")
               .build())
    """

    def __init__(self, message_id: str, conversation_id: str):
        self.data = {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "role": "user",
            "text": "Default text",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_hash": uuid4().hex[:16],
            "provider_meta": None,
        }

    def role(self, role: str) -> MessageBuilder:
        self.data["role"] = role
        return self

    def text(self, text: str) -> MessageBuilder:
        self.data["text"] = text
        return self

    def timestamp(self, timestamp: str | None) -> MessageBuilder:
        self.data["timestamp"] = timestamp
        return self

    def meta(self, meta: dict | None) -> MessageBuilder:
        self.data["provider_meta"] = meta
        return self

    def build(self) -> MessageRecord:
        return MessageRecord(**self.data)


class ConversationBuilder:
    """Fluent builder for creating conversations in test database.

    Example:
        (ConversationBuilder(db_path, "test-conv")
         .title("My Test")
         .provider("chatgpt")
         .add_message(msg1)
         .add_message(msg2)
         .add_attachment(att1)
         .save())

    Simplifies:
        - Creating ConversationRecord
        - Adding messages/attachments
        - Calling store_records with proper open_connection
    """

    def __init__(self, db_path: Path, conversation_id: str):
        self.db_path = db_path
        now = datetime.now(timezone.utc).isoformat()

        self.conv = ConversationRecord(
            conversation_id=conversation_id,
            provider_name="test",
            provider_conversation_id=f"ext-{conversation_id}",
            title="Test Conversation",
            created_at=now,
            updated_at=now,
            content_hash=uuid4().hex,
        )
        self.messages: list[MessageRecord] = []
        self.attachments: list[AttachmentRecord] = []

    def title(self, title: str | None) -> ConversationBuilder:
        self.conv = self.conv.model_copy(update={"title": title})
        return self

    def provider(self, provider: str) -> ConversationBuilder:
        self.conv = self.conv.model_copy(update={"provider_name": provider})
        return self

    def created_at(self, created_at: str) -> ConversationBuilder:
        self.conv = self.conv.model_copy(update={"created_at": created_at})
        return self

    def updated_at(self, updated_at: str) -> ConversationBuilder:
        self.conv = self.conv.model_copy(update={"updated_at": updated_at})
        return self

    def add_message(
        self,
        message_id: str | None = None,
        role: str = "user",
        text: str = "Test message",
        timestamp: str | None | object = ...,  # ... = auto-generate, None = no timestamp
        **kwargs,
    ) -> ConversationBuilder:
        """Add a message to the conversation.

        Can pass MessageRecord directly or use kwargs to build one.

        Usage:
            .add_message("m1", role="user", text="Hello!")
            .add_message("m2", timestamp=None)  # Explicitly no timestamp
            .add_message(message_record)
        """
        if message_id is None:
            msg_id = f"m{len(self.messages) + 1}"
        else:
            msg_id = message_id

        # Handle timestamp: ... = auto-generate, None = keep None, str = use value
        if timestamp is ...:
            ts = datetime.now(timezone.utc).isoformat()
        else:
            ts = timestamp

        msg = MessageRecord(
            message_id=msg_id,
            conversation_id=self.conv.conversation_id,
            role=role,
            text=text,
            timestamp=ts,
            content_hash=uuid4().hex[:16],
            **kwargs,
        )
        self.messages.append(msg)
        return self

    def add_attachment(
        self,
        attachment_id: str | None = None,
        message_id: str | None | object = ...,  # ... = auto-assign to last message, None = orphaned
        mime_type: str = "application/octet-stream",
        size_bytes: int = 1024,
        path: str | None = None,
        provider_meta: dict | None = None,
    ) -> ConversationBuilder:
        """Add an attachment to the conversation.

        Args:
            message_id: ... (default) = attach to last message, None = orphaned attachment
        """
        if attachment_id is None:
            att_id = f"att{len(self.attachments) + 1}"
        else:
            att_id = attachment_id

        # Handle message_id: ... = auto-assign to last message, None = orphaned
        if message_id is ...:
            msg_id = self.messages[-1].message_id if self.messages else None
        else:
            msg_id = message_id

        att = AttachmentRecord(
            attachment_id=att_id,
            conversation_id=self.conv.conversation_id,
            message_id=msg_id,
            mime_type=mime_type,
            size_bytes=size_bytes,
            path=path,
            provider_meta=provider_meta,
        )
        self.attachments.append(att)
        return self

    def save(self) -> ConversationRecord:
        """Save conversation, messages, and attachments to database."""
        with open_connection(self.db_path) as conn:
            store_records(
                conversation=self.conv,
                messages=self.messages,
                attachments=self.attachments,
                conn=conn,
            )
        return self.conv


# =============================================================================
# QUICK BUILDERS (For simple cases)
# =============================================================================


def make_conversation(
    conversation_id: str = "conv1",
    provider_name: str = "test",
    title: str = "Test Conversation",
    created_at: str | None = None,
    updated_at: str | None = None,
    **kwargs,
) -> ConversationRecord:
    """Quick conversation record creation without storing.

    Usage:
        conv = make_conversation("conv1", provider_name="claude", title="My Conv")
    """
    now = datetime.now(timezone.utc).isoformat()
    return ConversationRecord(
        conversation_id=conversation_id,
        provider_name=provider_name,
        provider_conversation_id=kwargs.pop("provider_conversation_id", f"ext-{conversation_id}"),
        title=title,
        created_at=created_at or now,
        updated_at=updated_at or now,
        content_hash=kwargs.pop("content_hash", uuid4().hex),
        **kwargs,
    )


def make_message(
    message_id: str = "m1",
    conversation_id: str = "conv1",
    role: str = "user",
    text: str = "Test message",
    timestamp: str | None = None,
    **kwargs,
) -> MessageRecord:
    """Quick message creation without builder.

    Usage:
        msg = make_message("m1", role="assistant", text="Reply")
        msg = make_message("m1", content_hash="explicit-hash")  # Override hash
    """
    return MessageRecord(
        message_id=message_id,
        conversation_id=conversation_id,
        role=role,
        text=text,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        content_hash=kwargs.pop("content_hash", uuid4().hex[:16]),
        **kwargs,
    )


def make_attachment(
    attachment_id: str = "att1",
    conversation_id: str = "conv1",
    message_id: str | None = None,
    mime_type: str = "application/octet-stream",
    size_bytes: int = 1024,
    name: str | None = None,
    **kwargs,
) -> AttachmentRecord:
    """Quick attachment creation.

    Usage:
        att = make_attachment("att1", name="file.pdf")
    """
    provider_meta = kwargs.pop("provider_meta", None)
    if name and provider_meta is None:
        provider_meta = {"name": name}

    return AttachmentRecord(
        attachment_id=attachment_id,
        conversation_id=conversation_id,
        message_id=message_id,
        mime_type=mime_type,
        size_bytes=size_bytes,
        provider_meta=provider_meta,
        **kwargs,
    )


# =============================================================================
# PARAMETRIZED TEST HELPERS
# =============================================================================


def assert_messages_ordered(markdown_text: str, *expected_order: str):
    """Assert messages appear in given order in markdown output.

    Usage:
        assert_messages_ordered(result.markdown_text, "First", "Second", "Third")
    """
    indices = []
    for text in expected_order:
        try:
            idx = markdown_text.index(text)
            indices.append((idx, text))
        except ValueError:
            raise AssertionError(f"Expected text '{text}' not found in markdown")

    # Verify order
    for i in range(len(indices) - 1):
        if indices[i][0] >= indices[i + 1][0]:
            raise AssertionError(
                f"Order violation: '{indices[i][1]}' (index {indices[i][0]}) "
                f"should come before '{indices[i + 1][1]}' (index {indices[i + 1][0]})"
            )


def assert_contains_all(text: str, *expected: str):
    """Assert text contains all expected substrings.

    Usage:
        assert_contains_all(result, "user", "assistant", "Hello")
    """
    for expected_text in expected:
        assert expected_text in text, f"Expected '{expected_text}' not found in text"


def assert_not_contains_any(text: str, *unexpected: str):
    """Assert text does NOT contain any of the given substrings.

    Usage:
        assert_not_contains_any(result, "ERROR", "FAIL", "```json")
    """
    for unexpected_text in unexpected:
        assert unexpected_text not in text, f"Unexpected '{unexpected_text}' found in text"


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================


def make_chatgpt_node(
    msg_id: str,
    role: str,
    content_parts: list[str],
    children: list[str] | None = None,
    timestamp: float | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Generate ChatGPT mapping node for importer tests.

    Usage:
        node = make_chatgpt_node("msg1", "user", ["Hello"], timestamp=1704067200)
    """
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
    """Generate Claude AI chat_messages entry for importer tests.

    Usage:
        msg = make_claude_chat_message("u1", "human", "Hello")
    """
    msg = {
        "uuid": uuid,
        "text": text,
    }

    if sender:
        msg["sender"] = sender
    if attachments:
        msg["attachments"] = attachments
    if files:
        msg["files"] = files
    if timestamp:
        msg["created_at"] = timestamp

    return msg


def make_claude_code_message(
    msg_type: str,
    text: str,
    **kwargs,
) -> dict[str, Any]:
    """Generate Claude Code message for importer tests.

    Usage:
        msg = make_claude_code_message("user_message", "Question")
        msg = make_claude_code_message("tool_use", '{"name": "read"}')
    """
    msg = {
        "type": msg_type,
        "text": text,
    }
    msg.update(kwargs)
    return msg


# =============================================================================
# FILE DATA BUILDERS (For creating test inbox files)
# =============================================================================


class InboxBuilder:
    """Fluent builder for creating test inbox directories with files.

    Example:
        inbox = (InboxBuilder(tmp_path)
                 .add_codex_conversation("conv1", messages=[("user", "Hello")])
                 .add_chatgpt_export("conv2", nodes=[...])
                 .add_claude_export("conv3", chat_messages=[...])
                 .build())
    """

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.files: list[tuple[Path, str]] = []

    def add_json_file(self, filename: str, data: Any) -> "InboxBuilder":
        """Add a raw JSON file."""
        import json
        path = self.base_path / filename
        self.files.append((path, json.dumps(data, indent=2)))
        return self

    def add_jsonl_file(self, filename: str, entries: list[Any]) -> "InboxBuilder":
        """Add a JSONL file with multiple entries."""
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
    ) -> "InboxBuilder":
        """Add a simple Codex/generic format conversation.

        Args:
            conv_id: Conversation ID
            title: Optional title
            messages: List of (role, content) tuples
            filename: Custom filename (default: {conv_id}.json)
        """
        if messages is None:
            messages = [("user", "Hello"), ("assistant", "Hi there!")]

        payload: dict[str, Any] = {
            "id": conv_id,
            "messages": [
                {"id": f"m{i+1}", "role": role, "content": content}
                for i, (role, content) in enumerate(messages)
            ],
        }
        if title:
            payload["title"] = title

        fname = filename or f"{conv_id}.json"
        return self.add_json_file(fname, payload)

    def add_chatgpt_export(
        self,
        conv_id: str,
        title: str | None = None,
        nodes: list[dict] | None = None,
        filename: str | None = None,
    ) -> "InboxBuilder":
        """Add a ChatGPT export format conversation.

        Args:
            conv_id: Conversation ID
            title: Optional title
            nodes: List of mapping node dicts (use make_chatgpt_node())
            filename: Custom filename
        """
        if nodes is None:
            nodes = [
                make_chatgpt_node("n1", "user", ["Hello"], timestamp=1704067200),
                make_chatgpt_node("n2", "assistant", ["Hi there!"], timestamp=1704067201),
            ]

        # Build mapping from nodes
        mapping = {}
        for node in nodes:
            mapping[node["id"]] = node

        payload: dict[str, Any] = {
            "id": conv_id,
            "mapping": mapping,
        }
        if title:
            payload["title"] = title

        fname = filename or f"chatgpt_{conv_id}.json"
        return self.add_json_file(fname, payload)

    def add_claude_export(
        self,
        conv_id: str,
        name: str | None = None,
        chat_messages: list[dict] | None = None,
        filename: str | None = None,
        wrap_in_conversations: bool = True,
    ) -> "InboxBuilder":
        """Add a Claude AI export format conversation.

        Args:
            conv_id: Conversation ID
            name: Conversation name/title
            chat_messages: List of message dicts (use make_claude_chat_message())
            filename: Custom filename
            wrap_in_conversations: Wrap in {"conversations": [...]} structure
        """
        if chat_messages is None:
            chat_messages = [
                make_claude_chat_message("m1", "human", "Hello"),
                make_claude_chat_message("m2", "assistant", "Hi there!"),
            ]

        conversation = {
            "id": conv_id,
            "chat_messages": chat_messages,
        }
        if name:
            conversation["name"] = name

        if wrap_in_conversations:
            payload = {"conversations": [conversation]}
        else:
            payload = conversation

        fname = filename or f"claude_{conv_id}.json"
        return self.add_json_file(fname, payload)

    def add_claude_code_jsonl(
        self,
        session_id: str,
        messages: list[dict] | None = None,
        filename: str | None = None,
    ) -> "InboxBuilder":
        """Add a Claude Code JSONL session file.

        Args:
            session_id: Session ID for filename
            messages: List of message dicts (use make_claude_code_message())
            filename: Custom filename
        """
        if messages is None:
            messages = [
                make_claude_code_message("user", "What is 2+2?"),
                make_claude_code_message("assistant", "4"),
            ]

        fname = filename or f"{session_id}.jsonl"
        return self.add_jsonl_file(fname, messages)

    def build(self) -> Path:
        """Write all files and return the inbox path."""
        for path, content in self.files:
            path.write_text(content, encoding="utf-8")
        return self.base_path

    def get_file_path(self, filename: str) -> Path:
        """Get path to a specific file in the inbox."""
        return self.base_path / filename


class ChatGPTExportBuilder:
    """Builder for ChatGPT export format with proper node structure.

    Example:
        export = (ChatGPTExportBuilder("conv1")
                  .title("My Chat")
                  .add_node("user", "Hello")
                  .add_node("assistant", "Hi there!")
                  .add_node("user", "How are you?", model_slug="gpt-4")
                  .build())
    """

    def __init__(self, conv_id: str):
        self.conv_id = conv_id
        self._title: str | None = None
        self._nodes: list[dict] = []
        self._node_counter = 0
        self._timestamp = 1704067200.0  # Base timestamp

    def title(self, title: str) -> "ChatGPTExportBuilder":
        self._title = title
        return self

    def add_node(
        self,
        role: str,
        *content_parts: str,
        node_id: str | None = None,
        metadata: dict | None = None,
        model_slug: str | None = None,
    ) -> "ChatGPTExportBuilder":
        """Add a message node."""
        self._node_counter += 1
        nid = node_id or f"node-{self._node_counter}"

        meta = metadata or {}
        if model_slug:
            meta["model_slug"] = model_slug

        node = make_chatgpt_node(
            nid,
            role,
            list(content_parts),
            timestamp=self._timestamp,
            metadata=meta if meta else None,
        )
        self._nodes.append(node)
        self._timestamp += 1.0
        return self

    def add_system_node(self, content: str, node_id: str | None = None) -> "ChatGPTExportBuilder":
        """Add a system message node."""
        return self.add_node("system", content, node_id=node_id)

    def add_tool_node(
        self,
        tool_name: str,
        result: str,
        node_id: str | None = None,
    ) -> "ChatGPTExportBuilder":
        """Add a tool result node."""
        return self.add_node(
            "tool",
            result,
            node_id=node_id,
            metadata={"name": tool_name},
        )

    def build(self) -> dict[str, Any]:
        """Build the ChatGPT export structure."""
        mapping = {node["id"]: node for node in self._nodes}
        result: dict[str, Any] = {
            "id": self.conv_id,
            "mapping": mapping,
        }
        if self._title:
            result["title"] = self._title
        return result

    def write_to(self, path: Path) -> Path:
        """Build and write to file."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(), indent=2), encoding="utf-8")
        return path


class ClaudeExportBuilder:
    """Builder for Claude AI export format.

    Example:
        export = (ClaudeExportBuilder("conv1")
                  .name("My Claude Chat")
                  .add_message("human", "Hello")
                  .add_message("assistant", "Hi!", attachments=[...])
                  .build())
    """

    def __init__(self, conv_id: str):
        self.conv_id = conv_id
        self._name: str | None = None
        self._messages: list[dict] = []
        self._msg_counter = 0
        self._wrap_in_conversations = True

    def name(self, name: str) -> "ClaudeExportBuilder":
        self._name = name
        return self

    def unwrapped(self) -> "ClaudeExportBuilder":
        """Don't wrap in {"conversations": [...]} structure."""
        self._wrap_in_conversations = False
        return self

    def add_message(
        self,
        sender: str,
        text: str,
        uuid: str | None = None,
        attachments: list[dict] | None = None,
        files: list[dict] | None = None,
        timestamp: str | None = None,
    ) -> "ClaudeExportBuilder":
        """Add a chat message."""
        self._msg_counter += 1
        msg_uuid = uuid or f"msg-{self._msg_counter}"

        msg = make_claude_chat_message(
            msg_uuid,
            sender,
            text,
            attachments=attachments,
            files=files,
            timestamp=timestamp,
        )
        self._messages.append(msg)
        return self

    def add_human(self, text: str, **kwargs) -> "ClaudeExportBuilder":
        """Shorthand for add_message with sender='human'."""
        return self.add_message("human", text, **kwargs)

    def add_assistant(self, text: str, **kwargs) -> "ClaudeExportBuilder":
        """Shorthand for add_message with sender='assistant'."""
        return self.add_message("assistant", text, **kwargs)

    def build(self) -> dict[str, Any]:
        """Build the Claude export structure."""
        conversation = {
            "id": self.conv_id,
            "chat_messages": self._messages,
        }
        if self._name:
            conversation["name"] = self._name

        if self._wrap_in_conversations:
            return {"conversations": [conversation]}
        return conversation

    def write_to(self, path: Path) -> Path:
        """Build and write to file."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(), indent=2), encoding="utf-8")
        return path


class GenericConversationBuilder:
    """Builder for simple/Codex format conversations.

    Example:
        conv = (GenericConversationBuilder("conv1")
                .title("Test Chat")
                .add_message("user", "Hello")
                .add_message("assistant", "Hi!")
                .build())
    """

    def __init__(self, conv_id: str):
        self.conv_id = conv_id
        self._title: str | None = None
        self._messages: list[dict] = []
        self._msg_counter = 0

    def title(self, title: str) -> "GenericConversationBuilder":
        self._title = title
        return self

    def add_message(
        self,
        role: str,
        content: str,
        message_id: str | None = None,
        text: str | None = None,  # Alias for content
    ) -> "GenericConversationBuilder":
        """Add a message. Uses 'content' key by default, but can use 'text'."""
        self._msg_counter += 1
        msg_id = message_id or f"m{self._msg_counter}"

        msg: dict[str, Any] = {
            "id": msg_id,
            "role": role,
        }
        # Support both content and text keys
        if text is not None:
            msg["text"] = text
        else:
            msg["content"] = content

        self._messages.append(msg)
        return self

    def add_user(self, content: str, **kwargs) -> "GenericConversationBuilder":
        """Shorthand for add_message with role='user'."""
        return self.add_message("user", content, **kwargs)

    def add_assistant(self, content: str, **kwargs) -> "GenericConversationBuilder":
        """Shorthand for add_message with role='assistant'."""
        return self.add_message("assistant", content, **kwargs)

    def build(self) -> dict[str, Any]:
        """Build the conversation structure."""
        result: dict[str, Any] = {
            "id": self.conv_id,
            "messages": self._messages,
        }
        if self._title:
            result["title"] = self._title
        return result

    def write_to(self, path: Path) -> Path:
        """Build and write to file."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(), indent=2), encoding="utf-8")
        return path


def write_test_inbox(
    base_path: Path,
    conversations: list[dict[str, Any]] | None = None,
    format: str = "codex",
) -> Path:
    """Quick helper to write a test inbox with default conversations.

    Args:
        base_path: Directory to create inbox in
        conversations: List of conversation payloads (auto-generated if None)
        format: "codex", "chatgpt", or "claude"

    Returns:
        Path to the inbox directory
    """
    import json

    inbox = base_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)

    if conversations is None:
        if format == "codex":
            conversations = [
                GenericConversationBuilder("conv1")
                .title("Test")
                .add_user("Hello")
                .add_assistant("Hi!")
                .build()
            ]
        elif format == "chatgpt":
            conversations = [
                ChatGPTExportBuilder("conv1")
                .title("Test")
                .add_node("user", "Hello")
                .add_node("assistant", "Hi!")
                .build()
            ]
        elif format == "claude":
            conversations = [
                ClaudeExportBuilder("conv1")
                .name("Test")
                .add_human("Hello")
                .add_assistant("Hi!")
                .build()
            ]

    for i, conv in enumerate(conversations):
        filename = conv.get("id", f"conv{i+1}") + ".json"
        (inbox / filename).write_text(json.dumps(conv), encoding="utf-8")

    return inbox


# =============================================================================
# COVERAGE VERIFICATION HELPERS
# =============================================================================


def parametrized_case_count(test_cases: list) -> dict[str, int]:
    """Count parametrized test cases by description pattern.

    Usage:
        CASES = [(input1, expected1, "desc1"), (input2, expected2, "desc2")]
        counts = parametrized_case_count(CASES)
        # Returns: {"total": 2, "desc1": 1, "desc2": 1}
    """
    counts = {"total": len(test_cases)}

    for case in test_cases:
        if len(case) >= 3:
            desc = case[-1]  # Description is usually last element
            counts[desc] = counts.get(desc, 0) + 1

    return counts


def verify_coverage(
    old_test_names: list[str],
    new_test_cases: list[tuple],
    mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify parametrized tests cover all original tests.

    Usage:
        old = ["test_foo_basic", "test_foo_with_arg", "test_foo_empty"]
        new = [(1, "basic"), (2, "with arg"), (None, "empty")]
        mapping = {"test_foo_basic": "basic", "test_foo_with_arg": "with arg"}

        result = verify_coverage(old, new, mapping)
        # Returns: {"covered": 3, "missing": [], "extra": []}
    """
    old_set = set(old_test_names)
    new_descriptions = {case[-1] for case in new_test_cases if len(case) >= 3}

    if mapping:
        # Map old test names to expected descriptions
        covered = set()
        for old_name in old_set:
            expected_desc = mapping.get(old_name)
            if expected_desc and expected_desc in new_descriptions:
                covered.add(old_name)

        missing = old_set - covered

        # Check for extra cases not in mapping
        expected_descriptions = set(mapping.values())
        extra = new_descriptions - expected_descriptions
    else:
        # Without mapping, just report counts
        covered = set()
        missing = old_set
        extra = new_descriptions

    return {
        "old_count": len(old_set),
        "new_count": len(new_test_cases),
        "covered": covered,
        "missing": list(missing),
        "extra": list(extra),
        "coverage_percent": (len(covered) / len(old_set) * 100) if old_set else 100,
    }
