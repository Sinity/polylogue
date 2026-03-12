"""Pinned regression and record-conversion tests for core models.

Broader semantic ownership lives in ``test_message_laws.py`` and
``test_conversation_semantics.py``. This file keeps only model-specific
regressions and storage-record conversions.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Attachment, Conversation, ConversationSummary, Message
from polylogue.lib.viewports import ToolCall, classify_tool
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord

TOOL_FILE_OPS = [
    ("Read", True),
    ("Write", True),
    ("Edit", True),
    ("NotebookEdit", True),
    ("Bash", False),
]
TOOL_GIT_OPS = [
    ("Bash", {"command": "git commit -m 'test'"}, True),
    ("Read", {"command": "git status"}, False),
    ("Bash", {"command": "ls -la"}, False),
    ("Bash", {"command": 123}, False),
    ("Bash", {"command": "  git push  "}, True),
    ("Bash", {}, False),
]
TOOL_AFFECTED_PATHS = [
    ("Read", {"file_path": "/tmp/test.txt"}, ["/tmp/test.txt"]),
    ("Write", {"file_path": "/tmp/output.txt"}, ["/tmp/output.txt"]),
    ("Edit", {"file_path": "/tmp/code.py"}, ["/tmp/code.py"]),
    ("Read", {"path": "/tmp/fallback.txt"}, ["/tmp/fallback.txt"]),
    ("Read", {"file_path": "/tmp/primary.txt", "path": "/tmp/fallback.txt"}, ["/tmp/primary.txt"]),
    ("Read", {"file_path": 123}, []),
    ("Glob", {"pattern": "**/*.py"}, ["**/*.py"]),
    ("Glob", {"pattern": ["*.py", "*.txt"]}, []),
    ("Bash", {"command": 123}, []),
    ("Task", {"prompt": "do something"}, []),
]
ATTACHMENT_NAME_CASES = [
    ({"name": "file.txt"}, "file.txt"),
    ({"name": 123}, "att1"),
    (None, "att1"),
]
MESSAGE_ROLE_CASES = [
    ("", "unknown"),
    ("   ", "unknown"),
    ("  assistant  ", "assistant"),
]


def _make_tool(tool_name: str, input_data: dict | None = None) -> ToolCall:
    tool_input = input_data or {}
    return ToolCall(name=tool_name, id="t1", input=tool_input, category=classify_tool(tool_name, tool_input))


class TestToolCallProperties:
    @pytest.mark.parametrize(("tool_name", "expected"), TOOL_FILE_OPS)
    def test_is_file_operation(self, tool_name, expected):
        assert _make_tool(tool_name).is_file_operation is expected

    @pytest.mark.parametrize(("tool_name", "input_data", "expected"), TOOL_GIT_OPS)
    def test_is_git_operation(self, tool_name, input_data, expected):
        assert _make_tool(tool_name, input_data).is_git_operation is expected

    @pytest.mark.parametrize(("tool_name", "input_data", "expected"), TOOL_AFFECTED_PATHS)
    def test_affected_paths(self, tool_name, input_data, expected):
        assert _make_tool(tool_name, input_data).affected_paths == expected

    def test_affected_paths_bash_extraction(self):
        tool = _make_tool("Bash", {"command": "ls /tmp/file1 /tmp/file2"})
        assert "/tmp/file1" in tool.affected_paths
        assert "/tmp/file2" in tool.affected_paths

    def test_affected_paths_bash_skips_flags(self):
        tool = _make_tool("Bash", {"command": "ls -la /tmp/file"})
        assert "-la" not in tool.affected_paths
        assert "/tmp/file" in tool.affected_paths


class TestMessageCollectionContracts:
    def test_is_lazy_is_always_false(self):
        assert MessageCollection(messages=[]).is_lazy is False

    def test_materialize_is_noop(self):
        collection = MessageCollection(messages=[Message(id="m1", role="user", text="hello")])
        assert collection.materialize() is collection

    def test_get_pydantic_core_schema(self):
        handler = Mock()
        handler.generate_schema.return_value = {"type": "object"}
        schema = MessageCollection.__get_pydantic_core_schema__(MessageCollection, handler)
        assert schema is not None
        assert hasattr(schema, "__iter__") or isinstance(schema, dict)

    def test_get_pydantic_json_schema(self):
        handler = Mock()
        handler.generate.return_value = {"type": "object"}
        handler.resolve_ref_schema.return_value = {"type": "object"}
        json_schema = MessageCollection.__get_pydantic_json_schema__(Mock(), handler)
        assert json_schema["type"] == "array"
        assert "items" in json_schema


class TestPinnedSemanticRegressions:
    def test_extract_thinking_prefers_db_content_blocks(self):
        msg = Message(
            id="m1",
            role="assistant",
            text="<thinking>xml fallback</thinking>",
            provider_meta={"content_blocks": [{"type": "thinking", "text": "provider meta thought"}]},
            content_blocks=[
                {"type": "thinking", "text": "db thought 1"},
                {"type": "thinking", "text": "db thought 2"},
            ],
        )
        assert msg.extract_thinking() == "db thought 1\n\ndb thought 2"

    def test_extract_thinking_db_blocks_require_thinking_type_and_string_text(self):
        msg = Message(
            id="m1",
            role="assistant",
            text="fallback text",
            content_blocks=[
                {"type": "text", "text": "ignore me"},
                {"type": "thinking", "text": 123},
                {"type": "thinking", "text": "db-only thinking"},
            ],
        )
        assert msg.extract_thinking() == "db-only thinking"

    def test_is_tool_use_detection_raw_claude_code(self):
        msg = Message(
            id="m-tool",
            role="assistant",
            text="I will inspect the file",
            provider="claude-code",
            provider_meta={
                "raw": {
                    "type": "assistant",
                    "uuid": "m-tool",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "I will inspect the file"},
                            {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"file_path": "README.md"}},
                        ],
                    },
                }
            },
        )
        assert msg.is_tool_use is True
        assert msg.harmonized is not None
        assert msg.harmonized.tool_calls[0].id == "tool-1"
        assert msg.harmonized.tool_calls[0].input == {"file_path": "README.md"}

    def test_is_thinking_detection_raw_claude_code(self):
        msg = Message(
            id="m-think",
            role="assistant",
            text="",
            provider="claude-code",
            provider_meta={
                "raw": {
                    "type": "assistant",
                    "uuid": "m-think",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "thinking", "thinking": "step by step"}],
                    },
                }
            },
        )
        assert msg.is_thinking is True
        assert msg.harmonized is not None
        assert msg.harmonized.reasoning_traces[0].text == "step by step"


class TestAttachmentFromRecord:
    @pytest.mark.parametrize(("provider_meta", "expected_name"), ATTACHMENT_NAME_CASES)
    def test_from_record_derives_name(self, provider_meta, expected_name):
        record = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
            provider_meta=provider_meta,
        )
        attachment = Attachment.from_record(record)
        assert attachment.name == expected_name


class TestMessageFromRecord:
    @pytest.mark.parametrize(("role", "expected_role"), MESSAGE_ROLE_CASES)
    def test_from_record_normalizes_role(self, role, expected_role):
        record = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role=role,
            text="hello",
            timestamp="2024-01-01T00:00:00Z",
            content_hash="hash1",
        )
        message = Message.from_record(record, [])
        assert message.role == expected_role


class TestConversationSummaryFromRecord:
    def test_from_record_projects_metadata(self):
        record = ConversationRecord(
            conversation_id="c1",
            provider_name="claude",
            provider_conversation_id="prov-c1",
            content_hash="hash1",
            title="Test",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
            provider_meta={"key": "value"},
            metadata={"tags": ["test"], "summary": "summary"},
        )
        summary = ConversationSummary.from_record(record)
        assert summary.id == "c1"
        assert summary.provider == "claude"
        assert summary.title == "Test"
        assert summary.tags == ["test"]
        assert summary.summary == "summary"


class TestConversationFromRecords:
    def test_from_records_attaches_records_to_messages(self):
        conversation_record = ConversationRecord(
            conversation_id="c1",
            provider_name="claude",
            provider_conversation_id="prov-c1",
            content_hash="hash-c1",
            title="Test",
        )
        message_record = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role="user",
            text="hello",
            timestamp="2024-01-01T00:00:00Z",
            content_hash="hash-m1",
        )
        attachment_record = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
            message_id="m1",
            provider_meta={"name": "file.txt"},
        )

        conversation = Conversation.from_records(conversation_record, [message_record], [attachment_record])

        assert conversation.id == "c1"
        assert conversation.provider == "claude"
        assert len(conversation.messages) == 1
        assert conversation.messages[0].role == "user"
        assert [attachment.name for attachment in conversation.messages[0].attachments] == ["file.txt"]
