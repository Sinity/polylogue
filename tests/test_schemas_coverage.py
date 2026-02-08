from datetime import datetime, timezone

from polylogue.lib.roles import Role
from polylogue.schemas.claude_code_records import (
    FileHistorySnapshot,
    ProgressRecord,
    QueueOperationRecord,
    RecordType,
    classify_record,
    extract_metadata_record,
)
from polylogue.schemas.common import CommonMessage, CommonToolCall


def test_record_type_enums():
    assert RecordType.is_message("user") is True
    assert RecordType.is_message("assistant") is True
    assert RecordType.is_message("system") is True
    assert RecordType.is_message("progress") is False

    assert RecordType.is_metadata("progress") is True
    assert RecordType.is_metadata("user") is False


def test_progress_record_from_raw():
    raw = {
        "type": "progress",
        "data": {"hookEvent": "SessionStart", "hookName": "on_session_start"},
        "toolUseID": "tool_123",
        "parentToolUseID": "parent_456",
        "timestamp": "2023-01-01T12:00:00Z",
        "sessionId": "session_abc",
    }
    record = ProgressRecord.from_raw(raw)
    assert record.hook_event == "SessionStart"
    assert record.hook_name == "on_session_start"
    assert record.tool_use_id == "tool_123"
    assert record.parent_tool_use_id == "parent_456"
    assert record.timestamp == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert record.session_id == "session_abc"
    assert record.raw == raw


def test_file_history_snapshot_from_raw():
    raw = {
        "type": "file-history-snapshot",
        "messageId": "msg_123",
        "snapshot": {
            "timestamp": "2023-01-01T12:00:00Z",
            "trackedFileBackups": {
                "/path/to/file1": {"hash": "hash1"},
                "/path/to/file2": None,  # Case where hash is missing or structure differs
            },
        },
        "isSnapshotUpdate": True,
    }
    snapshot = FileHistorySnapshot.from_raw(raw)
    assert snapshot.message_id == "msg_123"
    assert snapshot.timestamp == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert snapshot.is_snapshot_update is True
    assert len(snapshot.tracked_files) == 2
    assert snapshot.tracked_files[0].path == "/path/to/file1"
    assert snapshot.tracked_files[0].content_hash == "hash1"
    assert snapshot.tracked_files[1].path == "/path/to/file2"
    assert snapshot.tracked_files[1].content_hash is None


def test_queue_operation_record_from_raw():
    raw = {
        "type": "queue-operation",
        "operation": "enqueue",
        "timestamp": "2023-01-01T12:00:00Z",
        "sessionId": "session_abc",
        "content": {"foo": "bar"},
    }
    record = QueueOperationRecord.from_raw(raw)
    assert record.operation == "enqueue"
    assert record.timestamp == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert record.session_id == "session_abc"
    assert record.content == {"foo": "bar"}


def test_extract_metadata_record_dispatch():
    # Progress
    raw_progress = {"type": "progress", "data": {}}
    assert isinstance(extract_metadata_record(raw_progress), ProgressRecord)

    # File Snapshot
    raw_snapshot = {"type": "file-history-snapshot", "snapshot": {}}
    assert isinstance(extract_metadata_record(raw_snapshot), FileHistorySnapshot)

    # Queue Op
    raw_queue = {"type": "queue-operation"}
    assert isinstance(extract_metadata_record(raw_queue), QueueOperationRecord)

    # Message type (should return None)
    raw_message = {"type": "user"}
    assert extract_metadata_record(raw_message) is None

    # Unknown type
    raw_unknown = {"type": "unknown_thing"}
    assert extract_metadata_record(raw_unknown) is None


def test_classify_record():
    assert classify_record({"type": "user"}) == ("message", "user")
    assert classify_record({"type": "progress"}) == ("metadata", "progress")
    assert classify_record({"type": "unknown"}) == ("metadata", "unknown")


def test_common_message_instantiation():
    msg = CommonMessage(
        role=Role.USER,
        text="Hello",
        timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        id="msg_1",
        model="gpt-4",
        tokens=10,
        cost_usd=0.01,
        is_thinking=True,
        provider="test_provider",
        raw={"orig": "data"},
    )
    assert msg.role == Role.USER
    assert msg.text == "Hello"
    assert msg.timestamp is not None
    assert msg.id == "msg_1"
    assert msg.is_thinking is True


def test_common_tool_call_instantiation():
    tool = CommonToolCall(
        name="calculator",
        input={"a": 1, "b": 2},
        output="3",
        success=True,
        provider="test_provider",
        raw={"orig": "data"},
    )
    assert tool.name == "calculator"
    assert tool.input == {"a": 1, "b": 2}
    assert tool.output == "3"
    assert tool.success is True
