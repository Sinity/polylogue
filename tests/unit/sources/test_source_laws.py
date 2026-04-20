"""Law-based contracts for source detection, dispatch, and JSON iteration."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import Path
from typing import IO, BinaryIO
from unittest.mock import MagicMock

import ijson
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from typing_extensions import TypedDict

from polylogue.config import Source
from polylogue.lib.roles import Role, normalize_role
from polylogue.schemas.json_types import JSONDocument
from polylogue.sources import decoders as decoders_module
from polylogue.sources import dispatch as dispatch_module
from polylogue.sources import source_acquisition
from polylogue.sources.cursor import (
    _get_file_mtime,
    _initialize_cursor_state,
    _log_source_iteration_summary,
    _ParseContext,
    _record_cursor_failure,
    _select_paths_for_processing,
)
from polylogue.sources.decoders import (
    _decode_json_bytes,
    _iter_json_stream,
    _zip_entry_provider_hint,
    _ZipEntryValidator,
)
from polylogue.sources.dispatch import (
    detect_provider,
    parse_drive_payload,
    parse_payload,
)
from polylogue.sources.drive import (
    download_drive_files,
    drive_cache_file_path,
    iter_drive_raw_data,
)
from polylogue.sources.drive_types import DriveFile
from polylogue.sources.emitter import _ConversationEmitter
from polylogue.sources.parsers import chatgpt as chatgpt_parser
from polylogue.sources.parsers import claude as claude_parser
from polylogue.sources.parsers import drive as drive_parser
from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage
from polylogue.sources.parsers.claude import (
    SessionIndexEntry,
    enrich_conversation_from_index,
    find_sessions_index,
    parse_sessions_index,
)
from polylogue.sources.providers.chatgpt import (
    ChatGPTAuthor,
    ChatGPTContent,
    ChatGPTConversation,
    ChatGPTMessage,
    ChatGPTNode,
)
from polylogue.sources.providers.claude_code import (
    ClaudeCodeThinkingBlock,
    ClaudeCodeToolUse,
    ClaudeCodeUsage,
)
from polylogue.sources.source_acquisition import _iter_entry_payloads, iter_source_raw_data
from polylogue.sources.source_parsing import (
    iter_source_conversations,
    iter_source_conversations_with_raw,
)
from polylogue.sources.source_walk import _has_supported_extension
from polylogue.storage.blob_store import BlobStore, Heartbeat
from polylogue.storage.state_views import CursorFailurePayload, CursorStatePayload
from polylogue.types import Provider
from tests.infra.source_builders import GenericConversationBuilder, make_claude_chat_message
from tests.infra.strategies import (
    conversations_wrapper_bytes_strategy,
    json_array_bytes_strategy,
    json_document_strategy,
    jsonl_bytes_strategy,
    provider_export_strategy,
    provider_payload_case_strategy,
    provider_payload_strategy,
    provider_source_case_strategy,
)


class GeneratedSourceCase(TypedDict):
    path: Path
    raw: bytes
    provider: str


class FailedFile(TypedDict):
    path: str
    error: str


def _require_generated_source_case(value: object) -> GeneratedSourceCase:
    if not isinstance(value, dict):
        raise AssertionError(f"expected generated source case, got {type(value).__name__}")
    path = value.get("path")
    raw = value.get("raw")
    provider = value.get("provider")
    if not isinstance(path, Path):
        raise AssertionError(f"expected generated path, got {type(path).__name__}")
    if not isinstance(raw, bytes):
        raise AssertionError(f"expected generated raw bytes, got {type(raw).__name__}")
    if not isinstance(provider, str):
        raise AssertionError(f"expected generated provider, got {type(provider).__name__}")
    return GeneratedSourceCase(path=path, raw=raw, provider=provider)


def _parsed_message(
    provider_message_id: str,
    *,
    role: str | Role,
    text: str | None,
) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_message_id,
        role=Role.normalize(role) if isinstance(role, str) else role,
        text=text,
    )


def _parsed_conversation(
    *,
    provider_name: str | Provider,
    provider_conversation_id: str,
    title: str | None,
    created_at: str | None,
    updated_at: str | None,
    messages: list[ParsedMessage],
    provider_meta: dict[str, object] | None = None,
) -> ParsedConversation:
    return ParsedConversation(
        provider_name=Provider.from_string(provider_name),
        provider_conversation_id=provider_conversation_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=messages,
        provider_meta=provider_meta,
    )


def _failed_files(cursor_state: CursorStatePayload) -> list[FailedFile]:
    failed_files: list[CursorFailurePayload] = cursor_state.get("failed_files", [])
    return [FailedFile(path=str(item["path"]), error=str(item["error"])) for item in failed_files]


def _numeric_observation_value(observation: Mapping[str, object], key: str) -> float:
    value = observation.get(key)
    if not isinstance(value, (int, float, str)):
        raise AssertionError(f"expected numeric observation field {key}, got {type(value).__name__}")
    return float(value)


_CANONICAL_PROVIDERS = (
    Provider.CHATGPT.value,
    Provider.CLAUDE_AI.value,
    Provider.CLAUDE_CODE.value,
    Provider.CODEX.value,
    Provider.GEMINI.value,
)


def _empty_cursor_state() -> CursorStatePayload:
    return {}


@given(
    provider_payload_case_strategy(
        (
            Provider.CHATGPT.value,
            Provider.CLAUDE_AI.value,
            Provider.CLAUDE_CODE.value,
            Provider.CODEX.value,
            Provider.GEMINI.value,
        )
    )
)
@settings(max_examples=40)
def test_detect_provider_recognizes_generated_payloads(case: tuple[str, object]) -> None:
    """Generated provider payloads are self-identifying without filename hints."""
    provider, payload = case
    assert detect_provider(payload, Path("unknown.json")) == provider


def test_detect_provider_returns_none_for_unknown_payloads_without_shape_match() -> None:
    assert detect_provider({"unrelated": True}, Path("chatgpt-export.json")) is None


def test_detect_provider_prefers_payload_shape_over_conflicting_path_hint() -> None:
    payload = {
        "mapping": {
            "node-1": {
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hello"]},
                }
            }
        }
    }

    assert detect_provider(payload, Path("misleading/claude-code/session.jsonl")) == Provider.CHATGPT


@given(provider_payload_case_strategy(_CANONICAL_PROVIDERS))
@settings(max_examples=35)
def test_parse_payload_generated_exports_produce_provider_named_conversations(case: tuple[str, object]) -> None:
    """Runtime dispatch parses generated provider payloads into provider-owned conversations."""
    provider, payload = case
    conversations = parse_payload(provider, payload, "fallback-id")
    assert conversations
    assert all(str(conversation.provider_name) == provider for conversation in conversations)
    assert all(conversation.provider_conversation_id for conversation in conversations)


def test_parse_payload_accepts_provider_enum() -> None:
    payload = {
        "mapping": {
            "root": {
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hello"]},
                },
                "children": [],
            }
        },
        "current_node": "root",
        "title": "Test",
    }

    conversations = parse_payload(Provider.CHATGPT, payload, "fallback-id")

    assert len(conversations) == 1
    assert conversations[0].provider_name is Provider.CHATGPT


@pytest.mark.parametrize(
    ("provider", "wrapper"),
    [
        (Provider.CHATGPT.value, False),
        (Provider.CLAUDE_AI.value, False),
        (Provider.GEMINI.value, False),
        (Provider.CHATGPT.value, True),
    ],
    ids=["chatgpt-bundle", "claude-bundle", "gemini-bundle", "conversations-wrapper"],
)
@given(data=st.data())
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_parse_payload_bundle_cardinality_contract(
    provider: str,
    wrapper: bool,
    data: st.DataObject,
) -> None:
    """Bundle dispatch preserves one parsed conversation per bundled payload."""
    payloads = data.draw(st.lists(provider_payload_strategy(provider), min_size=1, max_size=4))
    payload = {"conversations": payloads} if wrapper else payloads
    conversations = parse_payload(provider, payload, "bundle")
    assert len(conversations) == len(payloads)


@given(
    json_document_strategy(),
    st.sampled_from(("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")),
)
@settings(max_examples=35)
def test_decode_json_bytes_round_trips_supported_encodings(document: dict[str, object], encoding: str) -> None:
    """Supported encodings decode back into the original JSON document."""
    blob = json.dumps(document).encode(encoding)
    decoded = _decode_json_bytes(blob)
    assert decoded is not None
    assert json.loads(decoded) == document


@given(json_array_bytes_strategy())
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_iter_json_stream_root_list_round_trips_documents(case: tuple[list[dict[str, object]], bytes]) -> None:
    """Streaming a root JSON array yields the original item sequence."""
    documents, raw = case
    assert list(_iter_json_stream(BytesIO(raw), "test.json")) == documents


@given(conversations_wrapper_bytes_strategy())
@settings(max_examples=30)
def test_iter_json_stream_conversations_wrapper_round_trips_documents(
    case: tuple[list[dict[str, object]], bytes],
) -> None:
    """Streaming a `{\"conversations\": [...]}` object yields the wrapped items."""
    documents, raw = case
    assert list(_iter_json_stream(BytesIO(raw), "test.json")) == documents


@given(json_array_bytes_strategy())
@settings(max_examples=30)
def test_iter_json_stream_unpack_lists_false_preserves_single_list(case: tuple[list[dict[str, object]], bytes]) -> None:
    """`unpack_lists=False` keeps the JSON root list intact as one item."""
    documents, raw = case
    assert list(_iter_json_stream(BytesIO(raw), "test.json", unpack_lists=False)) == [documents]


@given(jsonl_bytes_strategy())
@settings(max_examples=35)
def test_iter_json_stream_jsonl_preserves_valid_records_with_blank_lines(
    case: tuple[list[dict[str, object]], bytes],
) -> None:
    """JSONL parsing ignores blank lines but preserves valid record order exactly."""
    documents, raw = case
    assert list(_iter_json_stream(BytesIO(raw), "test.jsonl")) == documents


def test_iter_json_stream_jsonl_invalid_line_logging_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    # 4 broken lines: first 3 (non-trailing) get warning, last gets debug (truncation tolerance)
    raw = b'{"id": 1}\n{broken}\n{broken}\n{broken}\n{broken}\n'
    warnings: list[str] = []
    debugs: list[str] = []

    monkeypatch.setattr(
        decoders_module.logger,
        "warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )
    monkeypatch.setattr(
        decoders_module.logger,
        "debug",
        lambda message, *args: debugs.append(message % args if args else message),
    )

    items = list(_iter_json_stream(BytesIO(raw), "test.jsonl"))

    assert items == [{"id": 1}]
    # First 3 non-trailing broken lines get warning level
    assert len(warnings) == 3
    assert all("Skipping invalid JSON line in test.jsonl" in w for w in warnings)
    # Trailing broken line gets debug (in-progress file truncation tolerance)
    assert any("Skipping truncated trailing line in test.jsonl" in d for d in debugs)


def test_iter_json_stream_falls_back_to_full_json_load_when_streaming_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def broken_items(handle: BytesIO, prefix: str) -> Iterable[object]:
        del handle
        calls.append(prefix)
        raise ijson.common.JSONError("boom")

    monkeypatch.setattr(ijson, "items", broken_items)

    raw = b'{"conversations":[{"id":"one"},{"id":"two"}]}'
    items = list(_iter_json_stream(BytesIO(raw), "test.json"))

    assert calls == ["item", "conversations.item"]
    assert items == [{"conversations": [{"id": "one"}, {"id": "two"}]}]


def _materialize_generated_source(root: Path, *, hint_path: Path, raw: bytes, use_zip: bool) -> Source:
    """Write generated provider bytes either directly or inside a ZIP archive."""
    if use_zip:
        archive_path = root / "generated.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr(hint_path.as_posix(), raw)
        return Source(name="generated", path=archive_path)

    payload_path = root / hint_path
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(raw)
    return Source(name="generated", path=payload_path)


def _write_generic_conversation(path: Path, conversation_id: str, text: str = "hello") -> Path:
    GenericConversationBuilder(conversation_id).title(conversation_id).add_message(
        "user",
        f"{conversation_id}-message",
        text=text,
    ).write_to(path)
    return path


@given(
    provider_source_case_strategy(
        providers=(
            Provider.CHATGPT.value,
            Provider.CLAUDE_AI.value,
            Provider.CLAUDE_CODE.value,
            Provider.CODEX.value,
            Provider.GEMINI.value,
        )
    ),
    st.booleans(),
)
@settings(max_examples=30, deadline=None)
def test_iter_source_conversations_round_trips_generated_exports(case: object, use_zip: bool) -> None:
    """Generated provider exports should stay discoverable through file and ZIP iteration."""
    generated = _require_generated_source_case(case)
    with tempfile.TemporaryDirectory() as tmp:
        source = _materialize_generated_source(
            Path(tmp),
            hint_path=generated["path"],
            raw=generated["raw"],
            use_zip=use_zip,
        )
        conversations = list(iter_source_conversations(source))

    assert conversations
    assert all(str(conversation.provider_name) == generated["provider"] for conversation in conversations)


@given(
    provider_source_case_strategy(
        providers=(
            Provider.CHATGPT.value,
            Provider.CLAUDE_AI.value,
            Provider.CLAUDE_CODE.value,
            Provider.CODEX.value,
            Provider.GEMINI.value,
        )
    ),
    st.booleans(),
    st.booleans(),
)
@settings(max_examples=30, deadline=None)
def test_iter_source_conversations_with_raw_capture_contract(
    case: object,
    use_zip: bool,
    capture_raw: bool,
) -> None:
    """Raw iteration must preserve provider parsing while toggling raw payload capture cleanly."""
    generated = _require_generated_source_case(case)
    with tempfile.TemporaryDirectory() as tmp:
        source = _materialize_generated_source(
            Path(tmp),
            hint_path=generated["path"],
            raw=generated["raw"],
            use_zip=use_zip,
        )
        items = list(iter_source_conversations_with_raw(source, capture_raw=capture_raw))

    assert items
    assert all(str(conversation.provider_name) == generated["provider"] for _, conversation in items)
    if capture_raw:
        assert all(raw_data is not None for raw_data, _ in items)
        assert all(raw_data.raw_bytes or raw_data.blob_hash for raw_data, _ in items if raw_data is not None)
        assert all(raw_data.file_mtime is not None for raw_data, _ in items if raw_data is not None)
    else:
        assert all(raw_data is None for raw_data, _ in items)


@given(
    st.sampled_from((Provider.CLAUDE_CODE.value, Provider.CODEX.value)).flatmap(
        lambda provider: st.tuples(
            st.just(provider),
            provider_export_strategy(provider),
            st.sampled_from(("session.jsonl", "session.ndjson", "session.jsonl.txt")),
        )
    )
)
@settings(max_examples=20, deadline=None)
def test_iter_source_conversations_accepts_grouped_json_extensions(case: tuple[str, bytes, str]) -> None:
    """Grouped JSONL providers must remain discoverable through all supported session suffixes."""
    provider, raw, filename = case
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        path.write_bytes(raw)
        conversations = list(iter_source_conversations(Source(name=provider, path=path)))

    assert conversations
    assert all(str(conversation.provider_name) == provider for conversation in conversations)


def test_parse_payload_depth_guard_contract() -> None:
    assert parse_payload(Provider.CHATGPT.value, {}, "test", _depth=11) == []


def test_source_iteration_preserves_claude_attachment_metadata_contract(tmp_path: Path) -> None:
    payload = {
        "chat_messages": [
            make_claude_chat_message(
                "msg-1",
                "assistant",
                "Files",
                attachments=[{"id": "file-1", "name": "notes.txt", "size": 12, "mimeType": "text/plain"}],
            )
        ]
    }
    source_file = tmp_path / "claude.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    conversations = list(iter_source_conversations(Source(name="inbox", path=source_file)))

    assert len(conversations) == 1
    attachment = conversations[0].attachments[0]
    assert attachment.provider_attachment_id == "file-1"
    assert attachment.name == "notes.txt"


def test_iter_source_conversations_handles_empty_directories_contract(tmp_path: Path) -> None:
    cursor_state: CursorStatePayload = _empty_cursor_state()

    assert list(iter_source_conversations(Source(name="test", path=tmp_path), cursor_state=cursor_state)) == []
    assert cursor_state["file_count"] == 0
    assert cursor_state["failed_count"] == 0


@pytest.mark.parametrize("iterator", ["parsed", "raw"], ids=["parsed", "raw"])
def test_source_iteration_continues_after_invalid_json_contract(tmp_path: Path, iterator: str) -> None:
    _write_generic_conversation(tmp_path / "valid1.json", "valid1", "hi")
    (tmp_path / "invalid.json").write_text("{ broken json", encoding="utf-8")
    _write_generic_conversation(tmp_path / "valid2.json", "valid2", "bye")

    cursor_state: CursorStatePayload = _empty_cursor_state()
    if iterator == "parsed":
        results = list(iter_source_conversations(Source(name="test", path=tmp_path), cursor_state=cursor_state))
        conversation_ids = [conversation.provider_conversation_id for conversation in results]
        raw_items = []
    else:
        raw_items = list(
            iter_source_conversations_with_raw(Source(name="test", path=tmp_path), cursor_state=cursor_state)
        )
        results = [conversation for _, conversation in raw_items]
        conversation_ids = [conversation.provider_conversation_id for conversation in results]

    assert conversation_ids == ["valid1", "valid2"]
    assert cursor_state["file_count"] == 3
    assert cursor_state["failed_count"] == 1
    assert any("invalid.json" in str(item["path"]) for item in _failed_files(cursor_state))
    if iterator == "raw":
        assert all(raw_data is not None for raw_data, _ in raw_items)


def test_iter_source_conversations_tracks_file_disappearance_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _write_generic_conversation(tmp_path / "first.json", "first")
    second = _write_generic_conversation(tmp_path / "second.json", "second")
    cursor_state: CursorStatePayload = _empty_cursor_state()
    original_open = Path.open

    def flaky_open(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[str] | BinaryIO:
        if path == second:
            raise FileNotFoundError("deleted")
        return original_open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "open", flaky_open)

    conversations = list(iter_source_conversations(Source(name="test", path=tmp_path), cursor_state=cursor_state))

    assert [conversation.provider_conversation_id for conversation in conversations] == ["first"]
    assert cursor_state["failed_count"] == 1
    assert any("second.json" in str(item["path"]) for item in _failed_files(cursor_state))
    assert first.exists()


@pytest.mark.parametrize("skip_dir_name", ["analysis", "__pycache__"])
def test_source_iteration_prunes_skip_dirs_contract(tmp_path: Path, skip_dir_name: str) -> None:
    skip_dir = tmp_path / skip_dir_name
    skip_dir.mkdir()
    _write_generic_conversation(skip_dir / "skipped.json", "skipped")

    assert list(iter_source_conversations(Source(name="test", path=tmp_path))) == []
    assert list(iter_source_conversations_with_raw(Source(name="test", path=tmp_path))) == []


def test_source_iteration_follows_symlinked_directories_contract(tmp_path: Path) -> None:
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    _write_generic_conversation(subdir / "linked.json", "linked")

    link = tmp_path / "link"
    try:
        link.symlink_to(subdir)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this system")

    parsed = list(iter_source_conversations(Source(name="test", path=link)))
    raw = list(iter_source_conversations_with_raw(Source(name="test", path=link)))

    assert [conversation.provider_conversation_id for conversation in parsed] == ["linked"]
    assert [conversation.provider_conversation_id for _, conversation in raw] == ["linked"]


def test_iter_source_conversations_with_raw_accepts_single_file_sources_contract(tmp_path: Path) -> None:
    source_file = _write_generic_conversation(tmp_path / "single.json", "single")

    items = list(iter_source_conversations_with_raw(Source(name="test", path=source_file)))

    assert len(items) == 1
    raw_data, conversation = items[0]
    assert raw_data is not None
    assert raw_data.source_path == str(source_file)
    assert conversation.provider_conversation_id == "single"


@pytest.mark.parametrize(
    ("source_name", "filename", "use_zip", "needle"),
    [
        ("claude-code", "session.jsonl", False, b"Hello"),
        ("claude-code", "session.jsonl", True, b"From zip"),
    ],
    ids=["plain-jsonl", "zip-jsonl"],
)
def test_iter_source_conversations_with_raw_preserves_grouped_bytes_contract(
    tmp_path: Path,
    source_name: str,
    filename: str,
    use_zip: bool,
    needle: bytes,
) -> None:
    from polylogue.storage.blob_store import get_blob_store

    records = [
        {"type": "user", "uuid": "u1", "sessionId": "s1", "message": {"content": needle.decode("utf-8")}},
        {"type": "assistant", "uuid": "a1", "sessionId": "s1", "message": {"content": "Hi"}},
    ]
    content = "\n".join(json.dumps(record) for record in records) + "\n"

    if use_zip:
        source_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(source_path, "w") as zf:
            zf.writestr(filename, content)
    else:
        source_path = tmp_path / filename
        source_path.write_text(content, encoding="utf-8")

    items = list(iter_source_conversations_with_raw(Source(name=source_name, path=source_path)))

    assert len(items) == 1
    raw_data, conversation = items[0]
    assert raw_data is not None
    assert raw_data.source_index is None
    raw_bytes = raw_data.raw_bytes
    if not raw_bytes and raw_data.blob_hash is not None:
        raw_bytes = get_blob_store().read_all(raw_data.blob_hash)
    assert needle in raw_bytes
    assert conversation.provider_name == Provider.CLAUDE_CODE


def test_iter_source_conversations_with_raw_streams_plain_grouped_capture_to_blob_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.blob_store import get_blob_store

    source_path = tmp_path / "session.jsonl"
    content = (
        '{"type":"user","uuid":"u1","sessionId":"s1","message":{"content":"hello"}}\n'
        '{"type":"assistant","uuid":"a1","sessionId":"s1","message":{"content":"hi"}}\n'
    )
    source_path.write_text(content, encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def fail_read_bytes(path: Path) -> bytes:
        if path == source_path:
            raise AssertionError("Plain grouped parsing should not read whole files via Path.read_bytes")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    items = list(iter_source_conversations_with_raw(Source(name="claude-code", path=source_path)))

    assert len(items) == 1
    raw_data, conversation = items[0]
    assert raw_data is not None
    assert raw_data.blob_hash is not None
    assert raw_data.raw_bytes == b""
    assert get_blob_store().read_all(raw_data.blob_hash) == content.encode("utf-8")
    assert conversation.provider_name == Provider.CLAUDE_CODE


def test_iter_source_conversations_with_raw_streams_grouped_zip_capture_to_blob_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.blob_store import BlobStore, get_blob_store

    archive_path = tmp_path / "bundle.zip"
    content = (
        '{"type":"user","uuid":"u1","sessionId":"s1","message":{"content":"hello"}}\n'
        '{"type":"assistant","uuid":"a1","sessionId":"s1","message":{"content":"hi"}}\n'
    )
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("nested/session.jsonl", content)

    write_calls = 0
    original_write_from_fileobj = BlobStore.write_from_fileobj

    def tracking_write_from_fileobj(self: BlobStore, source: BinaryIO) -> tuple[str, int]:
        nonlocal write_calls
        write_calls += 1
        return original_write_from_fileobj(self, source)

    monkeypatch.setattr(BlobStore, "write_from_fileobj", tracking_write_from_fileobj)

    items = list(iter_source_conversations_with_raw(Source(name="claude-code", path=archive_path)))

    assert write_calls == 1
    assert len(items) == 1
    raw_data, conversation = items[0]
    assert raw_data is not None
    assert raw_data.blob_hash is not None
    assert raw_data.raw_bytes == b""
    assert get_blob_store().read_all(raw_data.blob_hash) == content.encode("utf-8")
    assert conversation.provider_name == Provider.CLAUDE_CODE


def test_iter_source_conversations_with_raw_assigns_source_indexes_for_multi_conversation_zip_contract(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "multi.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(
            "data.json",
            json.dumps(
                [
                    {"id": "c1", "messages": [{"id": "m1", "role": "user", "text": "Q1"}]},
                    {"id": "c2", "messages": [{"id": "m2", "role": "user", "text": "Q2"}]},
                ]
            ),
        )

    results = list(iter_source_conversations_with_raw(Source(name="test", path=archive_path)))

    assert len(results) == 2
    assert [raw_data.source_index for raw_data, _ in results if raw_data is not None] == [0, 1]


def test_iter_source_conversations_with_raw_tracks_unicode_decode_failures_contract(
    tmp_path: Path,
) -> None:
    bad_file = tmp_path / "bad_encoding.json"
    bad_file.write_bytes(b"\xff\xfe invalid utf-8 { bad json")

    cursor_state: CursorStatePayload = _empty_cursor_state()
    results = list(iter_source_conversations_with_raw(Source(name="test", path=tmp_path), cursor_state=cursor_state))

    assert results == []
    assert cursor_state["failed_count"] == 1
    assert any("bad_encoding.json" in str(item["path"]) for item in _failed_files(cursor_state))


def test_source_iteration_ignores_stat_failures_for_optional_mtime_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_file = _write_generic_conversation(source_dir / "conv.json", "conv")
    original_stat = Path.stat

    def flaky_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == source_file:
            raise OSError("no stat")
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    parsed_cursor: CursorStatePayload = _empty_cursor_state()
    raw_items = list(iter_source_conversations_with_raw(Source(name="test", path=source_dir), capture_raw=True))
    parsed_items = list(iter_source_conversations(Source(name="test", path=source_dir), cursor_state=parsed_cursor))

    assert len(raw_items) == 1
    assert raw_items[0][0] is not None and raw_items[0][0].file_mtime is None
    assert [conversation.provider_conversation_id for conversation in parsed_items] == ["conv"]
    assert parsed_cursor["file_count"] == 1
    assert "latest_mtime" not in parsed_cursor


@pytest.mark.parametrize(
    ("provider", "payload", "expected_provider", "expected_count"),
    [
        (
            "drive",
            {"id": "generic", "messages": [{"id": "m1", "role": "user", "text": "hello"}]},
            "drive",
            1,
        ),
        (
            "drive",
            {"mapping": {}, "conversation_id": "chatgpt-ish", "id": "chatgpt-ish"},
            "chatgpt",
            1,
        ),
        (
            "gemini",
            [{"role": "user", "text": "hello"}, {"role": "model", "text": "hi"}],
            "gemini",
            1,
        ),
    ],
)
def test_parse_drive_payload_contract(
    provider: str,
    payload: object,
    expected_provider: str,
    expected_count: int,
) -> None:
    conversations = parse_drive_payload(provider, payload, "fallback")
    assert len(conversations) >= expected_count
    assert all(conversation.provider_name == expected_provider for conversation in conversations)


def test_parse_payload_generic_messages_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_messages = [_parsed_message("m1", role="user", text="hello")]

    monkeypatch.setattr(dispatch_module, "extract_messages_from_list", lambda messages: sentinel_messages)

    conversations = parse_payload(
        Provider.DRIVE.value,
        {"id": "conv-1", "name": "Named", "messages": [{"ignored": True}]},
        "fallback",
    )

    assert len(conversations) == 1
    assert conversations[0].provider_name == Provider.DRIVE
    assert conversations[0].provider_conversation_id == "conv-1"
    assert conversations[0].title == "Named"
    assert conversations[0].messages == sentinel_messages


def test_parse_payload_dispatches_chatgpt_bundle_items_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, str]] = []

    def fake_parse(payload: object, fallback_id: str) -> ParsedConversation:
        calls.append((payload, fallback_id))
        return _parsed_conversation(
            provider_name=Provider.CHATGPT,
            provider_conversation_id=fallback_id,
            title=fallback_id,
            created_at=None,
            updated_at=None,
            messages=[],
        )

    monkeypatch.setattr(chatgpt_parser, "parse", fake_parse)
    payloads = [{"id": "one"}, {"id": "two"}]

    conversations = parse_payload(Provider.CHATGPT.value, payloads, "bundle")

    assert [conversation.provider_conversation_id for conversation in conversations] == ["bundle-0", "bundle-1"]
    assert calls == [(payloads[0], "bundle-0"), (payloads[1], "bundle-1")]


def test_parse_payload_dispatches_claude_code_messages_and_single_records(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, str]] = []

    def fake_parse_code(payload: object, fallback_id: str) -> ParsedConversation:
        calls.append((payload, fallback_id))
        return _parsed_conversation(
            provider_name=Provider.CLAUDE_CODE,
            provider_conversation_id=fallback_id,
            title=fallback_id,
            created_at=None,
            updated_at=None,
            messages=[],
        )

    monkeypatch.setattr(claude_parser, "parse_code", fake_parse_code)

    from_messages = parse_payload(
        Provider.CLAUDE_CODE.value,
        {"messages": [{"type": "user"}, {"type": "assistant"}]},
        "session",
    )
    from_single = parse_payload(
        Provider.CLAUDE_CODE.value,
        {"type": "assistant", "message": {"content": "hi"}},
        "single",
    )

    assert [conversation.provider_conversation_id for conversation in from_messages] == ["session"]
    assert [conversation.provider_conversation_id for conversation in from_single] == ["single"]
    assert calls == [
        ([{"type": "user"}, {"type": "assistant"}], "session"),
        ([{"type": "assistant", "message": {"content": "hi"}}], "single"),
    ]


def test_parse_drive_payload_recurses_lists_and_detected_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    drive_calls: list[tuple[str, object, str]] = []
    parse_calls: list[tuple[str, object, str]] = []

    def fake_chunked(provider: str, payload: object, fallback_id: str) -> ParsedConversation:
        drive_calls.append((provider, payload, fallback_id))
        return _parsed_conversation(
            provider_name=provider,
            provider_conversation_id=fallback_id,
            title=fallback_id,
            created_at=None,
            updated_at=None,
            messages=[],
        )

    def fake_parse_payload(
        provider: str,
        payload: object,
        fallback_id: str,
        _depth: int = 0,
    ) -> list[ParsedConversation]:
        parse_calls.append((provider, payload, fallback_id))
        return [
            _parsed_conversation(
                provider_name=provider,
                provider_conversation_id=fallback_id,
                title=fallback_id,
                created_at=None,
                updated_at=None,
                messages=[],
            )
        ]

    monkeypatch.setattr(drive_parser, "parse_chunked_prompt", fake_chunked)
    monkeypatch.setattr(dispatch_module, "parse_payload", fake_parse_payload)
    monkeypatch.setattr(dispatch_module, "detect_provider", lambda payload, path=None: Provider.CHATGPT)

    chunked = parse_drive_payload("gemini", [{"role": "user", "text": "hello"}], "chunks")
    recursive = parse_drive_payload("drive", [{"mapping": {}, "id": "chatgpt-ish"}], "wrapped")

    assert [conversation.provider_conversation_id for conversation in chunked] == ["chunks"]
    assert drive_calls == [("gemini", {"chunks": [{"role": "user", "text": "hello"}]}, "chunks")]
    assert [conversation.provider_conversation_id for conversation in recursive] == ["wrapped-0"]
    assert parse_calls == [("chatgpt", {"mapping": {}, "id": "chatgpt-ish"}, "wrapped-0")]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b'\xef\xbb\xbf{"id":"bom"}', {"id": "bom"}),
        (b'{\x00"id":"nulls"}', {"id": "nulls"}),
        (b"", None),
    ],
)
def test_decode_json_bytes_cleaning_contract(raw: bytes, expected: dict[str, object] | None) -> None:
    decoded = _decode_json_bytes(raw)
    if expected is None:
        assert decoded is None
    else:
        assert decoded is not None
        assert json.loads(decoded) == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("CHATGPT.JSON", True),
        ("Export.JSONL", True),
        ("data.jsonl.txt", True),
        ("conversation.ndjson", True),
        ("notes.txt", False),
    ],
)
def test_has_supported_extension_contract(filename: str, expected: bool) -> None:
    assert _has_supported_extension(Path(filename)) is expected


def test_record_cursor_failure_updates_state_exactly() -> None:
    cursor_state: CursorStatePayload = {"failed_files": [], "failed_count": 0}

    _record_cursor_failure(cursor_state, "/tmp/data.json", "broken")

    assert cursor_state == {
        "failed_files": [{"path": "/tmp/data.json", "error": "broken"}],
        "failed_count": 1,
    }


def test_record_cursor_failure_is_noop_without_cursor_state() -> None:
    _record_cursor_failure(None, "/tmp/data.json", "broken")


def test_initialize_cursor_state_tracks_latest_path_and_mtime(tmp_path: Path) -> None:
    older = tmp_path / "older.json"
    newer = tmp_path / "newer.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    cursor_state: CursorStatePayload = _empty_cursor_state()
    _initialize_cursor_state(cursor_state, [older, newer])

    assert cursor_state["file_count"] == 2
    assert cursor_state["failed_files"] == []
    assert cursor_state["failed_count"] == 0
    assert cursor_state["latest_path"] == str(newer)
    assert cursor_state["latest_mtime"] == newer.stat().st_mtime


def test_select_paths_for_processing_skips_known_mtimes_only_when_mtime_enabled(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")

    first_mtime = _get_file_mtime(first)
    second_mtime = _get_file_mtime(second)
    assert first_mtime is not None
    assert second_mtime is not None

    selected, skipped = _select_paths_for_processing(
        [first, second],
        include_file_mtime=True,
        known_mtimes={str(first): first_mtime},
    )
    assert skipped == 1
    assert selected == [(second, second_mtime)]

    selected_without_mtime, skipped_without_mtime = _select_paths_for_processing(
        [first, second],
        include_file_mtime=False,
        known_mtimes={str(first): first_mtime},
    )
    assert skipped_without_mtime == 0
    assert selected_without_mtime == [(first, None), (second, None)]


def test_log_source_iteration_summary_emits_only_relevant_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    infos: list[str] = []
    warnings: list[str] = []

    monkeypatch.setattr(
        decoders_module.logger,
        "info",
        lambda message, *args: infos.append(message % args if args else message),
    )
    monkeypatch.setattr(
        decoders_module.logger,
        "warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )

    _log_source_iteration_summary(
        source_name="inbox",
        total_paths=5,
        skipped_mtime=2,
        failed_count=1,
        failure_kind="read",
    )

    assert infos == ["Skipped 2 of 5 files from source 'inbox' (unchanged mtime)"]
    assert warnings == ["Skipped 1 of 5 files from source 'inbox' due to read errors. Run with --verbose for details."]

    _log_source_iteration_summary(
        source_name="inbox",
        total_paths=5,
        skipped_mtime=0,
        failed_count=0,
        failure_kind="read",
    )

    assert infos == ["Skipped 2 of 5 files from source 'inbox' (unchanged mtime)"]
    assert warnings == ["Skipped 1 of 5 files from source 'inbox' due to read errors. Run with --verbose for details."]


def test_find_sessions_index_and_enrichment_contract(tmp_path: Path) -> None:
    """Claude session-index lookup and enrichment must stay source-local and deterministic."""
    session_dir = tmp_path / "claude-ai"
    session_dir.mkdir()
    session_file = session_dir / "session.jsonl"
    session_file.write_text("{}\n", encoding="utf-8")
    index_file = session_dir / "sessions-index.json"
    index_file.write_text('{"entries":[]}', encoding="utf-8")

    assert find_sessions_index(session_file) == index_file

    conversation = _parsed_conversation(
        provider_name=Provider.CLAUDE_CODE,
        provider_conversation_id="session-1",
        title="session-1",
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        messages=[_parsed_message("m1", role="user", text="hello")],
        provider_meta={"raw": True},
    )
    entry = SessionIndexEntry(
        session_id="session-1",
        full_path=str(session_file),
        first_prompt="Summarize this repo",
        summary="Investigate parser contracts",
        message_count=12,
        created="2025-01-02T00:00:00Z",
        modified="2025-01-03T00:00:00Z",
        git_branch="main",
        project_path="/tmp/project",
        is_sidechain=True,
    )

    enriched = enrich_conversation_from_index(conversation, entry)

    assert enriched.title == "Investigate parser contracts"
    assert enriched.created_at == "2025-01-02T00:00:00Z"
    assert enriched.updated_at == "2025-01-03T00:00:00Z"
    assert enriched.provider_meta == {
        "raw": True,
        "gitBranch": "main",
        "projectPath": "/tmp/project",
        "isSidechain": True,
        "summary": "Investigate parser contracts",
        "firstPrompt": "Summarize this repo",
        "title_source": "session-index:summary",
    }


def test_parse_sessions_index_contract(tmp_path: Path) -> None:
    """Claude session index parsing keeps only valid entries and preserves fields."""
    index_path = tmp_path / "sessions-index.json"
    index_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "sessionId": "session-1",
                        "fullPath": "/tmp/session-1.jsonl",
                        "firstPrompt": "hello",
                        "summary": "summary",
                        "messageCount": 7,
                        "created": "2025-01-01T00:00:00Z",
                        "modified": "2025-01-01T00:01:00Z",
                        "gitBranch": "main",
                        "projectPath": "/tmp/project",
                        "isSidechain": True,
                        "fileMtime": 123,
                    },
                    {"summary": "missing id should be ignored"},
                ]
            }
        ),
        encoding="utf-8",
    )

    entries = parse_sessions_index(index_path)
    assert set(entries) == {"session-1"}
    assert entries["session-1"] == SessionIndexEntry(
        session_id="session-1",
        full_path="/tmp/session-1.jsonl",
        first_prompt="hello",
        summary="summary",
        message_count=7,
        created="2025-01-01T00:00:00Z",
        modified="2025-01-01T00:01:00Z",
        git_branch="main",
        project_path="/tmp/project",
        is_sidechain=True,
        file_mtime=123,
    )


def test_iter_source_conversations_skips_agent_meta_sidecars(tmp_path: Path) -> None:
    source_dir = tmp_path / "claude-ai"
    source_dir.mkdir()
    (source_dir / "agent-a123.meta.json").write_text('{"agentType":"general-purpose"}', encoding="utf-8")

    conversations = list(iter_source_conversations(Source(name="claude-code", path=source_dir)))
    raw_items = list(iter_source_raw_data(Source(name="claude-code", path=source_dir)))

    assert conversations == []
    assert len(raw_items) == 1
    assert raw_items[0].source_path.endswith("agent-a123.meta.json")


def test_drive_cache_file_path_sanitizes_and_normalizes_suffix_contract(tmp_path: Path) -> None:
    """Drive cache naming must sanitize names and append a supported JSON suffix."""
    sanitized = drive_cache_file_path(tmp_path, "../Prompt Export")

    assert sanitized.parent == tmp_path
    assert sanitized.suffix == ".json"
    assert ".." not in sanitized.name
    assert "Prompt" in sanitized.stem
    assert drive_cache_file_path(tmp_path, "session.jsonl") == tmp_path / "session.jsonl"
    assert drive_cache_file_path(tmp_path, "trace.ndjson") == tmp_path / "trace.ndjson"


def _parse_context(
    provider_hint: str | Provider,
    *,
    should_group: bool,
    source_path: str,
    fallback_id: str,
    capture_raw: bool = True,
    sidecar_data: dict[str, object] | None = None,
) -> _ParseContext:
    return _ParseContext(
        provider_hint=Provider.from_string(provider_hint),
        should_group=should_group,
        source_path_str=source_path,
        fallback_id=fallback_id,
        file_mtime="2026-03-11T00:00:00+00:00",
        capture_raw=capture_raw,
        sidecar_data=sidecar_data or {},
    )


@pytest.mark.parametrize(
    (
        "label",
        "ctx",
        "filename",
        "raw",
        "pre_read_bytes",
        "expected_ids",
        "expected_provider_hint",
        "expected_provider_name",
        "expected_indexes",
        "expected_message_count",
    ),
    [
        (
            "individual-raw-capture",
            _parse_context(Provider.DRIVE, should_group=False, source_path="/tmp/export.json", fallback_id="export"),
            "export.json",
            json.dumps(
                [
                    {"id": "conv-1", "messages": [{"id": "m1", "role": "user", "text": "first"}]},
                    {"id": "conv-2", "messages": [{"id": "m2", "role": "assistant", "text": "second"}]},
                ]
            ).encode("utf-8"),
            None,
            ["conv-1", "conv-2"],
            Provider.DRIVE,
            Provider.DRIVE,
            [0, 1],
            None,
        ),
        (
            "individual-detected-provider-hint",
            _parse_context(Provider.DRIVE, should_group=False, source_path="/tmp/export.json", fallback_id="export"),
            "export.json",
            json.dumps(
                [
                    {
                        "mapping": {
                            "node-1": {
                                "message": {
                                    "author": {"role": "user"},
                                    "content": {"content_type": "text", "parts": ["hello"]},
                                }
                            }
                        }
                    }
                ]
            ).encode("utf-8"),
            None,
            None,
            Provider.CHATGPT,
            Provider.CHATGPT,
            [0],
            None,
        ),
        (
            "grouped-jsonl",
            _parse_context(Provider.CODEX, should_group=True, source_path="/tmp/session.jsonl", fallback_id="session"),
            "session.jsonl",
            (
                b'{"role":"user","content":[{"type":"input_text","text":"hello"}]}\n'
                b'{"role":"assistant","content":[{"type":"output_text","text":"hi"}]}\n'
            ),
            None,
            None,
            Provider.CODEX,
            Provider.CODEX,
            [None],
            2,
        ),
        (
            "grouped-json-whole-file",
            _parse_context(Provider.CODEX, should_group=True, source_path="/tmp/session.json", fallback_id="session"),
            "session.json",
            json.dumps(
                [
                    {"type": "session_meta", "payload": {"id": "session-1", "timestamp": "2025-01-01T00:00:00Z"}},
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "id": "msg-1",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hello"}],
                        },
                    },
                ]
            ).encode("utf-8"),
            "use-raw",
            None,
            Provider.CODEX,
            Provider.CODEX,
            [None],
            None,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_conversation_emitter_contract_matrix(
    label: str,
    ctx: _ParseContext,
    filename: str,
    raw: bytes,
    pre_read_bytes: str | None,
    expected_ids: list[str] | None,
    expected_provider_hint: str,
    expected_provider_name: str,
    expected_indexes: list[int | None],
    expected_message_count: int | None,
) -> None:
    emitted = list(
        _ConversationEmitter(ctx).emit(
            BytesIO(raw),
            filename,
            pre_read_bytes=raw if pre_read_bytes is not None else None,
        )
    )

    assert emitted
    assert [raw_data.source_index for raw_data, _ in emitted if raw_data is not None] == expected_indexes
    assert all(raw_data is not None for raw_data, _ in emitted)
    assert all(raw_data.provider_hint == expected_provider_hint for raw_data, _ in emitted if raw_data is not None)
    assert all(conversation.provider_name == expected_provider_name for _, conversation in emitted)
    if expected_ids is not None:
        assert [conversation.provider_conversation_id for _, conversation in emitted] == expected_ids
    if expected_message_count is not None:
        assert len(emitted[0][1].messages) == expected_message_count
    if label.startswith("grouped"):
        assert emitted[0][0] is not None and emitted[0][0].raw_bytes == raw


def test_conversation_emitter_resolves_schema_for_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Conversation emitter should pass SchemaResolution into parse_payload."""
    ctx = _parse_context(
        Provider.CHATGPT.value,
        should_group=False,
        source_path="/tmp/session.jsonl",
        fallback_id="session",
    )

    fake_registry = MagicMock()
    fake_registry.resolve_payload.return_value = object()
    fake_parse = MagicMock()
    fake_message = _parsed_message("m1", role="user", text="hello")
    fake_conversation = _parsed_conversation(
        provider_name=Provider.CLAUDE_CODE,
        provider_conversation_id="session",
        title="session",
        created_at=None,
        updated_at=None,
        messages=[fake_message],
    )
    fake_parse.return_value = [fake_conversation]

    def fake_parse_payload(
        provider: object,
        payload: object,
        fallback_id: str,
        _depth: int = 0,
        *,
        schema_resolution: object | None = None,
    ) -> list[ParsedConversation]:
        fake_parse(provider=provider, payload=payload, fallback_id=fallback_id, schema_resolution=schema_resolution)
        return [fake_conversation]

    monkeypatch.setattr("polylogue.sources.emitter._schema_registry_factory", lambda: fake_registry)
    monkeypatch.setattr("polylogue.sources.emitter.parse_payload", fake_parse_payload)

    raw = (
        b'{"mapping":{"r1":{"message":{"author":{"role":"user"},"content":{"content_type":"text","parts":["first"]}}}}}\n'
        b'{"mapping":{"r1":{"message":{"author":{"role":"assistant"},"content":{"content_type":"text","parts":["second"]}}}}}\n'
    )

    emitted = list(_ConversationEmitter(ctx).emit(BytesIO(raw), "session.jsonl"))

    assert emitted
    assert fake_parse.call_count == 2
    assert fake_registry.resolve_payload.call_count == 2
    resolved_arg = fake_parse.call_args_list[0].kwargs["schema_resolution"]
    assert resolved_arg is fake_registry.resolve_payload.return_value


def test_conversation_emitter_reuses_jsonl_sniff_payloads_for_grouped_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ParseContext(
        provider_hint=Provider.UNKNOWN,
        should_group=False,
        source_path_str="/tmp/session.jsonl",
        fallback_id="session",
        file_mtime="2026-03-11T00:00:00+00:00",
        capture_raw=True,
        sidecar_data={},
    )
    raw = (
        b'{"role":"user","content":[{"type":"input_text","text":"hello"}]}\n'
        b'{"role":"assistant","content":[{"type":"output_text","text":"hi"}]}\n'
    )
    parse_calls = 0
    original_iter_json_stream = _iter_json_stream

    def tracking_iter_json_stream(
        handle: BinaryIO | IO[bytes],
        path_name: str,
        unpack_lists: bool = True,
    ) -> Iterable[object]:
        nonlocal parse_calls
        parse_calls += 1
        yield from original_iter_json_stream(handle, path_name, unpack_lists=unpack_lists)

    monkeypatch.setattr("polylogue.sources.emitter._iter_json_stream", tracking_iter_json_stream)

    emitted = list(_ConversationEmitter(ctx).emit(BytesIO(raw), "session.jsonl"))

    assert emitted
    assert parse_calls == 1
    assert emitted[0][0] is not None and emitted[0][0].raw_bytes == raw
    assert emitted[0][1].provider_name == Provider.CODEX
    assert len(emitted[0][1].messages) == 2


def test_conversation_emitter_reuses_jsonl_sniff_payloads_for_individual_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoWholeReadBytesIO(BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if size == -1:
                raise AssertionError("unexpected whole-file read")
            return super().read(size)

    ctx = _ParseContext(
        provider_hint=Provider.UNKNOWN,
        should_group=False,
        source_path_str="/tmp/session.jsonl",
        fallback_id="session",
        file_mtime="2026-03-11T00:00:00+00:00",
        capture_raw=True,
        sidecar_data={},
    )
    raw = (
        b'{"mapping":{"r1":{"message":{"author":{"role":"user"},"content":{"content_type":"text","parts":["first"]}}}}}\n'
        b'{"mapping":{"r1":{"message":{"author":{"role":"assistant"},"content":{"content_type":"text","parts":["second"]}}}}}\n'
    )
    parse_calls = 0
    original_iter_json_stream = _iter_json_stream

    def tracking_iter_json_stream(
        handle: BinaryIO | IO[bytes],
        path_name: str,
        unpack_lists: bool = True,
    ) -> Iterable[object]:
        nonlocal parse_calls
        parse_calls += 1
        yield from original_iter_json_stream(handle, path_name, unpack_lists=unpack_lists)

    monkeypatch.setattr("polylogue.sources.emitter._iter_json_stream", tracking_iter_json_stream)

    emitted = list(_ConversationEmitter(ctx).emit(NoWholeReadBytesIO(raw), "session.jsonl"))

    assert emitted
    assert parse_calls == 1
    assert [raw_data.source_index for raw_data, _ in emitted if raw_data is not None] == [0, 1]
    assert all(raw_data is not None for raw_data, _ in emitted)
    assert all(conversation.provider_name == Provider.CHATGPT for _, conversation in emitted)


def test_conversation_emitter_detects_individual_jsonl_provider_from_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoWholeReadBytesIO(BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if size == -1:
                raise AssertionError("unexpected whole-file read")
            return super().read(size)

    ctx = _ParseContext(
        provider_hint=Provider.UNKNOWN,
        should_group=False,
        source_path_str="/tmp/session.jsonl",
        fallback_id="session",
        file_mtime="2026-03-11T00:00:00+00:00",
        capture_raw=True,
        sidecar_data={},
    )
    raw = (
        b'{"mapping":{"r1":{"message":{"author":{"role":"user"},"content":{"content_type":"text","parts":["first"]}}}}}\n'
        b'{"mapping":{"r1":{"message":{"author":{"role":"assistant"},"content":{"content_type":"text","parts":["second"]}}}}}\n'
    )
    original_detect_provider = dispatch_module.detect_provider

    def tracking_detect_provider(payload: object, path: object | None = None) -> Provider | None:
        if isinstance(payload, list):
            raise AssertionError("individual JSONL sniff should not require whole-list provider detection")
        return original_detect_provider(payload, path)

    monkeypatch.setattr("polylogue.sources.emitter.detect_provider", tracking_detect_provider)

    emitted = list(_ConversationEmitter(ctx).emit(NoWholeReadBytesIO(raw), "session.jsonl"))

    assert emitted
    assert [raw_data.source_index for raw_data, _ in emitted if raw_data is not None] == [0, 1]
    assert all(raw_data is not None for raw_data, _ in emitted)
    assert all(conversation.provider_name == Provider.CHATGPT for _, conversation in emitted)


def test_conversation_emitter_only_enriches_matching_claude_code_sessions_contract() -> None:
    entry = SessionIndexEntry(
        session_id="session-1",
        full_path="/tmp/session.jsonl",
        first_prompt="Summarize this repo",
        summary="Indexed summary",
        message_count=12,
        created="2025-01-02T00:00:00Z",
        modified="2025-01-03T00:00:00Z",
        git_branch="main",
        project_path="/tmp/project",
        is_sidechain=False,
    )
    ctx = _ParseContext(
        provider_hint=Provider.CLAUDE_CODE,
        should_group=True,
        source_path_str="/tmp/session.jsonl",
        fallback_id="session",
        file_mtime="2026-03-11T00:00:00+00:00",
        capture_raw=False,
        sidecar_data={"session_index": {"session-1": entry}},
    )
    emitter = _ConversationEmitter(ctx)
    matching = _parsed_conversation(
        provider_name=Provider.CLAUDE_CODE,
        provider_conversation_id="session-1",
        title="session-1",
        created_at=None,
        updated_at=None,
        messages=[_parsed_message("m1", role="user", text="hello")],
    )
    other = _parsed_conversation(
        provider_name=Provider.CHATGPT,
        provider_conversation_id="session-2",
        title="untouched",
        created_at=None,
        updated_at=None,
        messages=[_parsed_message("m2", role="user", text="hello")],
    )

    enriched = emitter._maybe_enrich(matching)
    untouched = emitter._maybe_enrich(other, Provider.CHATGPT)

    assert enriched.title == "Indexed summary"
    assert untouched.title == "untouched"


def _zip_entry(name: str, *, size: int = 100, compressed: int = 50) -> zipfile.ZipInfo:
    entry = zipfile.ZipInfo(name)
    entry.file_size = size
    entry.compress_size = compressed
    return entry


@pytest.mark.parametrize(
    ("source_name", "entries", "expected_kept", "expected_failed_count", "expected_failed_fragment"),
    [
        (
            "claude-ai",
            [_zip_entry("nested/conversations.json"), _zip_entry("nested/other.json")],
            ["nested/conversations.json", "nested/other.json"],
            0,
            None,
        ),
        (
            "chatgpt",
            [
                _zip_entry("nested/conversations.json"),
                _zip_entry("nested/other.json"),
                _zip_entry("nested/conversations.jsonl", size=2_000_000, compressed=1),
                _zip_entry("nested/readme.txt"),
            ],
            ["nested/conversations.json", "nested/other.json"],
            1,
            "nested/conversations.jsonl",
        ),
        (
            "chatgpt",
            [
                _zip_entry("nested/conversations.json"),
                _zip_entry("nested/conversations.jsonl"),
                _zip_entry("nested/conversations.ndjson"),
                _zip_entry("nested/conversations.jsonl.txt"),
            ],
            [
                "nested/conversations.json",
                "nested/conversations.jsonl",
                "nested/conversations.ndjson",
                "nested/conversations.jsonl.txt",
            ],
            0,
            None,
        ),
        (
            "chatgpt",
            [zipfile.ZipInfo("nested/"), _zip_entry("nested/readme.txt")],
            [],
            0,
            None,
        ),
        (
            "chatgpt",
            [_zip_entry("nested/huge.json", size=11 * 1024 * 1024 * 1024, compressed=1024)],
            [],
            1,
            "huge.json",
        ),
    ],
    ids=[
        "claude-bundle-json-kept",
        "suspicious-entry-rejected",
        "supported-json-extensions-kept",
        "directories-and-unsupported-skipped",
        "oversized-entry-rejected",
    ],
)
def test_zip_entry_validator_policy_contract(
    source_name: str,
    entries: list[zipfile.ZipInfo],
    expected_kept: list[str],
    expected_failed_count: int,
    expected_failed_fragment: str | None,
) -> None:
    cursor_state: CursorStatePayload = {"failed_files": [], "failed_count": 0}
    validator = _ZipEntryValidator(
        source_name,
        cursor_state=cursor_state,
        zip_path=Path("/tmp/archive.zip"),
    )

    kept = list(validator.filter_entries(entries))

    assert [entry.filename for entry in kept] == expected_kept
    assert cursor_state["failed_count"] == expected_failed_count
    if expected_failed_fragment is None:
        assert cursor_state["failed_files"] == []
    else:
        assert expected_failed_fragment in str(_failed_files(cursor_state)[0]["path"])


def test_zip_entry_provider_hint_keeps_fallback_contract() -> None:
    assert _zip_entry_provider_hint("nested/chatgpt-export.json", Provider.CLAUDE_AI) == Provider.CLAUDE_AI
    assert _zip_entry_provider_hint("nested/gemini/session.json", Provider.CHATGPT) == Provider.CHATGPT
    assert _zip_entry_provider_hint("nested/session.jsonl", Provider.CLAUDE_CODE) == Provider.CLAUDE_CODE


class _StubDriveRawClient:
    def __init__(
        self, files: list[DriveFile], *, raw_bytes: dict[str, bytes], failures: dict[str, Exception] | None = None
    ) -> None:
        self.files = files
        self.raw_bytes = raw_bytes
        self.failures = failures or {}

    def resolve_folder_id(self, folder_ref: str) -> str:
        return f"folder:{folder_ref}"

    def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]:
        del folder_id
        yield from self.files

    def download_bytes(self, file_id: str) -> bytes:
        if file_id in self.failures:
            raise self.failures[file_id]
        return self.raw_bytes[file_id]

    def download_json_payload(self, file_id: str, *, name: str) -> object:
        del name
        return json.loads(self.download_bytes(file_id))

    def download_to_path(self, file_id: str, dest: Path) -> DriveFile:
        if file_id in self.failures:
            raise self.failures[file_id]
        dest.write_bytes(self.raw_bytes[file_id])
        return next(file for file in self.files if file.file_id == file_id)


def test_iter_drive_raw_data_contract() -> None:
    source = Source(name="gemini", folder="Google AI Studio", path=Path("/tmp/drive-cache"))
    files = [
        DriveFile("chatgpt-1", "chatgpt-export.json", "application/json", "2025-01-01T00:00:00Z", 12),
        DriveFile("gemini-1", "gemini-prompt.json", "application/json", "2025-01-01T00:05:00Z", 8),
    ]
    client = _StubDriveRawClient(
        files,
        raw_bytes={"chatgpt-1": b'{"id":"chatgpt-1"}', "gemini-1": b'{"role":"model"}'},
    )
    cursor_state: CursorStatePayload = _empty_cursor_state()

    items = list(iter_drive_raw_data(source=source, client=client, cursor_state=cursor_state))

    assert [item.source_path for item in items] == [
        "/tmp/drive-cache/chatgpt-export.json",
        "/tmp/drive-cache/gemini-prompt.json",
    ]
    assert [item.provider_hint for item in items] == [Provider.GEMINI, Provider.GEMINI]
    assert [item.file_mtime for item in items] == ["2025-01-01T00:00:00Z", "2025-01-01T00:05:00Z"]
    assert cursor_state["file_count"] == 2
    assert cursor_state["latest_file_id"] == "gemini-1"
    assert cursor_state["latest_file_name"] == "gemini-prompt.json"


def test_iter_drive_raw_data_skips_known_mtimes_and_tracks_failures() -> None:
    source = Source(name="gemini", folder="Google AI Studio", path=Path("/tmp/drive-cache"))
    files = [
        DriveFile("old", "cached.json", "application/json", "2025-01-01T00:00:00Z", 12),
        DriveFile("bad", "broken.json", "application/json", "2025-01-01T00:01:00Z", 12),
        DriveFile("new", "new.json", "application/json", "2025-01-01T00:02:00Z", 12),
    ]
    client = _StubDriveRawClient(
        files,
        raw_bytes={"old": b"{}", "new": b'{"id":"new"}'},
        failures={"bad": RuntimeError("download failed")},
    )
    cursor_state: CursorStatePayload = _empty_cursor_state()

    items = list(
        iter_drive_raw_data(
            source=source,
            client=client,
            cursor_state=cursor_state,
            known_mtimes={"/tmp/drive-cache/cached.json": "2025-01-01T00:00:00Z"},
        )
    )

    assert [item.source_path for item in items] == ["/tmp/drive-cache/new.json"]
    assert cursor_state["file_count"] == 3
    assert cursor_state["error_count"] == 1
    assert cursor_state["latest_error_file"] == "broken.json"
    assert "download failed" in str(cursor_state["latest_error"])


def test_download_drive_files_contract() -> None:
    folder_id = "folder-law"
    files = [
        DriveFile("one", "session", "application/json", None, None),
        DriveFile("two", "bad.jsonl", "application/json", None, None),
    ]
    client = _StubDriveRawClient(
        files,
        raw_bytes={"one": b'{"id":"ok"}'},
        failures={"two": OSError("boom")},
    )

    with tempfile.TemporaryDirectory() as tmp:
        result = download_drive_files(client, folder_id, Path(tmp))

        assert result.total_files == 2
        assert len(result.downloaded_files) == 1
        assert result.downloaded_files[0].name == "session.json"
        assert result.downloaded_files[0].read_bytes() == b'{"id":"ok"}'
        assert result.failed_files == [{"file_id": "two", "name": "bad.jsonl", "error": "boom"}]


def test_iter_source_raw_data_reads_plain_and_zip_sources_contract(tmp_path: Path) -> None:
    """Raw source iteration keeps whole-file capture for plain files and grouped ZIP entries."""
    plain_path = tmp_path / "chatgpt-export.json"
    plain_path.write_text('{"mapping": {}, "id": "chatgpt-1"}', encoding="utf-8")

    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(
            "nested/session.jsonl",
            b'{"type":"user","message":{"role":"user","content":[{"type":"text","text":"hello"}]}}\n',
        )

    plain_items = list(iter_source_raw_data(Source(name="chatgpt", path=plain_path)))
    zip_items = list(iter_source_raw_data(Source(name="claude-code", path=archive_path)))

    assert len(plain_items) == 1
    assert plain_items[0].source_path == str(plain_path)
    assert plain_items[0].provider_hint == Provider.CHATGPT
    assert plain_items[0].blob_hash is not None
    assert plain_items[0].blob_size == plain_path.stat().st_size
    assert plain_items[0].file_mtime is not None

    assert len(zip_items) == 1
    assert zip_items[0].source_path == f"{archive_path}:nested/session.jsonl"
    assert zip_items[0].provider_hint == Provider.CLAUDE_CODE
    assert zip_items[0].blob_hash is not None
    assert zip_items[0].file_mtime is not None


def test_iter_source_raw_data_streams_grouped_zip_entries_into_blob_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.blob_store import BlobStore, get_blob_store

    entry_bytes = b'{"type":"user","message":{"role":"user","content":[{"type":"text","text":"hello"}]}}\n'
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("nested/session.jsonl", entry_bytes)

    def _fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Grouped ZIP acquisition should stream into the blob store")

    monkeypatch.setattr(BlobStore, "write_from_bytes", _fail)

    items = list(iter_source_raw_data(Source(name="claude-code", path=archive_path)))

    assert len(items) == 1
    assert items[0].blob_hash is not None
    assert get_blob_store().read_all(items[0].blob_hash) == entry_bytes


@pytest.mark.parametrize(
    ("entry_name", "payload_bytes", "expected_provider", "id_field", "expected_ids"),
    [
        (
            "conversations.json",
            json.dumps(
                [
                    {"id": "chatgpt-1", "mapping": {}},
                    {"id": "chatgpt-2", "mapping": {}},
                ]
            ).encode("utf-8"),
            Provider.CHATGPT,
            "id",
            ["chatgpt-1", "chatgpt-2"],
        ),
        (
            "claude-conversations.json",
            json.dumps(
                [
                    {"uuid": "claude-1", "name": "one", "chat_messages": []},
                    {"uuid": "claude-2", "name": "two", "chat_messages": []},
                ]
            ).encode("utf-8"),
            Provider.CLAUDE_AI,
            "uuid",
            ["claude-1", "claude-2"],
        ),
    ],
)
def test_iter_source_raw_data_splits_multi_conversation_zip_entries_for_non_grouped_providers(
    tmp_path: Path,
    entry_name: str,
    payload_bytes: bytes,
    expected_provider: Provider,
    id_field: str,
    expected_ids: list[str],
) -> None:
    from polylogue.storage.blob_store import get_blob_store

    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(f"nested/{entry_name}", payload_bytes)

    items = list(iter_source_raw_data(Source(name="inbox", path=archive_path)))

    expected_path = f"{archive_path}:nested/{entry_name}"
    assert [item.source_path for item in items] == [expected_path, expected_path]
    assert [item.source_index for item in items] == [0, 1]
    assert [item.provider_hint for item in items] == [expected_provider, expected_provider]
    assert all(item.blob_hash is not None for item in items)
    assert all(item.raw_bytes == b"" for item in items)
    assert [
        json.loads(get_blob_store().read_all(item.blob_hash))[id_field] for item in items if item.blob_hash is not None
    ] == expected_ids


def test_iter_source_raw_data_avoids_whole_blob_provider_detection_for_zip_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(
            "nested/conversations.json",
            json.dumps(
                [
                    {"id": "chatgpt-1", "mapping": {}},
                    {"id": "chatgpt-2", "mapping": {}},
                ]
            ).encode("utf-8"),
        )

    def _fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("ZIP acquisition should not use whole-blob provider detection")

    monkeypatch.setattr(
        "polylogue.sources.source_acquisition._detect_provider_from_raw_bytes",
        _fail,
    )

    items = list(iter_source_raw_data(Source(name="inbox", path=archive_path)))

    assert len(items) == 2


def test_iter_source_raw_data_reports_split_payload_observations(tmp_path: Path) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(
            "nested/conversations.json",
            json.dumps(
                [
                    {"id": "chatgpt-1", "mapping": {}},
                    {"id": "chatgpt-2", "mapping": {}},
                ]
            ).encode("utf-8"),
        )

    observations: list[dict[str, object]] = []

    items = list(
        iter_source_raw_data(
            Source(name="inbox", path=archive_path),
            observation_callback=observations.append,
        )
    )

    assert len(items) == 2
    assert observations
    peak = max(observations, key=lambda observation: _numeric_observation_value(observation, "peak_rss_self_mb"))
    assert peak["phase"] == "zip-entry-split-payload-serialized"
    assert peak["source_path"] == f"{archive_path}:nested/conversations.json"
    assert peak["provider_hint"] == Provider.CHATGPT.value
    assert peak["source_index"] in {0, 1}
    assert _numeric_observation_value(peak, "blob_size") > 0
    assert peak["artifact_kind"]
    assert _numeric_observation_value(peak, "detect_provider_ms") >= 0.0
    assert _numeric_observation_value(peak, "classify_ms") >= 0.0
    assert _numeric_observation_value(peak, "serialize_ms") >= 0.0
    assert _numeric_observation_value(peak, "peak_rss_self_mb") > 0.0


def test_iter_source_raw_data_streams_preserved_zip_entries_into_blob_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.blob_store import get_blob_store

    entry_bytes = json.dumps({"id": "chatgpt-1", "mapping": {}}).encode("utf-8")
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("nested/chatgpt-export.json", entry_bytes)

    def _fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Preserved ZIP entries should stream into the blob store")

    monkeypatch.setattr(BlobStore, "write_from_bytes", _fail)

    items = list(iter_source_raw_data(Source(name="chatgpt", path=archive_path)))

    assert len(items) == 1
    assert items[0].blob_hash is not None
    assert get_blob_store().read_all(items[0].blob_hash) == entry_bytes


def test_iter_entry_payloads_locks_provider_after_first_detected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        {"id": "chatgpt-1", "mapping": {}},
        {"id": "chatgpt-2", "mapping": {}},
        {"id": "chatgpt-3", "mapping": {}},
    ]
    detect_calls: list[JSONDocument] = []
    original_detect_provider = dispatch_module.detect_provider

    def tracking_detect_provider(payload: object, path: object | None = None) -> Provider | None:
        del path
        if isinstance(payload, dict):
            detect_calls.append(payload)
        return original_detect_provider(payload)

    monkeypatch.setattr("polylogue.sources.source_acquisition.detect_provider", tracking_detect_provider)

    items = list(
        _iter_entry_payloads(
            BytesIO(json.dumps(payloads).encode("utf-8")),
            stream_name="conversations.json",
            provider_hint=Provider.UNKNOWN,
        )
    )

    assert [provider for provider, _, _ in items] == [
        Provider.CHATGPT,
        Provider.CHATGPT,
        Provider.CHATGPT,
    ]
    assert items[0][2] >= 0.0
    assert items[1][2] >= 0.0
    assert items[2][2] == 0.0
    assert detect_calls == payloads[:2]


def test_iter_source_raw_data_keeps_source_family_hints_for_mixed_zip_sources(tmp_path: Path) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("nested/chatgpt-export.json", b'{"mapping": {}, "id": "chatgpt-1"}')
        zf.writestr("nested/gemini-export.json", b'{"chunkedPrompt": {"chunks": []}}')

    items = list(iter_source_raw_data(Source(name="chatgpt", path=archive_path)))

    assert [item.source_path for item in items] == [
        f"{archive_path}:nested/chatgpt-export.json",
        f"{archive_path}:nested/gemini-export.json",
    ]
    assert [item.provider_hint for item in items] == [Provider.CHATGPT, Provider.GEMINI]


def test_iter_source_raw_data_skips_known_mtimes_without_reading_file(tmp_path: Path) -> None:
    skipped = tmp_path / "cached.json"
    fresh = tmp_path / "fresh.json"
    skipped.write_text('{"id":"cached"}', encoding="utf-8")
    fresh.write_text('{"id":"fresh"}', encoding="utf-8")

    items = list(
        iter_source_raw_data(
            Source(name="chatgpt", path=tmp_path),
            known_mtimes={str(skipped): str(_get_file_mtime(skipped))},
        )
    )

    assert [item.source_path for item in items] == [str(fresh)]


def test_iter_source_raw_data_skips_zero_byte_plain_files_and_tracks_failure(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")

    cursor_state: CursorStatePayload = _empty_cursor_state()
    items = list(iter_source_raw_data(Source(name="codex", path=empty), cursor_state=cursor_state))

    assert items == []
    assert cursor_state["failed_count"] == 1
    assert cursor_state["failed_files"] == [{"path": str(empty), "error": "empty file"}]


def test_iter_source_raw_data_summarizes_zero_byte_plain_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_bytes(b"")
    second.write_bytes(b"")

    warnings: list[str] = []
    debugs: list[str] = []

    monkeypatch.setattr(
        source_acquisition.logger,
        "warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )
    monkeypatch.setattr(
        source_acquisition.logger,
        "debug",
        lambda message, *args: debugs.append(message % args if args else message),
    )

    items = list(iter_source_raw_data(Source(name="codex", path=tmp_path)))

    assert items == []
    assert warnings == ["Skipped 2 empty artifacts from source 'codex'. Run with --verbose for details."]
    assert debugs == [
        f"Skipping empty source file: {first}",
        f"Skipping empty source file: {second}",
    ]


def test_iter_source_raw_data_skips_zero_byte_zip_entries_and_tracks_failure(tmp_path: Path) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("nested/empty.jsonl", b"")

    cursor_state: CursorStatePayload = _empty_cursor_state()
    items = list(iter_source_raw_data(Source(name="codex", path=archive_path), cursor_state=cursor_state))

    assert items == []
    assert cursor_state["failed_count"] == 1
    assert cursor_state["failed_files"] == [{"path": f"{archive_path}:nested/empty.jsonl", "error": "empty file"}]


def test_iter_source_raw_data_summarizes_zero_byte_zip_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("nested/empty-a.jsonl", b"")
        zf.writestr("nested/empty-b.jsonl", b"")

    warnings: list[str] = []
    debugs: list[str] = []

    monkeypatch.setattr(
        source_acquisition.logger,
        "warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )
    monkeypatch.setattr(
        source_acquisition.logger,
        "debug",
        lambda message, *args: debugs.append(message % args if args else message),
    )

    items = list(iter_source_raw_data(Source(name="codex", path=archive_path)))

    assert items == []
    assert warnings == ["Skipped 2 empty artifacts from source 'codex'. Run with --verbose for details."]
    assert debugs == [
        f"Skipping empty source entry: {archive_path}:nested/empty-a.jsonl",
        f"Skipping empty source entry: {archive_path}:nested/empty-b.jsonl",
    ]


def test_iter_source_raw_data_tracks_read_failures_without_stopping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    good.write_text('{"mapping": {}, "id": "good"}', encoding="utf-8")
    bad.write_text('{"mapping": {}, "id": "bad"}', encoding="utf-8")

    # Patch blob_store.write_from_path to fail for the bad file
    original_write = BlobStore.write_from_path

    def flaky_write(
        self: BlobStore,
        source: Path,
        *,
        heartbeat: Heartbeat | None = None,
    ) -> tuple[str, int]:
        del heartbeat
        if source == bad:
            raise OSError("boom")
        return original_write(self, source)

    monkeypatch.setattr(BlobStore, "write_from_path", flaky_write)

    cursor_state: CursorStatePayload = _empty_cursor_state()
    items = list(iter_source_raw_data(Source(name="chatgpt", path=tmp_path), cursor_state=cursor_state))

    assert [item.source_path for item in items] == [str(good)]
    assert cursor_state["failed_count"] == 1
    assert any(entry["path"] == str(bad) and entry["error"] == "boom" for entry in _failed_files(cursor_state))


# =============================================================================
# MERGED FROM test_acquisition_fs.py (robustness for raw file acquisition)
# =============================================================================


def test_jsonl_crlf_line_separator() -> None:
    """CRLF between JSONL lines (\r\n) decodes without stripping content."""
    line1 = json.dumps({"idx": 1})
    line2 = json.dumps({"idx": 2})
    raw = (line1 + "\r\n" + line2).encode("utf-8")
    result = _decode_json_bytes(raw)
    assert result is not None
    # Both records should be parseable from the result
    lines = [ln for ln in result.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["idx"] == 1
    assert json.loads(lines[1])["idx"] == 2


def test_latin1_file_does_not_crash() -> None:
    """A latin-1 byte sequence falls through to utf-8/ignore and returns a string."""
    # 0xe9 = 'é' in latin-1, but invalid as a standalone byte in UTF-8
    raw = b'{"note": "caf\xe9"}'
    result = _decode_json_bytes(raw)
    # Must not crash — invalid bytes are silently dropped by the ignore path
    assert result is not None
    # The returned string must be non-empty
    assert len(result.strip()) > 0


def test_null_bytes_stripped() -> None:
    """Embedded NUL bytes (\\x00) are stripped by the cleaner step."""
    payload = '{"key": "value"}'
    raw_with_nulls = (payload[:5] + "\x00\x00" + payload[5:]).encode("utf-8")
    result = _decode_json_bytes(raw_with_nulls)
    assert result is not None
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_empty_bytes_returns_none() -> None:
    """Empty byte input returns None rather than crashing."""
    result = _decode_json_bytes(b"")
    assert result is None


# =============================================================================
# MERGED FROM test_parse_laws.py (property laws for parser roles)
# =============================================================================

# ---------------------------------------------------------------------------
# Law 1: normalize_role never raises for any non-empty string
# ---------------------------------------------------------------------------


@given(st.text(min_size=1))
def test_normalize_role_never_raises_for_nonempty(text: str) -> None:
    """normalize_role handles any non-empty string without raising."""
    # normalize_role raises only on empty/whitespace-only strings
    stripped = text.strip()
    if stripped:
        result = normalize_role(text)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Law 2: normalize_role always returns one of the canonical roles
# ---------------------------------------------------------------------------

CANONICAL_ROLES = frozenset({"user", "assistant", "system", "tool", "unknown"})


@given(st.text(min_size=1))
def test_normalize_role_result_is_canonical(text: str) -> None:
    """normalize_role always returns a canonical role string."""
    stripped = text.strip()
    if stripped:
        result = normalize_role(text)
        assert result in CANONICAL_ROLES


# ---------------------------------------------------------------------------
# Law 3: normalize_role is idempotent on its own output
# ---------------------------------------------------------------------------


@given(st.sampled_from(sorted(CANONICAL_ROLES - {"unknown"})))
def test_normalize_role_idempotent_on_canonical(role: str) -> None:
    """Applying normalize_role to a canonical role returns the same value."""
    result = normalize_role(role)
    assert result == role


# ---------------------------------------------------------------------------
# Law 4: normalize_role is case-insensitive
# ---------------------------------------------------------------------------


@given(st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
def test_normalize_role_case_insensitive(text: str) -> None:
    """normalize_role gives the same result for any case variant."""
    stripped = text.strip()
    if stripped:
        lower_result = normalize_role(stripped.lower())
        upper_result = normalize_role(stripped.upper())
        title_result = normalize_role(stripped.title())
        assert lower_result == upper_result == title_result


# ---------------------------------------------------------------------------
# Law 5: normalize_role strips whitespace before normalizing
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(["user", "assistant", "system", "tool"]),
    st.integers(min_value=0, max_value=5),
)
def test_normalize_role_strips_whitespace(role: str, padding: int) -> None:
    """normalize_role ignores leading/trailing whitespace."""
    padded = " " * padding + role + " " * padding
    assert normalize_role(padded) == role


# ---------------------------------------------------------------------------
# Law 6: normalize_role raises ValueError for empty/whitespace-only input
# ---------------------------------------------------------------------------


@given(st.from_regex(r"^[\s]*$", fullmatch=True))
def test_normalize_role_raises_on_empty(empty: str) -> None:
    """normalize_role raises ValueError for empty or whitespace-only input."""
    with pytest.raises((ValueError, TypeError, AttributeError)):
        normalize_role(empty)


@pytest.mark.parametrize(
    ("roles", "expected_pairs"),
    [
        (["user", "assistant"], [("m0", "m1")]),
        (["system", "user", "assistant", "assistant"], [("m1", "m2")]),
        (["user", "tool", "assistant", "user", "assistant"], [("m3", "m4")]),
        (["assistant", "user"], []),
    ],
)
def test_chatgpt_iter_user_assistant_pairs_contract(
    roles: list[str],
    expected_pairs: list[tuple[str, str]],
) -> None:
    mapping: dict[str, ChatGPTNode] = {}
    children = [f"node-{idx}" for idx in range(len(roles))]
    mapping["root"] = ChatGPTNode(id="root", parent=None, children=children[:1])
    for idx, role in enumerate(roles):
        node_id = f"node-{idx}"
        next_child = [f"node-{idx + 1}"] if idx + 1 < len(roles) else []
        mapping[node_id] = ChatGPTNode(
            id=node_id,
            parent="root" if idx == 0 else f"node-{idx - 1}",
            children=next_child,
            message=ChatGPTMessage(
                id=f"m{idx}",
                author=ChatGPTAuthor(role=role),
                content=ChatGPTContent(content_type="text", parts=[f"{role}-{idx}"]),
            ),
        )

    conversation = ChatGPTConversation(
        id="conv-pairs",
        conversation_id="conv-pairs",
        title="pairs",
        create_time=1700000000.0,
        update_time=1700000100.0,
        mapping=mapping,
        current_node=f"node-{len(roles) - 1}" if roles else "root",
    )

    assert [(user.id, assistant.id) for user, assistant in conversation.iter_user_assistant_pairs()] == expected_pairs


def test_claude_code_helper_conversion_contracts() -> None:
    tool = ClaudeCodeToolUse(id="tool-1", name="bash", input={"command": "git status"})
    trace = ClaudeCodeThinkingBlock(thinking="chain of thought")
    usage = ClaudeCodeUsage(
        input_tokens=12,
        output_tokens=34,
        cache_creation_input_tokens=5,
        cache_read_input_tokens=6,
    )

    tool_call = tool.to_tool_call()
    reasoning = trace.to_reasoning_trace()
    token_usage = usage.to_token_usage()

    assert tool_call.name == "bash"
    assert tool_call.id == "tool-1"
    assert tool_call.input == {"command": "git status"}
    assert tool_call.provider == "claude-code"
    assert tool_call.raw == tool.model_dump()

    assert reasoning.text == "chain of thought"
    assert reasoning.provider == "claude-code"
    assert reasoning.raw == trace.model_dump()

    assert token_usage.input_tokens == 12
    assert token_usage.output_tokens == 34
    assert token_usage.cache_write_tokens == 5
    assert token_usage.cache_read_tokens == 6
