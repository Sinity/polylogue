"""Law-based contracts for source detection, dispatch, and JSON iteration."""

from __future__ import annotations

import json
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.config import Source
from polylogue.sources.drive import (
    download_drive_files,
    drive_cache_file_path,
    iter_drive_raw_data,
)
from polylogue.sources.drive_client import DriveFile
from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage
from polylogue.sources.parsers.claude import (
    SessionIndexEntry,
    enrich_conversation_from_index,
    find_sessions_index,
    parse_sessions_index,
)
from polylogue.sources.source import (
    _ConversationEmitter,
    _decode_json_bytes,
    _iter_json_stream,
    _ParseContext,
    _zip_entry_provider_hint,
    _ZipEntryValidator,
    detect_provider,
    iter_source_conversations,
    iter_source_conversations_with_raw,
    iter_source_raw_data,
    parse_drive_payload,
    parse_payload,
)
from polylogue.types import Provider
from tests.infra.strategies import (
    conversations_wrapper_bytes_strategy,
    json_array_bytes_strategy,
    json_document_strategy,
    jsonl_bytes_strategy,
    provider_export_strategy,
    provider_hint_path_strategy,
    provider_payload_case_strategy,
    provider_payload_strategy,
    provider_source_case_strategy,
)

_CANONICAL_PROVIDERS = (
    Provider.CHATGPT.value,
    Provider.CLAUDE.value,
    Provider.CLAUDE_CODE.value,
    Provider.CODEX.value,
    Provider.GEMINI.value,
)


@given(
    provider_payload_case_strategy(
        (
            Provider.CHATGPT.value,
            Provider.CLAUDE.value,
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


@given(st.sampled_from(_CANONICAL_PROVIDERS).flatmap(
    lambda provider: provider_hint_path_strategy(provider).map(lambda path: (provider, path))
))
@settings(max_examples=30)
def test_detect_provider_uses_path_hints_for_unknown_payload(case: tuple[str, Path]) -> None:
    """Filename/path hints still classify unknown payload shapes."""
    provider, path = case
    assert detect_provider({"unrelated": True}, path) == provider


@given(provider_payload_case_strategy(_CANONICAL_PROVIDERS))
@settings(max_examples=35)
def test_parse_payload_generated_exports_produce_provider_named_conversations(case: tuple[str, object]) -> None:
    """Runtime dispatch parses generated provider payloads into provider-owned conversations."""
    provider, payload = case
    conversations = parse_payload(provider, payload, "fallback-id")
    assert conversations
    assert all(str(conversation.provider_name) == provider for conversation in conversations)
    assert all(conversation.provider_conversation_id for conversation in conversations)


@given(st.lists(provider_payload_strategy(Provider.CHATGPT.value), min_size=1, max_size=4))
@settings(max_examples=20)
def test_parse_payload_chatgpt_bundle_preserves_item_count(payloads: list[object]) -> None:
    """ChatGPT bundle lists parse one conversation per item."""
    conversations = parse_payload(Provider.CHATGPT.value, payloads, "bundle")
    assert len(conversations) == len(payloads)


@given(st.lists(provider_payload_strategy(Provider.CLAUDE.value), min_size=1, max_size=4))
@settings(max_examples=20)
def test_parse_payload_claude_bundle_preserves_item_count(payloads: list[object]) -> None:
    """Claude AI bundle lists parse one conversation per item."""
    conversations = parse_payload(Provider.CLAUDE.value, payloads, "bundle")
    assert len(conversations) == len(payloads)


@given(st.lists(provider_payload_strategy(Provider.GEMINI.value), min_size=1, max_size=4))
@settings(max_examples=20)
def test_parse_payload_gemini_chunked_lists_preserve_item_count(payloads: list[object]) -> None:
    """Chunked Gemini conversation lists parse one conversation per item."""
    conversations = parse_payload(Provider.GEMINI.value, payloads, "bundle")
    assert len(conversations) == len(payloads)


@given(st.lists(provider_payload_strategy(Provider.CHATGPT.value), min_size=1, max_size=4))
@settings(max_examples=20)
def test_parse_payload_conversations_wrapper_preserves_items(payloads: list[object]) -> None:
    """Generic `conversations` wrappers delegate to the wrapped payloads."""
    wrapper = {"conversations": payloads}
    conversations = parse_payload(Provider.CHATGPT.value, wrapper, "wrapped")
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
@settings(max_examples=30)
def test_iter_json_stream_root_list_round_trips_documents(case: tuple[list[dict[str, object]], bytes]) -> None:
    """Streaming a root JSON array yields the original item sequence."""
    documents, raw = case
    assert list(_iter_json_stream(BytesIO(raw), "test.json")) == documents


@given(conversations_wrapper_bytes_strategy())
@settings(max_examples=30)
def test_iter_json_stream_conversations_wrapper_round_trips_documents(case: tuple[list[dict[str, object]], bytes]) -> None:
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
def test_iter_json_stream_jsonl_preserves_valid_records_with_blank_lines(case: tuple[list[dict[str, object]], bytes]) -> None:
    """JSONL parsing ignores blank lines but preserves valid record order exactly."""
    documents, raw = case
    assert list(_iter_json_stream(BytesIO(raw), "test.jsonl")) == documents


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


@given(
    provider_source_case_strategy(
        providers=(
            Provider.CHATGPT.value,
            Provider.CLAUDE.value,
            Provider.CLAUDE_CODE.value,
            Provider.CODEX.value,
            Provider.GEMINI.value,
        )
    ),
    st.booleans(),
)
@settings(max_examples=30, deadline=None)
def test_iter_source_conversations_round_trips_generated_exports(case: dict[str, object], use_zip: bool) -> None:
    """Generated provider exports should stay discoverable through file and ZIP iteration."""
    with tempfile.TemporaryDirectory() as tmp:
        source = _materialize_generated_source(
            Path(tmp),
            hint_path=case["path"],
            raw=case["raw"],
            use_zip=use_zip,
        )
        conversations = list(iter_source_conversations(source))

    assert conversations
    assert all(str(conversation.provider_name) == case["provider"] for conversation in conversations)


@given(
    provider_source_case_strategy(
        providers=(
            Provider.CHATGPT.value,
            Provider.CLAUDE.value,
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
    case: dict[str, object],
    use_zip: bool,
    capture_raw: bool,
) -> None:
    """Raw iteration must preserve provider parsing while toggling raw payload capture cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        source = _materialize_generated_source(
            Path(tmp),
            hint_path=case["path"],
            raw=case["raw"],
            use_zip=use_zip,
        )
        items = list(iter_source_conversations_with_raw(source, capture_raw=capture_raw))

    assert items
    assert all(str(conversation.provider_name) == case["provider"] for _, conversation in items)
    if capture_raw:
        assert all(raw_data is not None for raw_data, _ in items)
        assert all(raw_data.raw_bytes for raw_data, _ in items if raw_data is not None)
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"\xef\xbb\xbf{\"id\":\"bom\"}", {"id": "bom"}),
        (b"{\x00\"id\":\"nulls\"}", {"id": "nulls"}),
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


def test_find_sessions_index_and_enrichment_contract(tmp_path: Path) -> None:
    """Claude session-index lookup and enrichment must stay source-local and deterministic."""
    session_dir = tmp_path / "claude"
    session_dir.mkdir()
    session_file = session_dir / "session.jsonl"
    session_file.write_text("{}\n", encoding="utf-8")
    index_file = session_dir / "sessions-index.json"
    index_file.write_text('{"entries":[]}', encoding="utf-8")

    assert find_sessions_index(session_file) == index_file

    conversation = ParsedConversation(
        provider_name="claude-code",
        provider_conversation_id="session-1",
        title="session-1",
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        messages=[ParsedMessage(provider_message_id="m1", role="user", text="hello")],
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


def test_drive_cache_file_path_sanitizes_and_normalizes_suffix_contract(tmp_path: Path) -> None:
    """Drive cache naming must sanitize names and append a supported JSON suffix."""
    sanitized = drive_cache_file_path(tmp_path, "../Prompt Export")

    assert sanitized.parent == tmp_path
    assert sanitized.suffix == ".json"
    assert ".." not in sanitized.name
    assert "Prompt" in sanitized.stem
    assert drive_cache_file_path(tmp_path, "session.jsonl") == tmp_path / "session.jsonl"
    assert drive_cache_file_path(tmp_path, "trace.ndjson") == tmp_path / "trace.ndjson"


def test_conversation_emitter_individual_raw_capture_contract() -> None:
    payloads = [
        {"id": "conv-1", "messages": [{"id": "m1", "role": "user", "text": "first"}]},
        {"id": "conv-2", "messages": [{"id": "m2", "role": "assistant", "text": "second"}]},
    ]
    raw = json.dumps(payloads).encode("utf-8")
    ctx = _ParseContext(
        provider_hint=Provider.DRIVE,
        should_group=False,
        source_path_str="/tmp/export.json",
        fallback_id="export",
        file_mtime="2026-03-11T00:00:00+00:00",
        capture_raw=True,
        session_index={},
        detect_path=Path("export.json"),
    )

    emitted = list(_ConversationEmitter(ctx).emit(BytesIO(raw), "export.json"))

    assert [conversation.provider_conversation_id for _, conversation in emitted] == ["conv-1", "conv-2"]
    assert [raw_data.source_index for raw_data, _ in emitted if raw_data is not None] == [0, 1]
    assert all(raw_data is not None for raw_data, _ in emitted)
    assert all(raw_data.provider_hint == Provider.DRIVE for raw_data, _ in emitted if raw_data is not None)


def test_conversation_emitter_grouped_jsonl_contract() -> None:
    raw = (
        b'{"role":"user","content":[{"type":"input_text","text":"hello"}]}\n'
        b'{"role":"assistant","content":[{"type":"output_text","text":"hi"}]}\n'
    )
    ctx = _ParseContext(
        provider_hint=Provider.CODEX,
        should_group=True,
        source_path_str="/tmp/session.jsonl",
        fallback_id="session",
        file_mtime="2026-03-11T00:00:00+00:00",
        capture_raw=True,
        session_index={},
        detect_path=Path("session.jsonl"),
    )

    emitted = list(_ConversationEmitter(ctx).emit(BytesIO(raw), "session.jsonl"))

    assert len(emitted) == 1
    raw_data, conversation = emitted[0]
    assert raw_data is not None
    assert raw_data.raw_bytes == raw
    assert conversation.provider_name == Provider.CODEX
    assert len(conversation.messages) == 2


def test_zip_entry_validator_filters_claude_bundle_entries_contract() -> None:
    validator = _ZipEntryValidator(
        "claude",
        cursor_state={"failed_files": [], "failed_count": 0},
        zip_path=Path("/tmp/archive.zip"),
    )

    keep = zipfile.ZipInfo("nested/conversations.json")
    keep.file_size = 100
    keep.compress_size = 50

    wrong_name = zipfile.ZipInfo("nested/other.json")
    wrong_name.file_size = 100
    wrong_name.compress_size = 50

    assert [entry.filename for entry in validator.filter_entries([keep, wrong_name])] == [
        "nested/conversations.json"
    ]


def test_zip_entry_validator_rejects_suspicious_entries_contract() -> None:
    cursor_state = {"failed_files": [], "failed_count": 0}
    validator = _ZipEntryValidator(
        "chatgpt",
        cursor_state=cursor_state,
        zip_path=Path("/tmp/archive.zip"),
    )

    keep = zipfile.ZipInfo("nested/conversations.json")
    keep.file_size = 100
    keep.compress_size = 50

    wrong_name = zipfile.ZipInfo("nested/other.json")
    wrong_name.file_size = 100
    wrong_name.compress_size = 50

    suspicious = zipfile.ZipInfo("nested/conversations.jsonl")
    suspicious.file_size = 2_000_000
    suspicious.compress_size = 1

    unsupported = zipfile.ZipInfo("nested/readme.txt")
    unsupported.file_size = 100
    unsupported.compress_size = 50

    kept = list(validator.filter_entries([keep, wrong_name, suspicious, unsupported]))

    assert [entry.filename for entry in kept] == ["nested/conversations.json", "nested/other.json"]
    assert cursor_state["failed_count"] == 1
    assert cursor_state["failed_files"][0]["path"] == "/tmp/archive.zip:nested/conversations.jsonl"


def test_zip_entry_provider_hint_prefers_entry_name_contract() -> None:
    assert _zip_entry_provider_hint("nested/chatgpt-export.json", Provider.CLAUDE) == Provider.CHATGPT
    assert _zip_entry_provider_hint("nested/gemini/session.json", Provider.CHATGPT) == Provider.GEMINI
    assert _zip_entry_provider_hint("nested/session.jsonl", Provider.CLAUDE_CODE) == Provider.CLAUDE_CODE


class _StubDriveRawClient:
    def __init__(self, files: list[DriveFile], *, raw_bytes: dict[str, bytes], failures: dict[str, Exception] | None = None) -> None:
        self.files = files
        self.raw_bytes = raw_bytes
        self.failures = failures or {}

    def resolve_folder_id(self, folder_ref: str) -> str:
        return f"folder:{folder_ref}"

    def iter_json_files(self, folder_id: str):
        yield from self.files

    def download_bytes(self, file_id: str) -> bytes:
        if file_id in self.failures:
            raise self.failures[file_id]
        return self.raw_bytes[file_id]

    def download_to_path(self, file_id: str, dest: Path) -> None:
        if file_id in self.failures:
            raise self.failures[file_id]
        dest.write_bytes(self.raw_bytes[file_id])


def test_iter_drive_raw_data_contract() -> None:
    source = Source(name="drive", folder="Google AI Studio", path=Path("/tmp/drive-cache"))
    files = [
        DriveFile("chatgpt-1", "chatgpt-export.json", "application/json", "2025-01-01T00:00:00Z", 12),
        DriveFile("gemini-1", "gemini-prompt.json", "application/json", "2025-01-01T00:05:00Z", 8),
    ]
    client = _StubDriveRawClient(
        files,
        raw_bytes={"chatgpt-1": b'{"id":"chatgpt-1"}', "gemini-1": b'{"role":"model"}'},
    )
    cursor_state: dict[str, object] = {}

    items = list(iter_drive_raw_data(source=source, client=client, cursor_state=cursor_state))

    assert [item.source_path for item in items] == [
        "/tmp/drive-cache/chatgpt-export.json",
        "/tmp/drive-cache/gemini-prompt.json",
    ]
    assert [item.provider_hint for item in items] == ["chatgpt", "gemini"]
    assert [item.file_mtime for item in items] == ["2025-01-01T00:00:00Z", "2025-01-01T00:05:00Z"]
    assert cursor_state["file_count"] == 2
    assert cursor_state["latest_file_id"] == "gemini-1"
    assert cursor_state["latest_file_name"] == "gemini-prompt.json"


def test_iter_drive_raw_data_skips_known_mtimes_and_tracks_failures() -> None:
    source = Source(name="drive", folder="Google AI Studio", path=Path("/tmp/drive-cache"))
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
    cursor_state: dict[str, object] = {}

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
    """Raw source iteration yields one blob per file or ZIP entry without parsing."""
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
    assert plain_items[0].raw_bytes == plain_path.read_bytes()
    assert plain_items[0].file_mtime is not None

    assert len(zip_items) == 1
    assert zip_items[0].source_path == f"{archive_path}:nested/session.jsonl"
    assert zip_items[0].provider_hint == Provider.CLAUDE_CODE
    assert b'"type":"user"' in zip_items[0].raw_bytes
    assert zip_items[0].file_mtime is not None


def test_iter_source_raw_data_uses_per_entry_provider_hints_for_mixed_zip_sources(tmp_path: Path) -> None:
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
