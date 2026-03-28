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
from polylogue.sources.source import (
    _ConversationEmitter,
    _decode_json_bytes,
    _iter_json_stream,
    _ParseContext,
    _ZipEntryValidator,
    detect_provider,
    iter_source_conversations,
    iter_source_conversations_with_raw,
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
