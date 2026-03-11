"""Law-based contracts for source detection, dispatch, and JSON iteration."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.sources.source import _decode_json_bytes, _iter_json_stream, detect_provider, parse_payload
from polylogue.types import Provider
from tests.infra.strategies import (
    conversations_wrapper_bytes_strategy,
    json_array_bytes_strategy,
    json_document_strategy,
    jsonl_bytes_strategy,
    provider_hint_path_strategy,
    provider_payload_case_strategy,
    provider_payload_strategy,
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
