"""Hypothesis strategies for provider-specific export formats.

Schema-driven strategies powered by SyntheticCorpus. Each strategy
generates valid provider-format data from annotated JSON schemas,
replacing the previous hand-crafted builders.

The strategies maintain the same public API as the original module so
existing test imports continue to work unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from hypothesis import strategies as st

from polylogue.core.json import JSONDocument, json_document
from polylogue.core.json import loads as json_loads
from polylogue.scenarios import CorpusSpec
from polylogue.schemas.synthetic import SyntheticCorpus

# =============================================================================
# Internal helpers
# =============================================================================


def _corpus_for(provider: str) -> SyntheticCorpus:
    """Get or create a SyntheticCorpus for a provider (cached per provider)."""
    if provider not in _CORPUS_CACHE:
        _CORPUS_CACHE[provider] = SyntheticCorpus.from_spec(CorpusSpec.for_provider(provider))
    return _CORPUS_CACHE[provider]


_CORPUS_CACHE: dict[str, SyntheticCorpus] = {}
_JSONL_PROVIDERS = frozenset({"claude-code", "codex"})
_PROVIDER_HINT_PATHS: dict[str, tuple[str, ...]] = {
    "chatgpt": ("chatgpt-export.json", "exports/chatgpt/session.json"),
    "claude-ai": ("claude-export.json", "exports/claude/session.json"),
    "claude-code": ("claude-code-session.jsonl", "exports/claude_code/session.jsonl"),
    "codex": ("codex-session.jsonl", "exports/codex/session.jsonl"),
    "gemini": ("gemini-export.json", "exports/gemini/session.json"),
}


ProviderPayload = JSONDocument | list[JSONDocument]


class ProviderSourceCase(TypedDict):
    provider: str
    raw: bytes
    payload: ProviderPayload
    path: Path


def _json_document_from_bytes(raw: bytes) -> JSONDocument:
    return json_document(json_loads(raw))


def _jsonl_documents(raw: bytes) -> list[JSONDocument]:
    return [json_document(json_loads(line)) for line in raw.decode().splitlines() if line.strip()]


def decode_provider_payload(provider: str, raw: bytes) -> ProviderPayload:
    """Decode schema-generated provider bytes into the wire payload type."""
    if provider in _JSONL_PROVIDERS:
        return _jsonl_documents(raw)
    return _json_document_from_bytes(raw)


# =============================================================================
# ChatGPT Strategies
# =============================================================================


@st.composite
def chatgpt_export_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 10,
) -> JSONDocument:
    """Generate a complete ChatGPT export structure from schema.

    Returns a dict with id, title, mapping (UUID→node), create_time, etc.
    """
    corpus = _corpus_for("chatgpt")
    seed = draw(st.integers(min_value=0, max_value=2**32))
    n = draw(st.integers(min_value=max(min_messages, 2), max_value=max_messages))
    raw = corpus.generate(count=1, messages_per_conversation=range(n, n + 1), seed=seed)[0]
    return _json_document_from_bytes(raw)


@st.composite
def chatgpt_message_node_strategy(
    draw: st.DrawFn,
    with_children: bool = True,
) -> JSONDocument:
    """Generate a single ChatGPT mapping node.

    Extracts a random node from a generated export, so the node has
    valid schema-driven field values and realistic structure.
    """
    export = draw(chatgpt_export_strategy(min_messages=2, max_messages=6))
    nodes = [json_document(node) for node in json_document(export.get("mapping")).values()]
    if not nodes:
        # Fallback — shouldn't happen with valid generation
        return {"id": "fallback", "message": None, "children": []}
    idx = draw(st.integers(min_value=0, max_value=len(nodes) - 1))
    return nodes[idx]


# =============================================================================
# Claude AI Strategies
# =============================================================================


@st.composite
def claude_ai_export_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 10,
) -> JSONDocument:
    """Generate a complete Claude AI export structure from schema."""
    corpus = _corpus_for("claude-ai")
    seed = draw(st.integers(min_value=0, max_value=2**32))
    n = draw(st.integers(min_value=max(min_messages, 2), max_value=max_messages))
    raw = corpus.generate(count=1, messages_per_conversation=range(n, n + 1), seed=seed)[0]
    return _json_document_from_bytes(raw)


# =============================================================================
# Claude Code Strategies
# =============================================================================


@st.composite
def claude_code_message_strategy(draw: st.DrawFn) -> JSONDocument:
    """Generate a single Claude Code JSONL message entry."""
    session = draw(claude_code_session_strategy(min_messages=2, max_messages=6))
    if not session:
        return {"type": "user", "uuid": "fallback", "message": {"content": []}}
    idx = draw(st.integers(min_value=0, max_value=len(session) - 1))
    return session[idx]


@st.composite
def claude_code_session_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 10,
) -> list[JSONDocument]:
    """Generate a complete Claude Code JSONL session (list of messages)."""
    corpus = _corpus_for("claude-code")
    seed = draw(st.integers(min_value=0, max_value=2**32))
    n = draw(st.integers(min_value=max(min_messages, 2), max_value=max_messages))
    raw = corpus.generate(count=1, messages_per_conversation=range(n, n + 1), seed=seed)[0]
    return _jsonl_documents(raw)


# =============================================================================
# Gemini Strategies
# =============================================================================


@st.composite
def gemini_export_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 10,
) -> JSONDocument:
    """Generate a complete Gemini export structure from schema."""
    corpus = _corpus_for("gemini")
    seed = draw(st.integers(min_value=0, max_value=2**32))
    n = draw(st.integers(min_value=max(min_messages, 2), max_value=max_messages))
    raw = corpus.generate(count=1, messages_per_conversation=range(n, n + 1), seed=seed)[0]
    return _json_document_from_bytes(raw)


@st.composite
def gemini_message_strategy(draw: st.DrawFn) -> JSONDocument:
    """Generate a single Gemini chunk/message entry."""
    export = draw(gemini_export_strategy(min_messages=2, max_messages=6))
    chunks = json_document(export.get("chunkedPrompt")).get("chunks")
    messages = [json_document(message) for message in chunks] if isinstance(chunks, list) else []
    if not messages:
        return {"role": "user", "text": "test"}
    idx = draw(st.integers(min_value=0, max_value=len(messages) - 1))
    return messages[idx]


# =============================================================================
# Codex Strategies
# =============================================================================


@st.composite
def codex_message_strategy(draw: st.DrawFn) -> JSONDocument:
    """Generate a single Codex message entry."""
    session = draw(codex_session_strategy(min_messages=2, max_messages=6))
    if not session:
        return {"type": "message", "role": "user", "content": []}
    idx = draw(st.integers(min_value=0, max_value=len(session) - 1))
    return session[idx]


@st.composite
def codex_session_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 10,
    use_envelope: bool = True,
) -> list[JSONDocument]:
    """Generate a Codex JSONL session (list of message records).

    Args:
        use_envelope: If True, wrap messages in the envelope format
            (session_meta + response_item wrappers). If False, return
            direct message records.
    """
    corpus = _corpus_for("codex")
    seed = draw(st.integers(min_value=0, max_value=2**32))
    n = draw(st.integers(min_value=max(min_messages, 2), max_value=max_messages))
    raw = corpus.generate(count=1, messages_per_conversation=range(n, n + 1), seed=seed)[0]
    records = _jsonl_documents(raw)

    if use_envelope and records:
        # Wrap in the envelope format the parser also supports
        import uuid as _uuid

        envelope: list[JSONDocument] = [
            {
                "type": "session_meta",
                "payload": {
                    "id": str(_uuid.UUID(int=draw(st.integers(0, 2**128 - 1)), version=4)),
                    "timestamp": records[0].get("timestamp", "2024-01-01T00:00:00Z"),
                },
            },
        ]
        for rec in records:
            envelope.append({"type": "response_item", "payload": rec})
        return envelope

    return records


# =============================================================================
# Rich semantic fixture strategies
# =============================================================================


_CHATGPT_SEMANTIC_MESSAGES: tuple[JSONDocument, ...] = (
    {
        "id": "chatgpt-text",
        "author": {"role": "user"},
        "create_time": 1700000000.0,
        "content": {"content_type": "text", "parts": ["hello", {"text": "world"}]},
        "metadata": {"model_slug": "gpt-4o"},
    },
    {
        "id": "chatgpt-code",
        "author": {"role": "assistant"},
        "create_time": 1700000001.0,
        "content": {"content_type": "code", "text": "print('ok')", "language": "python"},
        "metadata": {"model_slug": "gpt-4.1"},
    },
    {
        "id": "chatgpt-browse",
        "author": {"role": "assistant"},
        "create_time": 1700000002.0,
        "content": {"content_type": "tether_browsing_display", "parts": ["search result"]},
        "metadata": {},
    },
)

_CLAUDE_AI_SEMANTIC_MESSAGES: tuple[JSONDocument, ...] = (
    {
        "uuid": "claude-ai-human",
        "text": "question",
        "sender": "human",
        "created_at": "2025-01-01T00:00:00Z",
    },
    {
        "uuid": "claude-ai-assistant",
        "text": "answer",
        "sender": "assistant",
        "created_at": "2025-01-01T00:00:01Z",
        "attachments": [{"file_name": "spec.pdf"}],
        "files": [{"file_name": "result.txt"}],
    },
)

_CLAUDE_CODE_SEMANTIC_RECORDS: tuple[JSONDocument, ...] = (
    {
        "type": "assistant",
        "uuid": "claude-code-rich",
        "timestamp": 1700000000000,
        "message": {
            "role": "assistant",
            "model": "claude-3-opus",
            "usage": {"input_tokens": 10, "output_tokens": 12},
            "content": [
                {"type": "text", "text": "answer"},
                {"type": "thinking", "thinking": "reason"},
                {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"path": "README.md"}},
                {"type": "tool_result", "tool_use_id": "tool-1", "content": "done"},
                {"type": "code", "text": "print('ok')", "language": "python"},
            ],
        },
        "costUSD": 0.12,
        "durationMs": 42,
    },
    {
        "type": "user",
        "uuid": "claude-code-user",
        "timestamp": "2025-01-01T00:00:00Z",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "question"}],
        },
    },
)

_GEMINI_SEMANTIC_MESSAGES: tuple[JSONDocument, ...] = (
    {
        "role": "model",
        "text": "thinking message",
        "isThought": True,
        "thinkingBudget": 256,
        "tokenCount": 64,
        "parts": [{"text": "thinking message"}],
    },
    {
        "role": "model",
        "text": "",
        "parts": [{"text": "inline text"}, {"fileData": {"mimeType": "image/png"}}],
        "executableCode": {"language": "python", "code": "print('ok')"},
        "codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "ok"},
        "tokenCount": 32,
    },
)

_CODEX_SEMANTIC_RECORDS: tuple[JSONDocument, ...] = (
    {
        "type": "message",
        "id": "codex-direct",
        "role": "assistant",
        "timestamp": "2025-01-01T00:00:00Z",
        "content": [
            {"type": "output_text", "output_text": "answer"},
            {"type": "code", "code": "print('ok')", "language": "python"},
            {"type": "misc", "text": "fallback"},
        ],
    },
    {
        "type": "response_item",
        "id": "codex-envelope",
        "timestamp": "2025-01-01T00:00:01Z",
        "payload": {
            "role": "user",
            "content": [
                {"type": "input_text", "input_text": "question"},
                {"type": "text", "text": "more context"},
            ],
        },
    },
)


@st.composite
def chatgpt_semantic_message_strategy(draw: st.DrawFn) -> JSONDocument:
    """Generate ChatGPT messages with richer semantic content variants."""
    return dict(draw(st.sampled_from(_CHATGPT_SEMANTIC_MESSAGES)))


@st.composite
def claude_ai_semantic_message_strategy(draw: st.DrawFn) -> JSONDocument:
    """Generate Claude AI messages with stable semantic metadata variants."""
    return dict(draw(st.sampled_from(_CLAUDE_AI_SEMANTIC_MESSAGES)))


@st.composite
def claude_code_semantic_record_strategy(draw: st.DrawFn) -> JSONDocument:
    """Generate Claude Code records with text/thinking/tool/code coverage."""
    return dict(draw(st.sampled_from(_CLAUDE_CODE_SEMANTIC_RECORDS)))


@st.composite
def gemini_semantic_message_strategy(draw: st.DrawFn) -> JSONDocument:
    """Generate Gemini messages exercising thought/code/file branches."""
    return dict(draw(st.sampled_from(_GEMINI_SEMANTIC_MESSAGES)))


@st.composite
def codex_semantic_record_strategy(draw: st.DrawFn) -> JSONDocument:
    """Generate Codex direct and envelope records with mixed block types."""
    raw = draw(st.sampled_from(_CODEX_SEMANTIC_RECORDS))
    return json_document(json_loads(json.dumps(raw)))


@st.composite
def provider_semantic_case_strategy(
    draw: st.DrawFn,
    providers: tuple[str, ...] = ("chatgpt", "claude-ai", "claude-code", "codex", "gemini"),
) -> tuple[str, JSONDocument]:
    """Generate provider/raw pairs for semantically rich viewport laws."""
    provider = draw(st.sampled_from(providers))
    raw = {
        "chatgpt": draw(chatgpt_semantic_message_strategy()),
        "claude-ai": draw(claude_ai_semantic_message_strategy()),
        "claude-code": draw(claude_code_semantic_record_strategy()),
        "codex": draw(codex_semantic_record_strategy()),
        "gemini": draw(gemini_semantic_message_strategy()),
    }[provider]
    return provider, raw


# =============================================================================
# Generic schema-driven strategy (for new tests)
# =============================================================================


@st.composite
def provider_export_strategy(
    draw: st.DrawFn,
    provider: str,
    min_msgs: int = 1,
    max_msgs: int = 10,
) -> bytes:
    """Schema-driven strategy for any provider.

    Returns raw wire-format bytes suitable for feeding to a parser.
    """
    corpus = _corpus_for(provider)
    seed = draw(st.integers(min_value=0, max_value=2**32))
    n = draw(st.integers(min_value=max(min_msgs, 2), max_value=max_msgs))
    return corpus.generate(count=1, messages_per_conversation=range(n, n + 1), seed=seed)[0]


@st.composite
def provider_payload_strategy(
    draw: st.DrawFn,
    provider: str,
    min_msgs: int = 1,
    max_msgs: int = 10,
) -> ProviderPayload:
    """Generate and decode a provider wire payload."""
    raw = draw(provider_export_strategy(provider, min_msgs=min_msgs, max_msgs=max_msgs))
    return decode_provider_payload(provider, raw)


@st.composite
def provider_payload_case_strategy(
    draw: st.DrawFn,
    providers: tuple[str, ...] = ("chatgpt", "claude-ai", "claude-code", "codex", "gemini"),
    min_msgs: int = 1,
    max_msgs: int = 10,
) -> tuple[str, ProviderPayload]:
    """Generate `(provider, payload)` pairs for dispatch/harmonization laws."""
    provider = draw(st.sampled_from(providers))
    payload = draw(provider_payload_strategy(provider, min_msgs=min_msgs, max_msgs=max_msgs))
    return provider, payload


@st.composite
def provider_hint_path_strategy(draw: st.DrawFn, provider: str) -> Path:
    """Generate representative filename/path hints for a provider."""
    return Path(draw(st.sampled_from(_PROVIDER_HINT_PATHS[provider])))


@st.composite
def provider_source_case_strategy(
    draw: st.DrawFn,
    providers: tuple[str, ...] = ("chatgpt", "claude-ai", "claude-code", "codex", "gemini"),
    min_msgs: int = 1,
    max_msgs: int = 10,
) -> ProviderSourceCase:
    """Generate a provider payload together with a representative hint path."""
    provider = draw(st.sampled_from(providers))
    raw = draw(provider_export_strategy(provider, min_msgs=min_msgs, max_msgs=max_msgs))
    payload = decode_provider_payload(provider, raw)
    path = Path(draw(st.sampled_from(_PROVIDER_HINT_PATHS[provider])))
    return {"provider": provider, "raw": raw, "payload": payload, "path": path}
