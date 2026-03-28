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
from typing import Any

from hypothesis import strategies as st

from polylogue.schemas.synthetic import SyntheticCorpus

# =============================================================================
# Internal helpers
# =============================================================================


def _corpus_for(provider: str) -> SyntheticCorpus:
    """Get or create a SyntheticCorpus for a provider (cached per provider)."""
    if provider not in _CORPUS_CACHE:
        _CORPUS_CACHE[provider] = SyntheticCorpus.for_provider(provider)
    return _CORPUS_CACHE[provider]


_CORPUS_CACHE: dict[str, SyntheticCorpus] = {}
_JSONL_PROVIDERS = frozenset({"claude-code", "codex"})
_PROVIDER_HINT_PATHS: dict[str, tuple[str, ...]] = {
    "chatgpt": ("chatgpt-export.json", "exports/chatgpt/session.json"),
    "claude": ("claude-export.json", "exports/claude/session.json"),
    "claude-ai": ("claude-export.json", "exports/claude/session.json"),
    "claude-code": ("claude-code-session.jsonl", "exports/claude_code/session.jsonl"),
    "codex": ("codex-session.jsonl", "exports/codex/session.jsonl"),
    "gemini": ("gemini-export.json", "exports/gemini/session.json"),
}
_CORPUS_PROVIDER_ALIASES = {
    "claude": "claude-ai",
}


def _corpus_provider(provider: str) -> str:
    """Map runtime provider names to synthetic-corpus provider names."""
    return _CORPUS_PROVIDER_ALIASES.get(provider, provider)


def decode_provider_payload(provider: str, raw: bytes) -> Any:
    """Decode schema-generated provider bytes into the wire payload type."""
    if provider in _JSONL_PROVIDERS:
        return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return json.loads(raw)


# =============================================================================
# ChatGPT Strategies
# =============================================================================


@st.composite
def chatgpt_export_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 10,
) -> dict[str, Any]:
    """Generate a complete ChatGPT export structure from schema.

    Returns a dict with id, title, mapping (UUID→node), create_time, etc.
    """
    corpus = _corpus_for("chatgpt")
    seed = draw(st.integers(min_value=0, max_value=2**32))
    n = draw(st.integers(min_value=max(min_messages, 2), max_value=max_messages))
    raw = corpus.generate(count=1, messages_per_conversation=range(n, n + 1), seed=seed)[0]
    return json.loads(raw)


@st.composite
def chatgpt_message_node_strategy(
    draw: st.DrawFn,
    with_children: bool = True,
) -> dict[str, Any]:
    """Generate a single ChatGPT mapping node.

    Extracts a random node from a generated export, so the node has
    valid schema-driven field values and realistic structure.
    """
    export = draw(chatgpt_export_strategy(min_messages=2, max_messages=6))
    nodes = list(export.get("mapping", {}).values())
    if not nodes:
        # Fallback — shouldn't happen with valid generation
        return {"id": "fallback", "message": None, "children": []}
    idx = draw(st.integers(min_value=0, max_value=len(nodes) - 1))
    return nodes[idx]


# =============================================================================
# Claude AI Strategies
# =============================================================================


@st.composite
def claude_ai_message_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a single Claude AI chat_messages entry."""
    export = draw(claude_ai_export_strategy(min_messages=2, max_messages=6))
    messages = export.get("chat_messages", [])
    if not messages:
        return {"uuid": "fallback", "sender": "human", "text": "test"}
    idx = draw(st.integers(min_value=0, max_value=len(messages) - 1))
    return messages[idx]


@st.composite
def claude_ai_export_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 10,
) -> dict[str, Any]:
    """Generate a complete Claude AI export structure from schema."""
    corpus = _corpus_for("claude-ai")
    seed = draw(st.integers(min_value=0, max_value=2**32))
    n = draw(st.integers(min_value=max(min_messages, 2), max_value=max_messages))
    raw = corpus.generate(count=1, messages_per_conversation=range(n, n + 1), seed=seed)[0]
    return json.loads(raw)


# =============================================================================
# Claude Code Strategies
# =============================================================================


@st.composite
def claude_code_message_strategy(draw: st.DrawFn) -> dict[str, Any]:
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
) -> list[dict[str, Any]]:
    """Generate a complete Claude Code JSONL session (list of messages)."""
    corpus = _corpus_for("claude-code")
    seed = draw(st.integers(min_value=0, max_value=2**32))
    n = draw(st.integers(min_value=max(min_messages, 2), max_value=max_messages))
    raw = corpus.generate(count=1, messages_per_conversation=range(n, n + 1), seed=seed)[0]
    return [json.loads(line) for line in raw.decode().strip().split("\n") if line.strip()]


# =============================================================================
# Gemini Strategies
# =============================================================================


@st.composite
def gemini_export_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 10,
) -> dict[str, Any]:
    """Generate a complete Gemini export structure from schema."""
    corpus = _corpus_for("gemini")
    seed = draw(st.integers(min_value=0, max_value=2**32))
    n = draw(st.integers(min_value=max(min_messages, 2), max_value=max_messages))
    raw = corpus.generate(count=1, messages_per_conversation=range(n, n + 1), seed=seed)[0]
    return json.loads(raw)


@st.composite
def gemini_message_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a single Gemini chunk/message entry."""
    export = draw(gemini_export_strategy(min_messages=2, max_messages=6))
    messages = export.get("chunkedPrompt", {}).get("chunks", [])
    if not messages:
        return {"role": "user", "text": "test"}
    idx = draw(st.integers(min_value=0, max_value=len(messages) - 1))
    return messages[idx]


# =============================================================================
# Codex Strategies
# =============================================================================


@st.composite
def codex_message_strategy(draw: st.DrawFn) -> dict[str, Any]:
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
) -> list[dict[str, Any]]:
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
    records = [json.loads(line) for line in raw.decode().strip().split("\n") if line.strip()]

    if use_envelope and records:
        # Wrap in the envelope format the parser also supports
        import uuid as _uuid

        envelope: list[dict[str, Any]] = [
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
    corpus = _corpus_for(_corpus_provider(provider))
    seed = draw(st.integers(min_value=0, max_value=2**32))
    n = draw(st.integers(min_value=max(min_msgs, 2), max_value=max_msgs))
    return corpus.generate(count=1, messages_per_conversation=range(n, n + 1), seed=seed)[0]


@st.composite
def provider_payload_strategy(
    draw: st.DrawFn,
    provider: str,
    min_msgs: int = 1,
    max_msgs: int = 10,
) -> Any:
    """Generate and decode a provider wire payload."""
    raw = draw(provider_export_strategy(provider, min_msgs=min_msgs, max_msgs=max_msgs))
    return decode_provider_payload(provider, raw)


@st.composite
def provider_payload_case_strategy(
    draw: st.DrawFn,
    providers: tuple[str, ...] = ("chatgpt", "claude", "claude-code", "codex", "gemini"),
    min_msgs: int = 1,
    max_msgs: int = 10,
) -> tuple[str, Any]:
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
) -> dict[str, Any]:
    """Generate a provider payload together with a representative hint path."""
    provider = draw(st.sampled_from(providers))
    raw = draw(provider_export_strategy(provider, min_msgs=min_msgs, max_msgs=max_msgs))
    payload = decode_provider_payload(provider, raw)
    path = Path(draw(st.sampled_from(_PROVIDER_HINT_PATHS[provider])))
    return {"provider": provider, "raw": raw, "payload": payload, "path": path}
