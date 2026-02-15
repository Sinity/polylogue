"""Hypothesis strategies for provider-specific export formats.

Schema-driven strategies powered by SyntheticCorpus. Each strategy
generates valid provider-format data from annotated JSON schemas,
replacing the previous hand-crafted builders.

The strategies maintain the same public API as the original module so
existing test imports continue to work unchanged.
"""

from __future__ import annotations

import json
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
    corpus = _corpus_for(provider)
    seed = draw(st.integers(min_value=0, max_value=2**32))
    n = draw(st.integers(min_value=max(min_msgs, 2), max_value=max_msgs))
    return corpus.generate(count=1, messages_per_conversation=range(n, n + 1), seed=seed)[0]
