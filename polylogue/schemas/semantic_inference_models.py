"""Typed semantic-role inference models."""

from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.schemas.json_types import JSONDocument

KNOWN_ROLE_VALUES = frozenset(
    {
        "user",
        "assistant",
        "system",
        "model",
        "tool",
        "human",
        "developer",
        "function",
        "admin",
    }
)

ANTI_TITLE_NAME_TOKENS = frozenset(
    {
        "uuid",
        "model",
        "version",
        "hash",
        "key",
        "token",
        "ref",
        "parent",
        "slug",
        "path",
        "config",
        "setting",
    }
)

SEMANTIC_ROLES = (
    "message_container",
    "message_role",
    "message_body",
    "message_timestamp",
    "conversation_title",
)


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    """A scored candidate for a semantic role assignment."""

    path: str
    role: str
    confidence: float
    evidence: JSONDocument = field(default_factory=dict)


__all__ = [
    "ANTI_TITLE_NAME_TOKENS",
    "KNOWN_ROLE_VALUES",
    "SEMANTIC_ROLES",
    "SemanticCandidate",
]
