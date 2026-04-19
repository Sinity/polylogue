"""Wire format configuration for provider export formats.

Describes HOW each provider structures their export data — encoding type,
tree vs. linear vs. JSONL, and message location paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

WireEncoding: TypeAlias = Literal["json", "jsonl"]


@dataclass(frozen=True)
class TreeConfig:
    """Configuration for tree-structured message formats."""

    container_path: str | None = None  # Top-level key containing the tree dict
    key_field: str = "id"
    parent_field: str = "parent"
    children_field: str | None = None
    session_field: str | None = None


@dataclass(frozen=True)
class WireFormat:
    """Wire format configuration for a provider's export format."""

    encoding: WireEncoding
    tree: TreeConfig | None = None
    messages_path: str | None = None  # Dot-path to messages array


# Per-provider wire format configs — the only manual piece (~50 lines).
# Describes HOW the format is structured, not WHAT conversations say.
PROVIDER_WIRE_FORMATS: dict[str, WireFormat] = {
    "chatgpt": WireFormat(
        encoding="json",
        tree=TreeConfig(
            container_path="mapping",
            key_field="id",
            parent_field="parent",
            children_field="children",
        ),
    ),
    "claude-code": WireFormat(
        encoding="jsonl",
        tree=TreeConfig(
            key_field="uuid",
            parent_field="parentUuid",
            session_field="sessionId",
        ),
    ),
    "claude-ai": WireFormat(
        encoding="json",
        messages_path="chat_messages",
    ),
    "codex": WireFormat(
        encoding="jsonl",
    ),
    "gemini": WireFormat(
        encoding="json",
        messages_path="chunkedPrompt.chunks",
    ),
}


__all__ = [
    "PROVIDER_WIRE_FORMATS",
    "TreeConfig",
    "WireEncoding",
    "WireFormat",
]
