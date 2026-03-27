"""Canonical action-event models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from polylogue.lib.viewports import ToolCategory
from polylogue.types import Provider


@dataclass(frozen=True, slots=True)
class ActionEvent:
    """Normalized semantic event derived from a message tool call."""

    event_id: str
    message_id: str
    timestamp: datetime | None
    sequence_index: int
    kind: ToolCategory
    tool_name: str
    tool_id: str | None
    provider: Provider | None
    affected_paths: tuple[str, ...]
    cwd_path: str | None
    branch_names: tuple[str, ...]
    command: str | None
    query: str | None
    url: str | None
    output_text: str | None
    search_text: str
    raw: dict[str, Any]

    @property
    def normalized_tool_name(self) -> str:
        return (self.tool_name or "unknown").strip().lower()


__all__ = ["ActionEvent"]
