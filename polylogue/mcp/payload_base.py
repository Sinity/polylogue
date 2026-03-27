"""Base MCP payload model contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, RootModel


def normalize_role(role: object) -> str:
    if not role:
        return "unknown"
    if hasattr(role, "value"):
        role = role.value
    return str(role)


class MCPPayload(BaseModel):
    """Base model for JSON payloads returned by MCP surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json(self, *, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=2, exclude_none=exclude_none)


class MCPRootPayload(RootModel[Any]):
    """Root-model variant for list/map payloads."""

    def to_json(self, *, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=2, exclude_none=exclude_none)


__all__ = ["MCPRootPayload", "MCPPayload", "normalize_role"]
