"""Branch type classification for conversation hierarchies."""

from __future__ import annotations

from enum import Enum


class BranchType(str, Enum):
    """Classification for how a conversation relates to its parent."""

    CONTINUATION = "continuation"
    SIDECHAIN = "sidechain"
    FORK = "fork"
    SUBAGENT = "subagent"

    def __str__(self) -> str:
        return self.value


__all__ = ["BranchType"]
