"""Typed decision semantic models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """An explicit decision point in a conversation."""

    index: int
    summary: str
    confidence: float
    context: str


__all__ = ["Decision"]
