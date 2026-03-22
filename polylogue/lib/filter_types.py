from __future__ import annotations

from typing import Literal

SortField = Literal["date", "tokens", "messages", "words", "longest", "random"]

__all__ = ["SortField"]
