"""Fluent projection builder for conversation filtering and transformation.

Provides lazy-evaluated, composable projections over conversations.

Example usage:
    # Fluent composition
    result = conv.project()
        .substantive()
        .min_words(50)
        .since(datetime(2024, 1, 1))
        .limit(10)
        .execute()

    # Lazy iteration (memory efficient)
    for msg in conv.project().user_messages().contains("error").iter():
        print(msg.text)

    # Count without materialization
    error_count = conv.project().contains("error").count()
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from polylogue.lib.projection_filter_mixin import ProjectionFilterMixin
from polylogue.lib.projection_terminal_mixin import ProjectionTerminalMixin
from polylogue.lib.projection_transform_mixin import ProjectionTransformMixin

if TYPE_CHECKING:
    from polylogue.lib.conversation_models import Conversation
    from polylogue.lib.message_models import Message


class ConversationProjection(ProjectionFilterMixin, ProjectionTransformMixin, ProjectionTerminalMixin):
    """Lazy-evaluated projection builder for conversations.

    Filters and transforms are accumulated and only applied when
    a terminal operation (execute, iter, count, etc.) is called.
    """

    def __init__(self, conversation: Conversation) -> None:
        self._conv = conversation
        self._filters: list[Callable[[Message], bool]] = []
        self._transforms: list[Callable[[Message], Message]] = []
        self._limit: int | None = None
        self._offset: int = 0
        self._reverse: bool = False


__all__ = ["ConversationProjection"]
