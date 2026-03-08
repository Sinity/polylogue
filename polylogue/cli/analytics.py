"""Analytics metrics computation.

Uses SQL aggregation to compute provider-level metrics without loading
all conversations into memory. This is critical for large databases
(tens of GB) where in-Python iteration would be prohibitively slow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProviderMetrics:
    """Metrics for a single provider."""

    provider_name: str
    conversation_count: int
    message_count: int
    user_message_count: int
    assistant_message_count: int
    avg_messages_per_conversation: float
    avg_user_words: float
    avg_assistant_words: float
    tool_use_count: int
    thinking_count: int
    total_conversations_with_tools: int
    total_conversations_with_thinking: int

    @property
    def tool_use_percentage(self) -> float:
        """Percentage of conversations with tool use."""
        if self.conversation_count == 0:
            return 0.0
        return (self.total_conversations_with_tools / self.conversation_count) * 100

    @property
    def thinking_percentage(self) -> float:
        """Percentage of conversations with thinking traces."""
        if self.conversation_count == 0:
            return 0.0
        return (self.total_conversations_with_thinking / self.conversation_count) * 100


async def compute_provider_comparison(db_path: Path | None = None) -> list[ProviderMetrics]:
    """Compute comparison metrics across all providers using SQL aggregation.

    Args:
        db_path: Optional path to database file. If None, uses default location.

    Returns:
        List of ProviderMetrics ordered by conversation count (descending)

    Examples:
        >>> metrics = await compute_provider_comparison()
        >>> for m in metrics:
        ...     print(f"{m.provider_name}: {m.conversation_count} conversations")
    """
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

    backend = SQLiteBackend(db_path=db_path)
    try:
        rows = await backend.get_provider_metrics_rows()
    finally:
        await backend.close()

    results: list[ProviderMetrics] = []
    for row in rows:
        provider_name = row["provider_name"] or "unknown"
        conv_count = row["conversation_count"]
        msg_count = row["message_count"]
        user_count = row["user_message_count"]
        asst_count = row["assistant_message_count"]
        user_word_sum = row["user_word_sum"] or 0
        asst_word_sum = row["assistant_word_sum"] or 0

        avg_msgs = msg_count / conv_count if conv_count > 0 else 0.0
        avg_user_words = user_word_sum / user_count if user_count > 0 else 0.0
        avg_asst_words = asst_word_sum / asst_count if asst_count > 0 else 0.0

        results.append(
            ProviderMetrics(
                provider_name=provider_name,
                conversation_count=conv_count,
                message_count=msg_count,
                user_message_count=user_count,
                assistant_message_count=asst_count,
                avg_messages_per_conversation=avg_msgs,
                avg_user_words=avg_user_words,
                avg_assistant_words=avg_asst_words,
                tool_use_count=row["tool_use_count"],
                thinking_count=row["thinking_count"],
                total_conversations_with_tools=row["conversations_with_tools"],
                total_conversations_with_thinking=row["conversations_with_thinking"],
            )
        )

    return results
