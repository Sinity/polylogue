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

    async with backend._get_connection() as conn:
        cursor = await conn.execute("""
            SELECT
                c.provider_name,
                COUNT(DISTINCT c.conversation_id) AS conversation_count,
                COUNT(m.message_id) AS message_count,
                SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) AS user_message_count,
                SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) AS assistant_message_count,
                SUM(CASE WHEN m.role = 'user' AND m.text IS NOT NULL AND TRIM(m.text) != ''
                    THEN LENGTH(TRIM(m.text)) - LENGTH(REPLACE(TRIM(m.text), ' ', '')) + 1
                    ELSE 0 END) AS user_word_sum,
                SUM(CASE WHEN m.role = 'assistant' AND m.text IS NOT NULL AND TRIM(m.text) != ''
                    THEN LENGTH(TRIM(m.text)) - LENGTH(REPLACE(TRIM(m.text), ' ', '')) + 1
                    ELSE 0 END) AS assistant_word_sum,
                SUM(CASE WHEN m.provider_meta LIKE '%"type":"tool_use"%'
                         OR m.role = 'tool'
                    THEN 1 ELSE 0 END) AS tool_use_count,
                SUM(CASE WHEN m.provider_meta LIKE '%"type":"thinking"%'
                    THEN 1 ELSE 0 END) AS thinking_count,
                COUNT(DISTINCT CASE
                    WHEN m.provider_meta LIKE '%"type":"tool_use"%'
                         OR m.role = 'tool'
                    THEN c.conversation_id END) AS conversations_with_tools,
                COUNT(DISTINCT CASE
                    WHEN m.provider_meta LIKE '%"type":"thinking"%'
                    THEN c.conversation_id END) AS conversations_with_thinking
            FROM conversations c
            LEFT JOIN messages m ON c.conversation_id = m.conversation_id
            GROUP BY c.provider_name
            ORDER BY conversation_count DESC
        """)
        rows = await cursor.fetchall()

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
