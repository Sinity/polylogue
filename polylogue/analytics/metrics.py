"""Analytics metrics computation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.lib.repository import ConversationRepository


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


def compute_provider_comparison(db_path: Path | None = None) -> list[ProviderMetrics]:
    """Compute comparison metrics across all providers.

    Args:
        db_path: Optional path to database file. If None, uses default location.

    Returns:
        List of ProviderMetrics ordered by conversation count (descending)

    Examples:
        >>> metrics = compute_provider_comparison()
        >>> for m in metrics:
        ...     print(f"{m.provider_name}: {m.conversation_count} conversations")
    """
    from polylogue.storage.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(db_path=db_path)
    repo = ConversationRepository(backend=backend)

    # Aggregate metrics by provider
    provider_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "conversation_count": 0,
            "message_count": 0,
            "user_message_count": 0,
            "assistant_message_count": 0,
            "user_word_sum": 0,
            "assistant_word_sum": 0,
            "tool_use_count": 0,
            "thinking_count": 0,
            "conversations_with_tools": set(),
            "conversations_with_thinking": set(),
        }
    )

    # Iterate all conversations (fetch in batches of 1000)
    offset = 0
    batch_size = 1000
    while True:
        batch = repo.list(limit=batch_size, offset=offset)
        if not batch:
            break

        for conv in batch:
            provider = conv.provider or "unknown"
            data = provider_data[provider]

            data["conversation_count"] += 1

            # Analyze messages
            for msg in conv.messages:
                data["message_count"] += 1

                if msg.role == "user":
                    data["user_message_count"] += 1
                    data["user_word_sum"] += msg.word_count
                elif msg.role == "assistant":
                    data["assistant_message_count"] += 1
                    data["assistant_word_sum"] += msg.word_count

                # Check for tool use and thinking
                if msg.is_tool_use:
                    data["tool_use_count"] += 1
                    data["conversations_with_tools"].add(conv.id)

                if msg.is_thinking:
                    data["thinking_count"] += 1
                    data["conversations_with_thinking"].add(conv.id)

        offset += batch_size

    # Convert to ProviderMetrics objects
    results: list[ProviderMetrics] = []
    for provider_name, data in provider_data.items():
        conv_count = data["conversation_count"]
        user_count = data["user_message_count"]
        asst_count = data["assistant_message_count"]

        avg_msgs = data["message_count"] / conv_count if conv_count > 0 else 0.0
        avg_user_words = data["user_word_sum"] / user_count if user_count > 0 else 0.0
        avg_asst_words = data["assistant_word_sum"] / asst_count if asst_count > 0 else 0.0

        results.append(
            ProviderMetrics(
                provider_name=provider_name,
                conversation_count=conv_count,
                message_count=data["message_count"],
                user_message_count=user_count,
                assistant_message_count=asst_count,
                avg_messages_per_conversation=avg_msgs,
                avg_user_words=avg_user_words,
                avg_assistant_words=avg_asst_words,
                tool_use_count=data["tool_use_count"],
                thinking_count=data["thinking_count"],
                total_conversations_with_tools=len(data["conversations_with_tools"]),
                total_conversations_with_thinking=len(data["conversations_with_thinking"]),
            )
        )

    # Sort by conversation count (descending)
    results.sort(key=lambda m: m.conversation_count, reverse=True)

    return results
