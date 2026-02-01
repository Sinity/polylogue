"""Hypothesis strategies for polylogue property-based testing.

This module provides composable strategies for generating test data
that matches the structure of provider exports and internal models.

Usage:
    from hypothesis import given
    from tests.strategies import chatgpt_message, conversation_strategy

    @given(chatgpt_message())
    def test_message_parsing(msg):
        ...

Strategies are organized by domain:
- messages: Message and content block strategies
- providers: Provider-specific export format strategies
- filters: Filter composition strategies for query testing
- adversarial: Malformed/attack data for security testing
"""

from tests.strategies.messages import (
    content_block_strategy,
    conversation_strategy,
    message_strategy,
    parsed_message_strategy,
    text_content_strategy,
    thinking_block_strategy,
    tool_use_block_strategy,
)

from tests.strategies.providers import (
    chatgpt_export_strategy,
    chatgpt_message_node_strategy,
    claude_ai_export_strategy,
    claude_code_message_strategy,
    codex_message_strategy,
)

from tests.strategies.filters import (
    filter_arg_strategy,
    filter_chain_strategy,
    filter_type_strategy,
)

from tests.strategies.adversarial import (
    malformed_json_strategy,
    path_traversal_strategy,
    sql_injection_strategy,
)

__all__ = [
    # Messages
    "content_block_strategy",
    "conversation_strategy",
    "message_strategy",
    "parsed_message_strategy",
    "text_content_strategy",
    "thinking_block_strategy",
    "tool_use_block_strategy",
    # Providers
    "chatgpt_export_strategy",
    "chatgpt_message_node_strategy",
    "claude_ai_export_strategy",
    "claude_code_message_strategy",
    "codex_message_strategy",
    # Filters
    "filter_arg_strategy",
    "filter_chain_strategy",
    "filter_type_strategy",
    # Adversarial
    "malformed_json_strategy",
    "path_traversal_strategy",
    "sql_injection_strategy",
]
